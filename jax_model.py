import time
import flax.linen as fnn
import jax
import jax.nn as jnn
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import optax
import random

from config import ModelConfig
from data import Enwik9Loader


class TransformerLayer(fnn.Module):
    d_model: int
    num_heads: int
    ff_dim: int
    dropout: float

    def setup(self):
        self.mha = fnn.SelfAttention(
            num_heads=self.num_heads,
            qkv_features=self.d_model,
            # dropout in the attention matrix was introduced in
            # https://arxiv.org/abs/1907.11065, it's *not* the normal thing
            # from Attention is All You Need.
            dropout_rate=0,
            deterministic=False,
        )
        self.layer_norm_1 = fnn.LayerNorm()
        self.linear_1 = fnn.Dense(features=self.ff_dim)
        self.linear_2 = fnn.Dense(features=self.d_model)
        self.layer_norm_2 = fnn.LayerNorm()
        self.dropout_layer = fnn.Dropout(self.dropout, deterministic=False)

    def __call__(
        self, x: npt.NDArray[np.float32], mask: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        # "correct" type annotations for jax DeviceArrays are numpy ndarrays
        x = x + self.dropout_layer(self.mha(self.layer_norm_1(x), mask=mask))
        x = x + self.dropout_layer(self._ff_block(self.layer_norm_2(x)))
        return x

    def _ff_block(self, x):
        x = jnn.relu(self.linear_1(x))
        x = self.dropout_layer(x)
        x = self.linear_2(x)
        return x


class LM(fnn.Module):
    cfg: ModelConfig

    def setup(self):
        self.byte_embedding = fnn.Embed(num_embeddings=256, features=self.cfg.d_model)
        self.transformer_layers = [
            TransformerLayer(
                self.cfg.d_model, self.cfg.num_heads, self.cfg.ff_dim, self.cfg.dropout
            )
            for _ in range(self.cfg.n_layers)
        ]
        self.prob_decoder = fnn.Dense(features=256)
        self.positional_encoding = self.param(
            "positional_encoding",
            jnn.initializers.normal(),
            (self.cfg.seq_len, self.cfg.d_model),
        )
        self.dropout_layer = fnn.Dropout(self.cfg.dropout, deterministic=False)

    def __call__(self, text):
        "Run the model, returning unnormalized log probabilities."
        if (
            len(text.shape) != 1
            or text.shape[0] != self.cfg.seq_len
            or text.dtype != jnp.uint8
        ):
            assert False, (
                f"expected input shape: [{self.cfg.seq_len}] dtype: uint8. "
                f"Got {text.shape}, {text.dtype}"
            )
        input_ = self.byte_embedding(text)
        # print(text.shape, mask.shape)
        # print(mask)
        # Shift input_ right so causality isn't violated
        input_ = jnp.concatenate(
            [jnp.zeros([1, self.cfg.d_model]), input_[:-1, :]], axis=0
        )
        input_ = input_ + self.positional_encoding
        input_ = self.dropout_layer(input_)

        mask = fnn.attention.make_causal_mask(text)
        for tl in self.transformer_layers:
            input_ = tl(input_, mask=mask)

        return self.prob_decoder(input_)


def compute_loss(params, model: LM, text, rng):
    model_out = model.apply(params, text=text, rngs={"dropout": rng})
    one_hots = jnn.one_hot(text, 256)
    loss = optax.softmax_cross_entropy(model_out, one_hots)
    return loss


def setup_model(rng, cfg: ModelConfig):
    model = LM(cfg)

    rng_p, rng_d = jax.random.split(rng)
    params = model.init(
        {"params": rng_p, "dropout": rng_d}, jnp.zeros([cfg.seq_len], dtype=jnp.uint8)
    )
    return params, model


def setup_optimizer(params, cfg: ModelConfig):
    optimizer = optax.adam(cfg.learning_rate)
    opt_state = optimizer.init(params)
    return optimizer, opt_state


def train_loop(
    model: LM, optimizer, opt_state, params, cfg: ModelConfig, datapath: str
):
    rng = jax.random.PRNGKey(1)

    def run_train_step(params, opt_state, text_batch, rng):
        rng, rng2 = jax.random.split(rng)
        loss, grad = jax.value_and_grad(
            lambda p: jax.vmap(
                lambda text: compute_loss(p, model, text=text, rng=rng),
                in_axes=0,
                out_axes=0,
            )(text_batch).mean()
        )(params)
        updates, opt_state = optimizer.update(grad, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, rng2

    fast_train_step = jax.jit(run_train_step, donate_argnums=[0, 1, 3])

    losses = []
    t = time.time()
    log_per = 20
    for idx, batch in enumerate(Enwik9Loader(cfg.batch_size, cfg.seq_len, datapath)):
        batch = jnp.array(batch)
        params, opt_state, loss, rng = fast_train_step(
            params, opt_state, batch, rng
        )
        losses.append(loss)

        if (idx+1) % log_per == 0:
            time_elps = time.time() - t
            speed = log_per * cfg.batch_size / time_elps
            print(f"At step {idx+1}, loss: {np.mean(losses):.4f}, Speed: {int(speed):d}")
            t = time.time()
            losses = []

    return params, opt_state


def setup_all(cfg: ModelConfig, rng=None):
    rng = (
        rng
        if rng is not None
        else jax.random.PRNGKey(random.randrange(-(2 ** 63), 2 ** 63))
    )

    params, model = setup_model(rng, cfg)
    optimizer, opt_state = setup_optimizer(params, cfg)

    return params, model, optimizer, opt_state


if __name__ == "__main__":
    from config import ModelConfig

    enwik9 = "./enwik9"
    # This is the highest batch size PyTorch can handle, the JAX model can do 79
    cfg = ModelConfig(
        seq_len=256,
        n_layers=1,
        d_model=512,
        num_heads=8,
        ff_dim=2048,
        dropout=0.1,
        batch_size=100,
        learning_rate=1e-3,
    )

    from jax_model import *

    params, model, optimizer, opt_state, = setup_all(cfg)
    params, opt_state = train_loop(model, optimizer, opt_state, params, cfg, enwik9)
