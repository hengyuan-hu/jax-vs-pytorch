import argparse
import os
import sys
import time

import flax.linen as fnn
import jax
import jax.nn as jnn
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import optax

from config import ModelConfig
from data import Enwik9Loader
from logger import Logger


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
            # this initilization is important to get learning curve similar to pytorch
            kernel_init=fnn.initializers.variance_scaling(1/3, "fan_in", "uniform"),
        )
        self.layer_norm_1 = fnn.LayerNorm(epsilon=1e-5)
        self.linear_1 = fnn.Dense(
            features=self.ff_dim,
            kernel_init=fnn.initializers.variance_scaling(1/3, "fan_in", "uniform"),
        )
        self.linear_2 = fnn.Dense(
            features=self.d_model,
            kernel_init=fnn.initializers.variance_scaling(1/3, "fan_in", "uniform"),
        )
        self.layer_norm_2 = fnn.LayerNorm(epsilon=1e-5)
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
        self.byte_embedding = fnn.Embed(
            num_embeddings=256,
            features=self.cfg.d_model,
            embedding_init=jnn.initializers.normal(),
        )
        self.positional_encoding = self.param(
            "positional_encoding",
            jnn.initializers.normal(),
            (self.cfg.seq_len, self.cfg.d_model),
        )
        self.dropout_layer = fnn.Dropout(self.cfg.dropout, deterministic=False)

        self.transformer_layers = [
            TransformerLayer(
                self.cfg.d_model, self.cfg.num_heads, self.cfg.ff_dim, self.cfg.dropout
            )
            for _ in range(self.cfg.n_layers)
        ]
        self.prob_decoder = fnn.Dense(
            features=256,
            kernel_init=fnn.initializers.variance_scaling(1/3, "fan_in", "uniform"),
        )

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
        x = self.byte_embedding(text)
        # Shift x right so causality isn't violated
        x = jnp.concatenate(
            [jnp.zeros([1, self.cfg.d_model]), x[:-1, :]], axis=0
        )
        x = x + self.positional_encoding
        x = self.dropout_layer(x)

        mask = fnn.attention.make_causal_mask(text)
        for layer in self.transformer_layers:
            x = layer(x, mask=mask)

        return self.prob_decoder(x)


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
    for idx, batch in enumerate(list(Enwik9Loader(cfg.batch_size, cfg.seq_len, datapath))):
        batch = jnp.array(batch)
        params, opt_state, loss, rng = fast_train_step(params, opt_state, batch, rng)
        losses.append(loss)

        if (idx + 1) % log_per == 0:
            jax.block_until_ready(loss)
            time_elps = time.time() - t
            speed = log_per * cfg.batch_size / time_elps
            print(
                f"At iter {idx+1}, loss: {np.mean(losses):.4f}, Speed: {int(speed):d}"
            )
            t = time.time()
            losses = []

        if (idx + 1) > cfg.max_num_batch:
            break

    return params, opt_state


def setup_all(cfg: ModelConfig, rng=None):
    rng = jax.random.PRNGKey(1)
    params, model = setup_model(rng, cfg)
    optimizer, opt_state = setup_optimizer(params, cfg)

    return params, model, optimizer, opt_state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--save_dir", type=str, default="exps/jax/run")
    parser.add_argument("--num_layer", type=int, default=1)
    args = parser.parse_args()

    args.save_dir = f"{args.save_dir}_layer{args.num_layer}"
    logger_path = os.path.join(args.save_dir, f"train.log")
    print(f"writing to {logger_path}")
    sys.stdout = Logger(logger_path, print_to_stdout=True)

    enwik9 = "./enwik9"
    cfg = ModelConfig(
        seq_len=256,
        n_layers=args.num_layer,
        d_model=512,
        num_heads=8,
        ff_dim=2048,
        dropout=0.1,
        batch_size=100,
        learning_rate=1e-3,
        max_num_batch=5000,
    )

    params, model, optimizer, opt_state = setup_all(cfg)
    # param_count = sum(x.size for x in jax.tree_leaves(params))
    params, opt_state = train_loop(model, optimizer, opt_state, params, cfg, enwik9)
