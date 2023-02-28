# %%

from jax_model import *


def setup_all(cfg: ModelConfig, rng=None):
    rng = (
        rng
        if rng is not None
        else jax.random.PRNGKey(random.randrange(-(2**63), 2**63))
    )

    params, model = setup_model(rng, cfg)
    optimizer, opt_state = setup_optimizer(params, cfg)

    return params, model, optimizer, opt_state


cfg = ModelConfig(
    seq_len=256,
    n_layers=1,
    d_model=512,
    num_heads=8,
    ff_dim=3072,
    dropout=0.1,
    batch_size=100,
    learning_rate=1e-3,
)

params, model, optimizer, opt_state = setup_all(cfg)

# %%
# print(type(params))


def _indent(x, num_spaces):
  indent_str = ' ' * num_spaces
  lines = x.split('\n')
  assert not lines[-1]
  # skip the final line because it's empty and should not be indented.
  return '\n'.join(indent_str + line for line in lines[:-1]) + '\n'


def pretty_repr(fd, num_spaces=4):
    """Returns an indented representation of the nested dictionary."""

    def pretty_dict(x):
        if not isinstance(x, dict):
            return f"{x.shape}, {abs(x.mean()):.2e}, {x.std():.2e}" #repr(x)
        rep = ""
        for key, val in x.items():
            rep += f"{key}: {pretty_dict(val)},\n"
        if rep:
            return "{\n" + _indent(rep, num_spaces) + "}"
        else:
            return "{}"

    return f"FrozenDict({pretty_dict(fd._dict)})"


print(pretty_repr(params))
# print(x)

y = """
FrozenDict({
    params: {
        positional_encoding: (256, 512), 4.38e-05, 6.26e-02,
        byte_embedding: {
            embedding: (256, 512), 2.04e-04, 4.43e-02,
        },
        transformer_layers_0: {
            layer_norm_1: {
                scale: (512,), 1.00e+00, 0.00e+00,
                bias: (512,), 0.00e+00, 0.00e+00,
            },
            mha: {
                query: {
                    kernel: (512, 8, 64), 1.23e-05, 4.42e-02,
                    bias: (8, 64), 0.00e+00, 0.00e+00,
                },
                key: {
                    kernel: (512, 8, 64), 4.79e-05, 4.42e-02,
                    bias: (8, 64), 0.00e+00, 0.00e+00,
                },
                value: {
                    kernel: (512, 8, 64), 9.87e-06, 4.42e-02,
                    bias: (8, 64), 0.00e+00, 0.00e+00,
                },
                out: {
                    kernel: (8, 64, 512), 9.29e-05, 4.42e-02,
                    bias: (512,), 0.00e+00, 0.00e+00,
                },
            },
            layer_norm_2: {
                scale: (512,), 1.00e+00, 0.00e+00,
                bias: (512,), 0.00e+00, 0.00e+00,
            },
            linear_1: {
                kernel: (512, 3072), 1.77e-05, 4.42e-02,
                bias: (3072,), 0.00e+00, 0.00e+00,
            },
            linear_2: {
                kernel: (3072, 512), 2.83e-05, 1.80e-02,
                bias: (512,), 0.00e+00, 0.00e+00,
            },
        },
        prob_decoder: {
            kernel: (512, 256), 2.52e-04, 4.42e-02,
            bias: (256,), 0.00e+00, 0.00e+00,
        },
    },
})
"""

#%%
from pytorch_model import *

# This is the highest batch size PyTorch can handle, the JAX model can do 79
cfg = ModelConfig(
    seq_len=256,
    n_layers=1,
    d_model=512,
    num_heads=8,
    ff_dim=3072,
    dropout=0.1,
    batch_size=100,
    learning_rate=1e-3,
)

lm = LM(cfg).cuda()
print(lm)

for v in lm.parameters():
    print(f"{v.size()}".ljust(30), f"{abs(v.mean().item()):.2e}", f"{v.std().item():.2e}")
