# %%
from jax_model import *


cfg = ModelConfig(
    seq_len=256,
    n_layers=1,
    d_model=512,
    num_heads=8,
    ff_dim=3072,
    dropout=0.1,
    batch_size=100,
    learning_rate=1e-3,
    max_num_batch=10000,
)

params, model, optimizer, opt_state = setup_all(cfg)

# %%
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


# FrozenDict({
#     params: {
#         positional_encoding: (256, 512), 4.38e-05, 6.26e-02,
#         byte_embedding: {
#             embedding: (256, 512), 2.04e-04, 4.43e-02,
#         },
#         transformer_layers_0: {
#             layer_norm_1: {
#                 scale: (512,), 1.00e+00, 0.00e+00,
#                 bias: (512,), 0.00e+00, 0.00e+00,
#             },
#             mha: {
#                 query: {
#                     kernel: (512, 8, 64), 1.23e-05, 4.42e-02,
#                     bias: (8, 64), 0.00e+00, 0.00e+00,
#                 },
#                 key: {
#                     kernel: (512, 8, 64), 4.79e-05, 4.42e-02,
#                     bias: (8, 64), 0.00e+00, 0.00e+00,
#                 },
#                 value: {
#                     kernel: (512, 8, 64), 9.87e-06, 4.42e-02,
#                     bias: (8, 64), 0.00e+00, 0.00e+00,
#                 },
#                 out: {
#                     kernel: (8, 64, 512), 9.29e-05, 4.42e-02,
#                     bias: (512,), 0.00e+00, 0.00e+00,
#                 },
#             },
#             layer_norm_2: {
#                 scale: (512,), 1.00e+00, 0.00e+00,
#                 bias: (512,), 0.00e+00, 0.00e+00,
#             },
#             linear_1: {
#                 kernel: (512, 3072), 1.77e-05, 4.42e-02,
#                 bias: (3072,), 0.00e+00, 0.00e+00,
#             },
#             linear_2: {
#                 kernel: (3072, 512), 2.83e-05, 1.80e-02,
#                 bias: (512,), 0.00e+00, 0.00e+00,
#             },
#         },
#         prob_decoder: {
#             kernel: (512, 256), 2.52e-04, 4.42e-02,
#             bias: (256,), 0.00e+00, 0.00e+00,
#         },
#     },
# })


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
    max_num_batch=10000,
)

lm = HandCraftLM(cfg, False).cuda()
print(lm)

for v in lm.parameters():
    print(f"{v.size()}".ljust(30), f"{abs(v.mean().item()):.2e}", f"{v.std().item():.2e}")

# torch.Size([256, 1, 512])      1.10e-03 9.98e-01
# torch.Size([256, 512])         3.78e-03 1.00e+00
# torch.Size([1536, 512])        4.96e-05 2.55e-02
# torch.Size([1536])             9.61e-04 2.53e-02
# torch.Size([512, 512])         1.13e-05 2.55e-02
# torch.Size([512])              6.32e-04 2.63e-02
# torch.Size([512])              1.00e+00 0.00e+00
# torch.Size([512])              0.00e+00 0.00e+00
# torch.Size([3072, 512])        1.46e-05 2.55e-02
# torch.Size([3072])             1.30e-03 2.54e-02
# torch.Size([512, 3072])        7.59e-06 1.04e-02
# torch.Size([512])              2.29e-05 1.04e-02
# torch.Size([512])              1.00e+00 0.00e+00
# torch.Size([512])              0.00e+00 0.00e+00
# torch.Size([256, 512])         7.77e-05 2.55e-02
# torch.Size([256])              1.56e-04 2.59e-02
