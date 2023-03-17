#%%%

import torch
import jax_model
import pytorch_model
import pytorch_run

from config import ModelConfig
from data import Enwik9Loader


enwik9 = "./enwik9"
cfg = ModelConfig(
    seq_len=256,
    n_layers=1,
    d_model=512,
    num_heads=8,
    ff_dim=2048,
    dropout=0.1,
    batch_size=100,
    learning_rate=1e-3,
    max_num_batch=5000,
)

dataloader = list(Enwik9Loader(cfg.batch_size, cfg.seq_len, enwik9))

#%%
### forward jax
import jax
import optax
import jax.numpy as jnp


def setup_optimizer(params, cfg: ModelConfig):
    # optimizer = optax.sgd(0.1) #cfg.learning_rate)
    optimizer = optax.adam(1e-3) #cfg.learning_rate)
    opt_state = optimizer.init(params)
    return optimizer, opt_state


def setup_all(cfg: ModelConfig, rng=None):
    rng = jax.random.PRNGKey(1)#random.randrange(-(2**63), 2**63))

    params, model = jax_model.setup_model(rng, cfg)
    optimizer, opt_state = setup_optimizer(params, cfg)

    return params, model, optimizer, opt_state

params_jx, model_jx, optimizer_jx, opt_state_jx = setup_all(cfg)


#%%%
### create pytorch with the same weight
import numpy as np


model_pt = pytorch_model.HandCraftLM(cfg, False)#.to("cuda")


def set_linear(pt, jx):
    weight = torch.from_numpy(np.array(jx["kernel"]))
    if weight.dim() == 2:
        pt.weight.data = weight.transpose(0, 1)
    else:
        pt.weight.data = weight.view(-1, weight.size(2)).transpose(0, 1)
    pt.bias.data = torch.from_numpy(np.array(jx["bias"]))


def sync_params():
    for k, v in params_jx["params"].items():
        print(k)
        if k == "byte_embedding":
            # print()
            model_pt.byte_embedding.weight.data = torch.from_numpy(np.array(v["embedding"]))
        if k == "positional_encoding":
            model_pt.positional_encoding.data = torch.from_numpy(np.array(v)).unsqueeze(1)
        if k == "prob_decoder":
            # for kk, vv in v.items():
            #     print(kk, vv.shape)
            model_pt.prob_decoder.weight.data = torch.from_numpy(np.array(v["kernel"])).transpose(0, 1)
            model_pt.prob_decoder.bias.data = torch.from_numpy(np.array(v["bias"]))
        if k == "transformer_layers_0":
            for kk, vv in v.items():
                print("\t", kk)
                if kk == "linear_1" or kk == "linear_2":
                    name = kk.replace("_", "")
                    set_linear(getattr(model_pt.transformer[0], name), vv)
                if kk == "mha":
                    qkv = [None, None, None]
                    qkv_bias = [None, None, None]
                    for kkk, vvv in vv.items():
                        print("\t\t", kkk)
                        if kkk == "out":
                            # print(vvv[""].shape)
                            set_linear(model_pt.transformer[0].mha.out_proj, vvv)
                        else:
                            index = "qkv".index(kkk[0])
                            # print(kkk, index)
                            # print(kkk, vvv["kernel"].shape)
                            qkv[index] = torch.from_numpy(np.array(vvv["kernel"])).view(512, -1)
                            qkv_bias[index] = torch.from_numpy(np.array(vvv["bias"])).view(-1)
                    qkv = torch.cat(qkv, 1).transpose(0, 1)
                    qkv_bias = torch.cat(qkv_bias)
                    # print(qkv.size(), qkv_bias.size())
                    # model_pt.transformer[0].mha.qkv_proj.weight.data = qkv
                    # model_pt.transformer[0].mha.qkv_proj.bias.data = qkv_bias
    return

# sync_params()


def _indent(x, num_spaces):
  indent_str = ' ' * num_spaces
  lines = x.split('\n')
  assert not lines[-1]
  # skip the final line because it's empty and should not be indented.
  return '\n'.join(indent_str + line for line in lines[:-1]) + '\n'


def pretty_repr(fd, num_spaces=4):
    """Returns an indented representation of the nested dictionary."""
    sum = 0

    def pretty_dict(x):
        if not isinstance(x, dict):
            nonlocal sum
            sum += abs(x).sum()
            return f"{x.shape}, [{abs(x).sum():.2f}], {abs(x).mean():.2e}, {x.std():.2e}" #repr(x)
        rep = ""
        for key, val in x.items():
            rep += f"{key}: {pretty_dict(val)},\n"
        if rep:
            return "{\n" + _indent(rep, num_spaces) + "}"
        else:
            return "{}"

    ret = f"FrozenDict({pretty_dict(fd._dict)})"
    print("jax total sum:", sum)
    return ret, sum

jx_print, jx_sum = pretty_repr(params_jx)
print(jx_print)


pt_sum = 0
for k, v in model_pt.named_parameters():
    pt_sum += abs(v).sum().item()
    print(f"{k}".ljust(35),  f"{v.size()}".ljust(25),
          f"[{abs(v).sum().item():.2f}]",
          f"{abs(v).mean().item():.2e}",
          f"{v.std().item():.2e}")

print("pt_sum", pt_sum-1024, "jx_sum", jx_sum-1024, "diff: ", jx_sum - pt_sum)


#%% run jax

def run_train_step(params, opt_state, text_batch, rng):
    rng, rng2 = jax.random.split(rng)
    loss, grad = jax.value_and_grad(
        lambda p: jax.vmap(
            lambda text: jax_model.compute_loss(p, model_jx, text=text, rng=rng),
            in_axes=0,
            out_axes=0,
        )(text_batch).mean()
    )(params)
    updates, opt_state = optimizer_jx.update(grad, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss, rng2

fast_train_step = jax.jit(run_train_step, donate_argnums=[0, 1, 3])
rng = jax.random.PRNGKey(1)


for i in range(4):
    batch = jnp.array(dataloader[i])
    params_jx, opt_state_jx, loss, rng = fast_train_step(params_jx, opt_state_jx, batch, rng)
    print(i, loss)

#%%%
# run pytorch

model_pt = model_pt.to("cuda")
optim_pt = torch.optim.Adam(model_pt.parameters(), lr=1e-3)
# optim_pt = torch.optim.SGD(model_pt.parameters(), lr=0.1)

for i in range(4):
    batch = dataloader[i]
    data = torch.tensor(batch, device="cuda").transpose(0, 1).contiguous()
    loss = pytorch_run.compute_loss(model_pt, data)
    loss.backward()
    optim_pt.step()
    optim_pt.zero_grad()
    print(f"{i}: {loss.item():.4f}")
