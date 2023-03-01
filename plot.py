#%%
import matplotlib.pyplot as plt
plt.switch_backend("agg")
%matplotlib inline

def parse_log(path):
    lines = open(path, 'r').readlines()
    losses = []
    speeds = []
    for l in lines:
        # print(l)
        if 'At step' in l or 'At iter' in l:
            loss = float(l.split()[-3][:-1])
            speed = float(l.split()[-1])
            losses.append(loss)
            speeds.append(speed)

    return losses, speeds

def generate_grid(cols, rows, figsize=7):
    fig = plt.figure(figsize=(cols * figsize, rows * figsize))
    ax = fig.subplots(rows, cols)
    return fig, ax

#%%
import numpy as np

fp32_losses, fp32_speeds = parse_log(
    "./exps/pytorch/run_handcraft_layer8_float32_compiled/train.log")
fp16_losses, fp16_speeds = parse_log(
    "./exps/pytorch/run_handcraft_layer8_float16_compiled/train.log")
bfp16_losses, bfp16_speeds = parse_log(
    "./exps/pytorch/run_handcraft_layer8_bfloat16_compiled/train.log")
fp16_nc_losses, fp16_nc_speeds = parse_log(
    "./exps/pytorch/run_handcraft_layer8_float16/train.log")

print("fp32 speed:", np.mean(fp32_speeds[-10:]))
print("fp16 speed:", np.mean(fp16_speeds[-10:]))
# print("fp16 no compile speed:", np.mean(fp16_nc_speeds[-10:]))

fig, ax = generate_grid(1, 1, figsize=5)
ax.set_ylim(ymax=2)
ax.set_ylim(ymin=1)
ax.plot(fp32_losses, label="fp32")
ax.plot(fp16_losses, label="fp16")
ax.plot(bfp16_losses, label="bfp16")
fig.legend(loc="upper right")
fig.tight_layout()

#%%
fp16_losses, fp16_speeds = parse_log(
    "./exps/pytorch/run_handcraft_layer8_float16_compiled/train.log")
fp16f_losses, fp16f_speeds = parse_log(
    "./exps/pytorch/dev_handcraft_layer8_float16_flash/train.log")
fp16fc_losses, fp16fc_speeds = parse_log(
    "./exps/pytorch/dev_handcraft_layer8_float16_compiled_flash/train.log")

print("fp16 speed:", np.mean(fp16_speeds[-10:]))
print("fp16f speed:", np.mean(fp16f_speeds[-10:]))
print("fp16f compiled speed:", np.mean(fp16fc_speeds[-10:]))

fig, ax = generate_grid(1, 1, figsize=5)
ax.set_ylim(ymax=2)
ax.set_ylim(ymin=1)
ax.plot(fp16_losses, label="fp16")
ax.plot(fp16f_losses, label="fp16-flash")
ax.plot(fp16fc_losses, label="fp16-compiled-flash")
fig.legend(loc="upper right")
fig.tight_layout()


#%%

jax, _ = parse_log("./exps/jax/run_layer8/train.log")
torch, _ = parse_log("./exps/pytorch/run_handcraft_layer8_bfloat16/train.log")

fig, ax = generate_grid(1, 1, figsize=5)
ax.set_ylim(ymax=2)
ax.set_ylim(ymin=1)
ax.plot(jax, label="jax")
ax.plot(torch, label="pytorch")
fig.legend(loc="upper right")
fig.tight_layout()

