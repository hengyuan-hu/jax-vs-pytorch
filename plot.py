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

jax_losses, jax_speeds = parse_log("./exps/jax_layer1/train.log")
pytorch_losses, pytorch_speeds = parse_log("./exps/pytorch_layer1/train.log")
craft_losses, craft_speeds = parse_log("./exps/pytorch_run2_modelhandcraft_layer1_fphighest/train.log")

print("jax speed:", np.mean(jax_speeds[-10:]))
print("pytorch speed:", np.mean(pytorch_speeds[-10:]))
print("craft speed:", np.mean(craft_speeds[-10:]))

fig, ax = generate_grid(1, 1, figsize=5)
ax.set_ylim(ymax=2)
ax.set_ylim(ymin=1)
ax.plot(jax_losses, label="jax")
ax.plot(pytorch_losses, label="torch")
ax.plot(craft_losses, label="craft")
fig.legend()
# fig.show()

#%%
import numpy as np

fp32_losses, fp32_speeds = parse_log(
    "./exps/pytorch_compile_handcraft_layer8_float32_compiled/train.log")
fp16_losses, fp16_speeds = parse_log(
    "./exps/pytorch_compile_modelhandcraft_layer8_bfloat16/train.log")

print("fp32 speed:", np.mean(fp32_speeds[-10:]))
print("fp16 speed:", np.mean(fp16_speeds[-10:]))

fig, ax = generate_grid(1, 1, figsize=5)
ax.set_ylim(ymax=2)
ax.set_ylim(ymin=1)
ax.plot(fp32_losses, label="fp32")
ax.plot(fp16_losses, label="fp16")
fig.legend()
# fig.show()
