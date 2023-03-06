## Benchmark speed of transformers on jax vs pytorch(2.0)

### Prerequisite
Check the requirement.txt file, not tested :). You need to install pytorch nightly to use `torch.compile`.

It is also interesting to know that pytorch's initialization for linear layer, i.e. $\text{uniform}(-\sqrt{\frac{1}{n}}, \sqrt{\frac{1}{n}})$ where $n$ is the number of input feature , makes the model converges faster. This is equivalent (maybe?) to using jax's variance initializer with variance = $\sqrt{\frac{1}{3n}}$, i.e. scale = 1/3, mode= "fan\_in".

#### Usage

download the data
```
wget http://mattmahoney.net/dc/enwik9.zip
```

run the code
```shell
# usage
# --model pytorch or handcraft
#     pytorch: model that builds on top of pytorch's transformer modules
#     handcraft: same model defined from basic pytorch components such as linear, einsum
# --compile 0 or 1, whether to turn on the torch.compile
# --dtype float32 or bfloat16
python pytorch_run.py --model handcraft --save_dir exps/pytorch/run --compile 1 --num_layer 8 --dtype float32
```

### Speed

Tested on RTX4090 graphics card. `--num_layer` is set to 8

All data types have similar training curve.

**Speed of torch model vs handcraft model**

|          | torch | handcraft | Speed up |
| -------- | ----- | --------- | -------- |
| bfloat16 | 1093  | 1273      | 1.16x    |

For the subsequent experiment we use handcraft model.

**Speed by dtypes on handcraft model**
| dtype    | torch | torch.compile | speed up | jax     | jax vs torch | jax vs<br>torch.compile |
| -------- | ----- | ------------- | -------- | ------- | ------------ | ----------------------- |
| float32  | 672   | 736           | 1.10x    | **797** | 1.19x        | 1.08x                   |
| float16  | 991   | 1287          | 1.30x    |         |              |                         |
| bfloat16 | 982   | 1273          | 1.30x    |         |              |                         |

Jax seems to beat torch.compile, at least on float32.
I have not learned how to write fp16 training in jax. It is not as easy as in pytorch.


**Speed of flash-attention**

flash-attention only works on fp16 and bfp16. 
Here we use the flash attention implemented in pytorch's `torch.nn.functional.scaled_dot_product`
instead of the standalone flash-attention repo. The later may be even faster but harder to use.

It is interesting that a naive implementaion is accelated to be almost as fast as the flash-attention.

| dtype    | torch.compile | flash    | speed up |
| -------- | ------------- | -------- | -------- |
| float16  | 1287          | **1425** | 1.11x    |
| bfloat16 | 1273          | **1435** | 1.13x    |


**NOTE**: somehow flash attention is slower when compiled

| dtype    | flash    | flash compile |
| -------- | -------- | ------------- |
| float16  | **1425** | 1170          |
| bfloat16 | **1435** | 1176          |






