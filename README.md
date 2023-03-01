### Benchmark speed of transformers on jax vs pytorch(2.0)

#### prerequisite
Check the requirement.txt file, not tested :). You need to install pytorch nightly to use `torch.compile`.

It is also interesting to know that pytorch's initialization for linear layer, i.e. $\text{uniform}(-\sqrt{\frac{1}{n}}, \sqrt{\frac{1}{n}})$ where $n$ is the number of input feature , makes the training faster. This is equivalent to using jax's variance initializer with variance = $\sqrt{\frac{1}{3n}}$, mode = "fan\_in".


#### usage

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

#### Speed

Tested on RTX4090 graphics card. `--num_layer` is set to 8

All data types have similar training curve.

**Speed of torch model vs handcraft model**

|          | torch | handcraft | Speed up |
| -------- | ----- | --------- | -------- |
| bfloat16 | 1093  | 1273      | 1.16x    |

For the subsequent experiment we use handcraft model.

**Speed by dtypes on handcraft model**
| dtype    | torch | torch.compile | speed up | jax | jax vs torch | jax vs torch.compile |
| -------- | ----- | ------------- | -------- | --- | ------------ | -------------------- |
| float32  | 672   | 736           | 1.10x    | 797 | 1.19x        | 1.08x                |
| float16  | 991   | 1287          | 1.30x    |     |              |                      |
| bfloat16 | 982   | 1273          | 1.30x    |     |              |                      |

Jax seems to beat torch.compile, at least on float32.
I have not learned how to write fp16 training in jax. It is not as easy as in pytorch.


**Speed of flash attention**

flash attention only works on fp16 and bfp16

| dtype   | compile | normal | flash | speed up |
| ------- | ------- | ------ | ----- | -------- |
| float16 | yes     | 991    | 1425  | 1.44x    |
| float16 | no      | 1287   | 1170  | 0.9x     |

**NOTE**: somehow flash attention is slower when compiled






