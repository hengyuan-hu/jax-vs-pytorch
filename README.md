### Benchmark speed of transformers on jax vs pytorch(2.0)

#### prerequisite
Check the requirement.txt file, not tested :). You need to install pytorch nightly to use `torch.compile`.

#### usage
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

Tested on RTX4090 graphics card. num_player is set to 8

All data types have similar training curve.

**Speed by dtypes on handcraft model**
| dtype    | no compile | compile | Speed up |
| -------- | ---------- | ------- | -------- |
| float32  | 672        | 736     | 1.10x    |
| float16  | 991        | 1287    | 1.30x    |
| bfloat16 | 982        | 1273    | 1.30x    |

**Speed of torch model vs handcraft model**

|          | torch | handcraft | Speed up |
| -------- | ----- | --------- | -------- |
| bfloat16 | 1093  | 1273      | 1.16x    |

**Speed of flash attention**

flash attention only works on fp16

| dtype   | compile? | normal | flash | Speed up |
| ------- | -------- | ------ | ----- | -------- |
| float16 | 0        | 991    | 1425  | 1.44x    |
| float16 | 1        | 1287   | 1170  | 0.9x     |

**NOTE**: somehow flash attention is slower when compiled






