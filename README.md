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
