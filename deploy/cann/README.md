## CANN Deployment

### Model Conversion


Make sure you have set up the CANN environment. Then install ACL runtime and AISBench from [AIS-Bench Inference Tool User Guide](https://gitee.com/ascend/tools/blob/master/ais-bench_workload/tool/ais_bench/README_EN.md)

Export ```.onnx``` and ```.onnx.data``` from the project root:

```sh
python test.py --export --weights WEIGHTS_PATH --configs CONFIGS_PATH
```

Convert ```.onnx``` to ```.om```:

```sh
atc --model=ONNX_PATH --framework=5 \
  --soc_version=`python -c "import torch, torch_npu; print(torch.npu.get_device_name(0))"` \
  --input_format=NCHW --input_shape="input:1,3,352,352" \
  --output=fastestdetv2
```

### Running

#### Run benchmark:

```sh
python bench_cann.py --weights fastestdetv2.om
```

It's ok to see this error below, caused by ACL runtime itself. It does not affect normal usage and only happens on model releasing.

```
corrupted size vs. prev_size in fastbins
Aborted (core dumped)
```

#### Run test:

```sh
python test_cann.py --weights fastestdetv2.om --image input.jpg --result output.jpg
```
