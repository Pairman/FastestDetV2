## RKNN Deployment

### Model Conversion

Install [RKNN-Toolkit2](https://github.com/airockchip/rknn-toolkit2) following the official guide.

Export ```.onnx``` from the project root:

```sh
python test.py --export --weights WEIGHTS_PATH --configs CONFIGS_PATH
```

Convert ```.onnx``` to ```.rknn```:

```sh
python export_rknn.py --onnx ONNX_PATH --target rk3588
```

### Running

#### Run benchmark:

Run on-device benchmark:

```sh
python bench_rknn.py --rknn RKNN_PATH
```
