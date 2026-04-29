# FastestDetV2

Even faster and stronger than [FastestDet](https://github.com/dog-qiuqiu/FastestDet).

## Improvements

- **2.5% mAP50 & 1% mAP50:95 improvement, with ~20% faster speed** compared to [FastestDet](https://github.com/dog-qiuqiu/FastestDet)
- **Assign Guidance Module** and **SimOTA** label assignment for better accuracy
- **Quantization-aware**, reparameterizable **MobileOne** backbone and convolution modules

## Gallery
<img src="https://github.com/Pairman/FastestDetV2/blob/main/.github/assets/readme_gallery_1.png">
<center><img src="https://github.com/Pairman/FastestDetV2/blob/main/.github/assets/readme_gallery_2.png" width="85%"></center>

## Benchmarks
Model|mAP50|mAP50:95|Resolution|Inference time (4x core)|Inference time (1x core)|Params (M)
:---:|:---:|:---:|:---:|:---:|:---:|:---:
**[FastestDetV2](https://github.com/Pairman/FastestDetV2)**|**27.8%**|**14.0%**|**352X352**|**2.83ms**|**6.95ms**|**0.33M**
[FastestDet](https://github.com/dog-qiuqiu/FastestDet)|25.3%|13.0%|352X352|3.68ms|8.48ms|0.24M
[nanodet_m](https://github.com/RangiLyu/nanodet)|-|20.6%|320X320|7.76ms|22.23ms|0.95M
[yolox-nano](https://github.com/Megvii-BaseDetection/YOLOX)|-|25.8%|416X416|36.88ms|92.52ms|0.91M
[yolov8n](https://github.com/ultralytics/ultralytics)|56.8%|37.4%|640X640|57.03ms|122.63ms|7.2M

> Tested on EmbedFire LubanCat-4 RK3588S ARM 4\*Cortex-A76 CPU@2.0GHz, using [NCNN](https://github.com/Tencent/ncnn).

## Multi-platform benchmarks
Device|Computing backend|System|Framework|Inference time (4x core)|Inference time (1x core)
:---:|:---:|:---:|:---:|:---:|:---:
EmbedFire LubanCat-4|RK3588 (CPU@2.0GHz)|Linux (arm64)|NCNN|2.83ms|6.95ms
EmbedFire LubanCat-4|RK3588 (NPU)|Linux (arm64)|RKNN|7.067ms <sup>1</sup>|7.532ms
Google Pixel 10 Pro XL|Tensor G5 (CPU)|Android (arm64)|NCNN|2.69ms|3.88ms
OnePlus|Snapdragon 845 (CPU)|Android (arm64)|NCNN|4.73ms|8.14ms
Dell Precision 3630 Tower|Core i9-9900 (CPU) <sup>2</sup>|Linux (x86)|NCNN|2.90m|7.31ms
> <sup>1</sup>: RKNNLite.NPU_CORE_0_1_2 is used. <br>
> <sup>2</sup>: At 800MHz.

## Model Zoo
Download|Note
:---:|:---:
[fastestdetv2.pth](https://github.com/Pairman/FastestDetV2/releases/download/v1/fastestdetv2.pth)|Fused weights
[fastestdetv2_unfused.pth](https://github.com/Pairman/FastestDetV2/releases/download/v1/fastestdetv2_unfused.pth)|Unfused weights
[qamobileone.pth](https://github.com/Pairman/FastestDetV2/releases/download/v1/qamobileone.pth)|Backbone weights
[fastestdetv2.apk](https://github.com/Pairman/FastestDetV2/releases/download/v1/fastestdetv2.apk)|Android demo
[fastestdetv2.bin](https://github.com/Pairman/FastestDetV2/releases/download/v1/fastestdetv2.bin)<br>[fastestdetv2.param](https://github.com/Pairman/FastestDetV2/releases/download/v1/fastestdetv2.param)|NCNN files
[fastestdetv2.rknn](https://github.com/Pairman/FastestDetV2/releases/download/v1/fastestdetv2.rknn)|RKNN file
[fastestdetv2.onnx](https://github.com/Pairman/FastestDetV2/releases/download/v1/fastestdetv2.onnx)|ONNX file
[fastestdetv2.pt](https://github.com/Pairman/FastestDetV2/releases/download/v1/fastestdetv2.pt)<br>[fastestdetv2_ptq,arm.pt](https://github.com/Pairman/FastestDetV2/releases/download/v1/fastestdetv2_ptq,arm.pt)<br>[fastestdetv2_ptq,x86.pt](https://github.com/Pairman/FastestDetV2/releases/download/v1/fastestdetv2_ptq,x86.pt)|TorchScript files

# Usage

## Dependencies

```sh
pip install -r requirements.txt
```

## Datasets and Configurations

Datasets can be either in **Darknet format** (like FastestDet, using a text file to list image paths, with labels stored in separate .txt files in the same directory) or in **YOLO format** (like YOLOv8, where each image has a corresponding .txt label file in a seperate directory). Labels are in ```cls cx cy w h``` normalized bboxes.

The .yaml configurations file specifies dataset paths, model settings, and training hyperparameters. Dataset could be either in Darknet format or YOLO format. Class names can also be in a single text file with each line representing a class name. See [configs/coco.yaml](https://github.com/Pairman/FastestDetV2/blob/main/configs/coco.yaml) for example.

## Evaluation or testing

You can evaluate the model with a fused (reparameterized) model weights file.

```sh
python3 eval.py --configs CONFIGS_PATH --weight WEIGHTS_PATH
```

Or test it on an image:

```sh
python3 test.py --configs CONFIGS_PATH --weights WEIGHTS_PATH --image IMAGE_PATH
```

## Training

Download the backbone weights and place it under ```weights/qamobileone.pth```, and run:

```sh
python3 train.py --configs CONFIGS_PATH
```

Or finetune it with an unfused weights file:
```sh
python3 train.py --configs CONFIGS_PATH --weights WEIGHTS_PATH
```

## Deployment

### PT2E PTQ

Post-training quantization for x86 (with ```X86InductorQuantizer```) or arm (with ```XNNPackQuantizer```) platforms, with fused weights:

```sh
python3 quant.py --configs CONFIGS_PATH --weights WEIGHTS_PATH --image IMAGE_PATH --target TARGET_PLATFORM
```

### NCNN

Follow [deploy/ncnn/README.md](https://github.com/Pairman/FastestDetV2/blob/main/deploy/ncnn/README.md) or [deploy/ncnn_android/README.md](https://github.com/Pairman/FastestDetV2/blob/main/deploy/ncnn_android/README.md) (for Android).

### RKNN

Follow [deploy/rknn/README.md](https://github.com/Pairman/FastestDetV2/blob/main/deploy/rknn/README.md)

# Citation

```
@misc{=FastestDetV2,
    title={FastestDetV2: Even faster and stronger than FastestDet},
    author={Pairman},
    howpublished = {\url{https://github.com/Pairman/FastestDetV2}},
    year={2025}
}
```

# References

- FastestDet: https://github.com/dog-qiuqiu/FastestDet
- Assign guidance module and NCNN deployment: https://github.com/RangiLyu/nanodet
- MobileOne: https://github.com/apple/ml-mobileone and https://github.com/glory-wan/TF-Net
- Quantization-aware RepConv: https://github.com/meituan/YOLOv6
- SimOTA label assignment: https://github.com/open-mmlab/mmyolo
- NCNN: https://github.com/Tencent/ncnn
