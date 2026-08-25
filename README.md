# FastestDetV2 [**[中文]**](https://github.com/Pairman/FastestDetV2/blob/main/README_zh.md)

🔥🔥Even faster and stronger than [FastestDet](https://github.com/dog-qiuqiu/FastestDet)🔥🔥<br>
🔥🔥比[FastestDet](https://github.com/dog-qiuqiu/FastestDet)更快更强🔥🔥

## Improvements

- ⚡**2.5% mAP50 & 1% mAP50:95 improvement, with ~20% faster speed** compared to [FastestDet](https://github.com/dog-qiuqiu/FastestDet)
- **Assign Guidance Module** and **SimOTA** label assignment for better precision
- **Quantization-aware**, reparameterizable **MobileOne** backbone and convolution modules
- ⚡相比[FastestDet](https://github.com/dog-qiuqiu/FastestDet)，**mAP50提升2.5%，mAP50:95提升1%，同时速度提升约20%**
- 采用**Assign Guidance Module**和**SimOTA**标签分配策略，以获得更好的精度
- 基于支持**量化感知训练**、可重参数化的**MobileOne**骨干网络和卷积模块

## Gallery
<img src="https://github.com/Pairman/FastestDetV2/blob/main/.github/assets/readme_gallery_1.png">
<center><img src="https://github.com/Pairman/FastestDetV2/blob/main/.github/assets/readme_gallery_2.png" width="85%"></center>

## Benchmarks
Model|mAP50|mAP50:95|Resolution|Inference time (4x core)|Inference time (1x core)|Params (M)
:---:|:---:|:---:|:---:|:---:|:---:|:---:
**[FastestDetV2](https://github.com/Pairman/FastestDetV2)**|**27.8%**|**14.0%**|**352X352**|**2.83ms**|**6.95ms**|**0.33M**
**[FastestDetV2-2x](https://github.com/Pairman/FastestDetV2)**|**36.6%**|**19.9%**|**352X352**|**6.81ms**|**19.88ms**|**1.22M**
[FastestDet](https://github.com/dog-qiuqiu/FastestDet)|25.3%|13.0%|352X352|3.68ms|8.48ms|0.24M
[NanoDet-m](https://github.com/RangiLyu/nanodet)|-|20.6%|320X320|7.76ms|22.23ms|0.95M
[YOLOX-Nano](https://github.com/Megvii-BaseDetection/YOLOX)|-|25.8%|416X416|36.88ms|92.52ms|0.91M
[YOLOv8n](https://github.com/ultralytics/ultralytics)|56.8%|37.4%|640X640|57.03ms|122.63ms|7.2M

> Tested on EmbedFire LubanCat-4 RK3588S ARM 4\*Cortex-A76 CPU@2.0GHz, using [NCNN](https://github.com/Tencent/ncnn).

## Multi-platform Benchmarks
Device|Computing backend|System|Framework|Inference time (4x core)|Inference time (1x core) | 2x Inference time (4x core)| 2x Inference time (1x core)
:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:
Huawei Atlas 800I A3|Ascend 910_9362 (NPU)|Linux (arm64)|CANN|/|0.45ms|/|0.58ms
EmbedFire LubanCat-4|RK3588 (CPU) <sup>1</sup>|Linux (arm64)|NCNN|2.83ms|6.95ms|6.81ms|19.88ms
EmbedFire LubanCat-4|RK3588 (NPU)|Linux (arm64)|RKNN|7.067ms <sup>2</sup>|7.532ms|8.04ms <sup>3 </sup>|9.56ms
Google Pixel 10 Pro XL|Tensor G5 (CPU)|Android (arm64)|NCNN|2.69ms|3.88ms|4.66ms|6.26ms
OnePlus 6|Snapdragon 845 (CPU)|Android (arm64)|NCNN|4.73ms|8.14ms|11.56ms|17.84ms
Dell Precision 3630 Tower|Core i9-9900 (CPU) <sup>4</sup>|Linux (x86_64)|NCNN|2.90m|7.31ms|6.86ms|19.94ms
> <sup>1</sup>: At 2.0 GHz.<br>
> <sup>2</sup>, <sup>3</sup>: RKNNLite.NPU_CORE_0_1_2 is used.<br>
> <sup>4</sup>: At 800MHz.

## Model Zoo
Download|Note
:---:|:---:
[fastestdetv2.pth](https://github.com/Pairman/FastestDetV2/releases/download/v1/fastestdetv2.pth), [fastestdetv2_unfused.pth](https://github.com/Pairman/FastestDetV2/releases/download/v1/fastestdetv2_unfused.pth)<br>[fastestdetv2-2x.pth](https://github.com/Pairman/FastestDetV2/releases/download/v1/fastestdetv2-2x.pth), [fastestdetv2-2x_unfused.pth](https://github.com/Pairman/FastestDetV2/releases/download/v1/fastestdetv2-2x_unfused.pth)|Model weights
[qamobileone.pth](https://github.com/Pairman/FastestDetV2/releases/download/v1/qamobileone.pth)<br>[qamobileone-2x.pth](https://github.com/Pairman/FastestDetV2/releases/download/v1/qamobileone-2x.pth)|Backbone weights
[fastestdetv2.apk](https://github.com/Pairman/FastestDetV2/releases/download/v1/fastestdetv2.apk)|Android demo
[fastestdetv2.bin](https://github.com/Pairman/FastestDetV2/releases/download/v1/fastestdetv2.bin), [fastestdetv2.param](https://github.com/Pairman/FastestDetV2/releases/download/v1/fastestdetv2.param)<br>[fastestdetv2-2x.bin](https://github.com/Pairman/FastestDetV2/releases/download/v1/fastestdetv2-2x.bin), [fastestdetv2-2x.param](https://github.com/Pairman/FastestDetV2/releases/download/v1/fastestdetv2-2x.param)|NCNN files
[fastestdetv2.onnx](https://github.com/Pairman/FastestDetV2/releases/download/v1/fastestdetv2.onnx)<br>[fastestdetv2-2x.onnx](https://github.com/Pairman/FastestDetV2/releases/download/v1/fastestdetv2-2x.onnx)|ONNX files
(target platform-specific, not provided)|CANN files
(target platform-specific, not provided)|RKNN files
[fastestdetv2.pt](https://github.com/Pairman/FastestDetV2/releases/download/v1/fastestdetv2.pt), [fastestdetv2_ptq.arm.pt](https://github.com/Pairman/FastestDetV2/releases/download/v1/fastestdetv2_ptq.arm.pt), [fastestdetv2_ptq.x86.pt](https://github.com/Pairman/FastestDetV2/releases/download/v1/fastestdetv2_ptq.x86.pt)<br>[fastestdetv2-2x.pt](https://github.com/Pairman/FastestDetV2/releases/download/v1/fastestdetv2-2x.pt), [fastestdetv2-2x_ptq.arm.pt](https://github.com/Pairman/FastestDetV2/releases/download/v1/fastestdetv2-2x_ptq.arm.pt), [fastestdetv2-2x_ptq.x86.pt](https://github.com/Pairman/FastestDetV2/releases/download/v1/fastestdetv2_ptq.x86.pt)|TorchScript files

# Usage

## Dependencies

```sh
pip install -r requirements.txt
```

## Datasets & Configurations

Datasets can be either in **Darknet format** (like FastestDet, using a text file to list image paths, with labels stored in separate .txt files in the same directory) or in **YOLO format** (like YOLOv8, where each image has a corresponding .txt label file in a seperate directory). Labels are in ```cls cx cy w h``` normalized bboxes.

The .yaml configurations file specifies dataset paths, model settings, and training hyperparameters. Dataset could be either in Darknet format or YOLO format. Class names can also be in a single text file with each line representing a class name. See [configs/coco.yaml](https://github.com/Pairman/FastestDetV2/blob/main/configs/coco.yaml) for example.

## Evaluation & Testing

You can evaluate the model with a fused (reparameterized) model weights file.

```sh
python3 eval.py --configs CONFIGS_PATH --weight WEIGHTS_PATH
```

Or test it on an image:

```sh
python3 test.py --configs CONFIGS_PATH --weights WEIGHTS_PATH --image IMAGE_PATH
```

## Training

Download the backbone weights and place it under `weights/qamobileone.pth` and `weights/qamobileone-2x.pth`, and run:

```sh
python3 train.py --configs CONFIGS_PATH
```

Or finetune it with an unfused weights file:
```sh
python3 train.py --configs CONFIGS_PATH --weights WEIGHTS_PATH
```

## Deployment

### ONNX & TorchScript

Export to ONNX and TorchScript format with:

```sh
python3 test.py --configs CONFIGS_PATH --weights WEIGHTS_PATH --export
```

### PT2E PTQ

Post-training quantization for x86 (with ```X86InductorQuantizer```) or arm (with ```XNNPackQuantizer```) platforms, with fused weights:

```sh
python3 quant.py --configs CONFIGS_PATH --weights WEIGHTS_PATH --image IMAGE_PATH --target TARGET_PLATFORM
```

### NCNN

Follow [deploy/ncnn/README.md](https://github.com/Pairman/FastestDetV2/blob/main/deploy/ncnn/README.md) or [deploy/ncnn_android/README.md](https://github.com/Pairman/FastestDetV2/blob/main/deploy/ncnn_android/README.md) (for Android).

### CANN

Follow [deploy/cann/README.md](https://github.com/Pairman/FastestDetV2/blob/main/deploy/cann/README.md).

### RKNN

Follow [deploy/rknn/README.md](https://github.com/Pairman/FastestDetV2/blob/main/deploy/rknn/README.md).

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
- Assign Guidance Module and NCNN deployment: https://github.com/RangiLyu/nanodet
- MobileOne: https://github.com/apple/ml-mobileone and https://github.com/glory-wan/TF-Net
- Quantization-aware RepConv: https://github.com/meituan/YOLOv6
- SimOTA label assignment: https://github.com/open-mmlab/mmyolo
- NCNN: https://github.com/Tencent/ncnn
