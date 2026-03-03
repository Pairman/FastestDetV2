# FastestDetV2

Even faster and stronger than [FastestDet](https://github.com/dog-qiuqiu/FastestDet).

> This is still a work in progress.

## Improvements

* Auxiliary Guidance Module for better accuracy
* Quantization-aware Reparameterizable Convolution Modules
* Quantization-aware MobileOne Backbone

## Benchmarks
Network|mAPval 0.5|mAPval 0.5:0.95|Resolution|Run Time(4xCore)|Run Time(1xCore)|Params(M)
:---:|:---:|:---:|:---:|:---:|:---:|:---:
FastestDetV2|27.3%|13.8%|352X352|??ms|??ms|0.92M
[FastestDet](https://github.com/dog-qiuqiu/FastestDet)|25.3%|13.0%|352X352|??ms|??ms|0.24M
[nanodet_m](https://github.com/RangiLyu/nanodet)|-|20.6%|320X320|??ms|??ms|0.95M
[yolox-nano](https://github.com/Megvii-BaseDetection/YOLOX)|-|25.8%|416X416|??ms|??ms|0.91M
[yolov5s](https://github.com/ultralytics/yolov5)|56.8%|37.4%|640X640|??ms|??ms|7.2M

> Test platform EmbedFire LubanCat-4 RK3588S ARM 4\*Cortex-A76 + 4\*Cortex-A55 CPU，Based on [NCNN](https://github.com/Tencent/ncnn). CPU lock frequency 2.0GHz.

## Multi-platform benchmarks
Equipment|Computing backend|System|Framework|Run time(Single core)|Run time(Multi core)
:---:|:---:|:---:|:---:|:---:|:---:
EmbedFire LubanCat-4|RK3588 (CPU)|Linux (arm)|ncnn|??ms|??ms
EmbedFire LubanCat-4 | RK3568 (NPU) |Linux (arm)|rknn|??ms|-
Google Pixel 10 Pro XL|Tensor G5 (CPU)|Android (arm)|ncnn|??ms|??ms
Dell Precision 3630 Tower|Core i9-9900 (CPU)|Linux (x86)|ncnn|??ms|??ms

# Usage

## Dependencies

```sh
pip install -r requirements.txt
```

## Datasets

Datasets can be either in Darknet format (like FastestDet, using a text file to list image paths, with labels stored in separate .txt files in the same directory) or in YOLO format (like YOLOv8, where each image has a corresponding .txt label file in a seperate directory). Labels are in ```cls cx cy w h``` normalized bboxes.

## Configurations

The .yaml configurations file specifies dataset paths, model settings, and training hyperparameters. Dataset could be either in Darknet format or YOLO format. Class names can also be in a single text file with each line representing a class name.

```yaml
DATASET:
  # Path to training images list file (darknet style) or directory (yolo style)
  TRAIN: "/data/datasets/coco-darknet/train2017.txt"
  # Path to evaluation images list file (darknet style) or directory (yolo style)
  VAL: "/data/datasets/coco-darknet/val2017.txt"
  # Path to class names list file or list of class names
  NAMES: [person, bicycle, ..., toothbrush]
MODEL:
  # Number of classes
  NUM_CLASSES: 80
  # Input width and height
  INPUT_SIZE: [352, 352]
TRAIN:
  # Initial learning rate
  LEARNING_RATE: 0.001
  # Number of warm-up epochs
  WARMUP_EPOCH: 5
  # Batch size
  BATCH_SIZE: 256
  # Total training epochs
  END_EPOCH: 300
  # Epochs for learning rate decay
  MILESTIONES: [100, 200, 250]
```

## Training

Train from start:

```sh
python3 train.py --configs CONFIGS_PATH
```

Finetune with unfused weights:
```sh
python3 train.py --configs CONFIGS_PATH --weights WEIGHTS_PATH
```

## Evaluation

Evaluate with fused weights:

```sh
python3 eval.py --configs CONFIGS_PATH --weight WEIGHTS_PATH
```

## Testing

Test on an image with fused weights:

```sh
python3 test.py --configs CONFIGS_PATH --weights WEIGHTS_PATH --image IMAGE_PATH
```

## Deployment

Post-training quantization for x86 (with ```X86InductorQuantizer```) or arm (with ```XNNPackQuantizer```) platforms, with fused weights:

```sh
python3 quant.py --configs CONFIGS_PATH --weights WEIGHTS_PATH --image IMAGE_PATH --target TARGET_PLATFORM
```

> Currently only ```X86InductorQuantizer``` in fully supported. ```XNNPackQuantizer``` has some issues on filtering submodules to quantize.

> Deployment using ONNX, NCNN and other methods will be available if I have time.

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
- Auxiliary Guidance Module: https://github.com/RangiLyu/nanodet
- Quantization-aware RepConv: https://github.com/meituan/YOLOv6
- MobileOne: https://github.com/apple/ml-mobileone and https://github.com/glory-wan/TF-Net
- SGD with Stable Weight Decay: https://github.com/zeke-xie/stable-weight-decay-regularization
- NCNN: https://github.com/Tencent/ncnn
