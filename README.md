# FastestDetV2

Even faster and stronger than [FastestDet](https://github.com/dog-qiuqiu/FastestDet).

> This is still a work in progress.

## Improvements

* Auxiliary Guidance Module and SimOTA label assignment for better accuracy
* Quantization-aware Reparameterizable Convolution Modules
* Quantization-aware MobileOne Backbone

## Benchmarks
Model|mAP 0.5|mAP 0.5:0.95|Resolution|Inference time (4x core)|Inference time (1x core)|Params (M)
:---:|:---:|:---:|:---:|:---:|:---:|:---:
[FastestDetV2](https://github.com/Pairman/FastestDetV2)|27.3%|13.8%|352X352|3.26ms|8.51ms|0.20M
[FastestDet](https://github.com/dog-qiuqiu/FastestDet)|25.3%|13.0%|352X352|3.70ms|8.79ms|0.24M
[nanodet_m](https://github.com/RangiLyu/nanodet)|-|20.6%|320X320|7.76ms|22.23ms|0.95M
[yolox-nano](https://github.com/Megvii-BaseDetection/YOLOX)|-|25.8%|416X416|36.88ms|92.52ms|0.91M
[yolov8n](https://github.com/ultralytics/ultralytics)|56.8%|37.4%|640X640|57.03ms|122.63ms|7.2M

> Tested on EmbedFire LubanCat-4 RK3588S ARM 4\*Cortex-A76 CPU@2.0GHz, using [NCNN](https://github.com/Tencent/ncnn).

## Multi-platform benchmarks
Device|Computing backend|System|Framework|Inference time (4x core)|Inference time (1x core)
:---:|:---:|:---:|:---:|:---:|:---:
EmbedFire LubanCat-4|RK3588 (CPU@2.0GHz)|Linux (arm)|NCNN|3.26ms|8.51ms
EmbedFire LubanCat-4|RK3588 (NPU)|Linux (arm)|RKNN|-|12.81ms
Google Pixel 10 Pro XL|Tensor G5 (CPU)|Android (arm)|NCNN|2.28ms|3.98ms
OnePlus|Snapdragon 845 (CPU)|Android (arm)|NCNN|9.39ms|5.37ms
Dell Precision 3630 Tower|Core i9-9900 (CPU@800MHz)|Linux (x86)|NCNN|3.344m|7.943ms

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
  # Optional: backbone type
  BACKBONE_TYPE: qamobileone  # or shufflenetv2 or hybrid
  # Optional: MobileOne stage depths, channels and detection head width
  BACKBONE_BLOCKS: [4, 6, 8, 3]
  BACKBONE_CHANNELS: [24, 32, 64, 128]
  HEAD_CHANNELS: 80
  # Optional: only enable if the backbone architecture matches the pretrained qamobileone.pth
  BACKBONE_PRETRAINED: false
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

For a V1-scale FastestDetV2 preset tuned for NCNN CPU latency, start from `configs/coco_lite.yaml`.
For a ShuffleNetV2-style fast backbone preset, use `configs/coco-shuffle.yaml`.
For a hybrid backbone preset (Shuffle early stages + MobileOne tail), use `configs/coco-hybrid.yaml`.

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

### PT2E PTQ

Post-training quantization for x86 (with ```X86InductorQuantizer```) or arm (with ```XNNPackQuantizer```) platforms, with fused weights:

```sh
python3 quant.py --configs CONFIGS_PATH --weights WEIGHTS_PATH --image IMAGE_PATH --target TARGET_PLATFORM
```

### NCNN

Export to TorchScript with fused weights:

```sh
python test.py --weights WEIGHTS_PATH --export
```

Then follow [deploy/ncnn/README.md](https://github.com/Pairman/FastestDetV2/blob/main/deploy/ncnn/README.md)

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
- Auxiliary Guidance Module and NCNN deployment: https://github.com/RangiLyu/nanodet
- Quantization-aware RepConv: https://github.com/meituan/YOLOv6
- MobileOne: https://github.com/apple/ml-mobileone and https://github.com/glory-wan/TF-Net
- NCNN: https://github.com/Tencent/ncnn
