# FastestDetV2

🔥🔥比[FastestDet](https://github.com/dog-qiuqiu/FastestDet)更快更强🔥🔥

## 关键改进

- ⚡相比[FastestDet](https://github.com/dog-qiuqiu/FastestDet)，**mAP50提升2.5%，mAP50:95提升1%，同时速度提升约20%**
- 采用**Assign Guidance Module**和**SimOTA**标签分配策略，以获得更好的精度
- 基于支持**量化感知训练**、可重参数化的**MobileOne**骨干网络和卷积模块

## 图片展示
<img src="https://github.com/Pairman/FastestDetV2/blob/main/.github/assets/readme_gallery_1.png">
<center><img src="https://github.com/Pairman/FastestDetV2/blob/main/.github/assets/readme_gallery_2.png" width="85%"></center>

## 基准测试
Model|mAP50|mAP50:95|Resolution|Inference time (4x core)|Inference time (1x core)|Params (M)
:---:|:---:|:---:|:---:|:---:|:---:|:---:
**[FastestDetV2](https://github.com/Pairman/FastestDetV2)**|**27.8%**|**14.0%**|**352X352**|**2.83ms**|**6.95ms**|**0.33M**
[FastestDet](https://github.com/dog-qiuqiu/FastestDet)|25.3%|13.0%|352X352|3.68ms|8.48ms|0.24M
[NanoDet-m](https://github.com/RangiLyu/nanodet)|-|20.6%|320X320|7.76ms|22.23ms|0.95M
[YOLOX-Nano](https://github.com/Megvii-BaseDetection/YOLOX)|-|25.8%|416X416|36.88ms|92.52ms|0.91M
[YOLOv8n](https://github.com/ultralytics/ultralytics)|56.8%|37.4%|640X640|57.03ms|122.63ms|7.2M

> 测试平台为野火鲁班猫4 RK3588S，ARM 4\*Cortex-A76 CPU@2.0GHz，使用[NCNN](https://github.com/Tencent/ncnn)。

## 多平台基准测试
Device|Computing backend|System|Framework|Inference time (4x core)|Inference time (1x core)
:---:|:---:|:---:|:---:|:---:|:---:
EmbedFire LubanCat-4|RK3588 (CPU) <sup>1</sup>|Linux (arm64)|NCNN|2.83ms|6.95ms
EmbedFire LubanCat-4|RK3588 (NPU)|Linux (arm64)|RKNN|7.067ms <sup>2</sup>|7.532ms
Google Pixel 10 Pro XL|Tensor G5 (CPU)|Android (arm64)|NCNN|2.69ms|3.88ms
OnePlus 6|Snapdragon 845 (CPU)|Android (arm64)|NCNN|4.73ms|8.14ms
Dell Precision 3630 Tower|Core i9-9900 (CPU) <sup>3</sup>|Linux (x86_64)|NCNN|2.90m|7.31ms
> <sup>1</sup>: 频率为2.0GHz。<br>
> <sup>2</sup>: 使用RKNNLite.NPU_CORE_0_1_2。<br>
> <sup>3</sup>: 频率为800MHz。

## 模型下载
Download|Note
:---:|:---:
[fastestdetv2.pth](https://github.com/Pairman/FastestDetV2/releases/download/v1/fastestdetv2.pth)|融合后的权重
[fastestdetv2_unfused.pth](https://github.com/Pairman/FastestDetV2/releases/download/v1/fastestdetv2_unfused.pth)|未融合的权重
[qamobileone.pth](https://github.com/Pairman/FastestDetV2/releases/download/v1/qamobileone.pth)|骨干网络权重
[fastestdetv2.apk](https://github.com/Pairman/FastestDetV2/releases/download/v1/fastestdetv2.apk)|安卓演示应用
[fastestdetv2.bin](https://github.com/Pairman/FastestDetV2/releases/download/v1/fastestdetv2.bin)<br>[fastestdetv2.param](https://github.com/Pairman/FastestDetV2/releases/download/v1/fastestdetv2.param)|NCNN文件
[fastestdetv2.rknn](https://github.com/Pairman/FastestDetV2/releases/download/v1/fastestdetv2.rknn)|RKNN文件
[fastestdetv2.onnx](https://github.com/Pairman/FastestDetV2/releases/download/v1/fastestdetv2.onnx)|ONNX文件
[fastestdetv2.pt](https://github.com/Pairman/FastestDetV2/releases/download/v1/fastestdetv2.pt)<br>[fastestdetv2_ptq,arm.pt](https://github.com/Pairman/FastestDetV2/releases/download/v1/fastestdetv2_ptq,arm.pt)<br>[fastestdetv2_ptq,x86.pt](https://github.com/Pairman/FastestDetV2/releases/download/v1/fastestdetv2_ptq,x86.pt)|TorchScript文件

# 使用

## 安装依赖

```sh
pip install -r requirements.txt
```

## 数据集、配置文件

数据集既可以是**Darknet 格式**（类似FastestDet，使用一个文本文件列出图片路径，标签存放在同目录下单独的`.txt`文件中），也可以是**YOLO 格式**（类似 YOLOv8，每张图片在单独目录中有一个对应的`.txt`标签文件）。标签格式为`cls cx cy w h`，即归一化后的边界框。

`.yaml`配置文件用于指定数据集路径、模型设置和训练超参数。数据集既可以使用Darknet格式，也可以使用YOLO格式。类别名称也可以存放在一个纯文本文件中，每行表示一个类别。示例参考[configs/coco.yaml](https://github.com/Pairman/FastestDetV2/blob/main/configs/coco.yaml)。

## 评估与测试

使用融合后的（重参数化后的）模型权重进行评估：

```sh
python3 eval.py --configs CONFIGS_PATH --weight WEIGHTS_PATH
```

或在单张图片上进行测试：

```sh
python3 test.py --configs CONFIGS_PATH --weights WEIGHTS_PATH --image IMAGE_PATH
```

## 训练

下载骨干网络权重并放到 `weights/qamobileone.pth`，然后运行：

```sh
python3 train.py --configs CONFIGS_PATH
```

或使用未融合权重进行微调：
```sh
python3 train.py --configs CONFIGS_PATH --weights WEIGHTS_PATH
```

## 部署

### ONNX和TorchScript

导出为ONNX和TorchScript:

```sh
python3 test.py --configs CONFIGS_PATH --weights WEIGHTS_PATH --export
```

### PT2E PTQ

针对x86（使用`X86InductorQuantizer`）或arm（使用`XNNPackQuantizer`）平台的训练后量化，输入为融合后的权重：

```sh
python3 quant.py --configs CONFIGS_PATH --weights WEIGHTS_PATH --image IMAGE_PATH --target TARGET_PLATFORM
```

### NCNN

参考[deploy/ncnn/README.md](https://github.com/Pairman/FastestDetV2/blob/main/deploy/ncnn/README.md)或[deploy/ncnn_android/README.md](https://github.com/Pairman/FastestDetV2/blob/main/deploy/ncnn_android/README.md)（Android）。

### RKNN

参考[deploy/rknn/README.md](https://github.com/Pairman/FastestDetV2/blob/main/deploy/rknn/README.md)。

# 引用

```
@misc{=FastestDetV2,
    title={FastestDetV2: Even faster and stronger than FastestDet},
    author={Pairman},
    howpublished = {\url{https://github.com/Pairman/FastestDetV2}},
    year={2025}
}
```

# 参考资料

- FastestDet: https://github.com/dog-qiuqiu/FastestDet
- Assign Guidance Module和NCNN部署: https://github.com/RangiLyu/nanodet
- MobileOne: https://github.com/apple/ml-mobileone and https://github.com/glory-wan/TF-Net
- 支持量化感知的可重参数化卷积: https://github.com/meituan/YOLOv6
- SimOTA标签分配策略: https://github.com/open-mmlab/mmyolo
- NCNN: https://github.com/Tencent/ncnn
