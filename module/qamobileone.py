from pathlib import Path
import sys
import torch
import torch.nn as nn
_ROOT = str(Path(__file__).resolve().parents[1])
if not _ROOT in sys.path:
    sys.path.append(_ROOT)
from module.repconv import QARepConv

# https://github.com/apple/ml-mobileone/blob/b7f4e6d/mobileone.py#L279
# faster than QARepConv at inference-time, but slower at training-time
class QAMobileOneBlock(nn.Module):
    """Re-parameterizable depthwise + pointwise conv block."""

    def __init__(self, in_channels: int, out_channels: int,
        stride: int=1, inference_mode: bool=False):
        super().__init__()
        self.dw = QARepConv(in_channels, in_channels, 3,
            stride=stride, padding=1, groups=in_channels,
            inference_mode=inference_mode)
        self.dw_act = nn.ReLU(inplace=True)
        self.pw = QARepConv(in_channels, out_channels, 1,
            stride=1, padding=0, groups=1,
            inference_mode=inference_mode)

    def forward(self, x):
        x = self.dw(x)
        x = self.dw_act(x)
        x = self.pw(x)
        return x

class QAMobileOne(nn.Module):
    """Quantization-aware mini MobileOne."""
    def __init__(self, num_blocks_per_stage: list[int]=[4, 8, 4],
        base_channels: list[int]=[24, 48, 96, 192], inference_mode=False):
        super().__init__()
        self.inference_mode = inference_mode
        self.base_channels = base_channels
        self.stem = nn.Sequential(
            QARepConv(3, base_channels[0], 3,
                stride=2, padding=1, inference_mode=inference_mode),
            nn.ReLU(inplace=True),
            QAMobileOneBlock(base_channels[0], base_channels[0],
                stride=2, inference_mode=inference_mode),
            nn.ReLU(inplace=True))
        self.s1 = self.make_stage(base_channels[0], base_channels[1],
            num_blocks_per_stage[0], 2)
        self.s2 = self.make_stage(base_channels[1], base_channels[2],
            num_blocks_per_stage[1], 2)
        self.s3 = self.make_stage(base_channels[2], base_channels[3],
            num_blocks_per_stage[2], 2)

    def make_stage(self, in_channels, out_channels, num_blocks, stride):
        """Construct a network stage with specified parameters."""
        layers = nn.Sequential()
        layers.append(QAMobileOneBlock(in_channels, out_channels,
            stride=stride, inference_mode=self.inference_mode))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_blocks - 1):
            layers.append(QAMobileOneBlock(out_channels, out_channels,
                stride=1, inference_mode=self.inference_mode))
            layers.append(nn.ReLU(inplace=True))
        return layers

    def forward(self, x):
        x = self.stem(x)
        p1 = self.s1(x)
        p2 = self.s2(p1)
        p3 = self.s3(p2)
        return [p1, p2, p3]

class QAMobileOneClassifier(nn.Module):
    """Classification model with QAMobileOne backbone."""

    def __init__(self, num_classes=1000, num_blocks_per_stage: list[int]=[4, 8, 4],
        base_channels: list[int]=[24, 48, 96, 192], inference_mode=False):
        super().__init__()
        self.backbone = QAMobileOne(num_blocks_per_stage, base_channels,
            inference_mode=inference_mode)
        # Classification head
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(base_channels[-1], num_classes)

    def forward(self, x):
        _, _, p4 = self.backbone(x) # (B, C, H, W)
        y = self.gap(p4)            # (B, C, 1, 1)
        y = y.view(y.size(0), -1)   # (B, C)
        y = self.fc(y)              # (B, CLS)
        return y

if __name__ == "__main__":
    x = torch.randn(1, 3, 352, 352)
    model = QAMobileOne()
    print(*[p.shape for p in model(x)])
