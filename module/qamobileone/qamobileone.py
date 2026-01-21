from pathlib import Path
import sys
import torch
import torch.nn as nn
_ROOT = str(Path(__file__).resolve().parents[2])
if not _ROOT in sys.path:
    sys.path.append(_ROOT)
from module.repconv import QARepConv

# https://github.com/apple/ml-mobileone/blob/b7f4e6d/mobileone.py#L279
# https://github.com/glory-wan/TF-Net/blob/9647ca2/TFNet/mdoel/TFNet.py#L124
class QAMobileOne(nn.Module):
    """Quantization-aware mini MobileOne."""
    def __init__(self, num_blocks_per_stage: list[int]=[1, 2, 3, 2],
        base_channels: list[int]=[24, 48, 96, 192],
        load_weights=False, inference_mode=False):
        super().__init__()
        self.inference_mode = inference_mode
        self.base_channels = base_channels
        self.stem = nn.Sequential(
            QARepConv(3, base_channels[0], 3,
                stride=2, padding=1, inference_mode=inference_mode),
            nn.ReLU(inplace=True),
            QARepConv(base_channels[0], base_channels[0], 3,
                stride=1, padding=1, inference_mode=inference_mode),
            nn.ReLU(inplace=True))
        self.stage1 = self.make_stage(base_channels[0], base_channels[0], num_blocks_per_stage[0], 1)
        self.stage2 = self.make_stage(base_channels[0], base_channels[1], num_blocks_per_stage[1], 2)
        self.stage3 = self.make_stage(base_channels[1], base_channels[2], num_blocks_per_stage[2], 2)
        self.stage4 = self.make_stage(base_channels[2], base_channels[3], num_blocks_per_stage[3], 2)
        if load_weights:
            weights = str(Path(__file__).resolve().parent/f"{type(self).__name__.lower()}.pth")
            self.load_state_dict(torch.load(weights))
            print(f"Loaded backbone weights {weights}")

    def make_stage(self, in_channels, out_channels, num_blocks, stride):
        """Construct a network stage with specified parameters."""
        layers = nn.Sequential()
        layers.append(QARepConv(in_channels, out_channels, 3,
            stride=stride, padding=1, inference_mode=self.inference_mode))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_blocks - 1):
            layers.append(QARepConv(out_channels, out_channels, 3,
                stride=1, padding=1, inference_mode=self.inference_mode))
            layers.append(nn.ReLU(inplace=True))
        return layers

    def forward(self, x):
        x = self.stem(x)
        p1 = self.stage1(x)
        p2 = self.stage2(p1)
        p3 = self.stage3(p2)
        p4 = self.stage4(p3)
        return [p2, p3, p4]

class QAMobileOneClassifier(nn.Module):
    """Classification model with QAMobileOne backbone."""

    def __init__(self, num_blocks_per_stage=[1, 2, 3, 2],
                 base_channels=[24, 48, 96, 192],
                 num_classes=1000,
                 inference_mode=False):
        super().__init__()
        self.backbone = QAMobileOne(num_blocks_per_stage, base_channels,
            inference_mode=inference_mode)
        # Classification head
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(base_channels[-1], num_classes)

    def forward(self, x):
        _, _, p3 = self.backbone(x) # (B, C, H, W)
        x = self.gap(p3)            # (B, C, 1, 1)
        x = x.view(x.size(0), -1)   # (B, C)
        x = self.fc(x)              # (B, CLS)
        return x

if __name__ == "__main__":
    x = torch.randn(1, 3, 352, 352)
    model = QAMobileOne(load_weights=True)
    print(*[p.shape for p in model(x)])
