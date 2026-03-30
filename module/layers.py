from pathlib import Path
import sys
import torch
import torch.nn as nn
_ROOT = str(Path(__file__).resolve().parents[1])
if not _ROOT in sys.path:
    sys.path.append(_ROOT)
from module.repconv import QARepConv

class Head(nn.Module):
    def __init__(self, in_channels, out_channels, inference_mode=False):
        super().__init__()
        self.conv5x5 = QARepConv(in_channels, in_channels, 5,
            stride=1, padding=2, groups=in_channels, inference_mode=inference_mode)
        self.relu = nn.ReLU(inplace=True)
        self.conv1x1 = QARepConv(in_channels, out_channels, 1,
            stride=1, padding=0, groups=1, inference_mode=inference_mode)

    def forward(self, x):
        x = self.conv5x5(x)
        x = self.relu(x)
        x = self.conv1x1(x)
        return x

class DetectHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, inference_mode=False):
        super().__init__()
        self.conv1x1 = QARepConv(in_channels, in_channels, 1,
            stride=1, padding=0, groups=1, inference_mode=inference_mode)
        self.relu = nn.ReLU(inplace=True)
        self.obj_head = Head(in_channels, 1, inference_mode)
        self.reg_head = Head(in_channels, 4, inference_mode)
        self.cls_head = Head(in_channels, num_classes, inference_mode)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1x1(x)
        x = self.relu(x)
        obj = self.sigmoid(self.obj_head(x))
        reg = self.reg_head(x)
        cls = self.softmax(self.cls_head(x))
        return torch.cat((obj, reg, cls), dim=1)

class SPP(nn.Module):
    def __init__(self, in_channels, out_channels, inference_mode=False):
        super().__init__()
        self.stem = QARepConv(in_channels, out_channels, 1,
            stride=1, padding=0, groups=1, inference_mode=inference_mode)
        self.s1 = self.make_stage(out_channels, 5, 1, inference_mode)
        self.s2 = self.make_stage(out_channels, 5, 2, inference_mode)
        self.s3 = self.make_stage(out_channels, 5, 3, inference_mode)
        self.out = QARepConv(out_channels * 3, out_channels, 1,
            stride=1, padding=0, groups=1, inference_mode=inference_mode)
        self.relu = nn.ReLU(inplace=True)

    @staticmethod
    def make_stage(channels: int, kernel_size: int, num_convs: int,
        inference_mode=False):
        layers = []
        for _ in range(num_convs):
            layers.append(QARepConv(channels, channels, kernel_size,
                stride=1, padding=kernel_size // 2, groups=channels,
                inference_mode=inference_mode))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        y0 = self.stem(x)
        y1 = self.s1(y0)
        y2 = self.s2(y0)
        y3 = self.s3(y0)
        y = torch.cat((y1, y2, y3), dim=1)
        return self.relu(y0 + self.out(y))

if __name__ == "__main__":
    x = torch.randn(1, 48+96+192, 22, 22)
    model = SPP(x.shape[1], x.shape[1]//3, True)
    print(*[p.shape for p in model(x)])
