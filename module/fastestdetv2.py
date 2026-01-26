from pathlib import Path
import sys
import torch
import torch.nn as nn
_ROOT = str(Path(__file__).resolve().parents[1])
if not _ROOT in sys.path:
    sys.path.append(_ROOT)
from module.layers import DetectHead, SPP
from module.shufflenetv2.shufflenetv2 import ShuffleNetV2
from module.qamobileone.qamobileone import QAMobileOne

class FastestDetV2(nn.Module):
    def __init__(self, num_classes: int, load_weights: bool=False, inference_mode: bool=False):
        super().__init__()
        # self.backbone = ShuffleNetV2([4, 8, 4], [-1, 24, 48, 96, 192], not load_weights)
        # channels = sum(self.backbone.stage_out_channels[-3:])
        self.backbone = QAMobileOne(load_weights=not load_weights, inference_mode=inference_mode)
        channels = sum(self.backbone.base_channels[-3:])
        self.upsample = nn.Upsample(scale_factor=2)
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.spp = SPP(channels, channels//3, inference_mode)
        self.det = DetectHead(channels//3, num_classes, inference_mode)

    def forward(self, x):
        p1, p2, p3 = self.backbone(x)
        p1 = self.avg_pool(p1)
        p3 = self.upsample(p3)
        p = torch.cat((p1, p2, p3), dim=1)
        y = self.spp(p)
        return self.det(y)

if __name__ == "__main__":
    model = FastestDetV2(80, load_weights=False, inference_mode=False)
    x = torch.rand(1, 3, 352, 352)
    model.eval()
    with torch.no_grad():
        print(model(x).shape)
