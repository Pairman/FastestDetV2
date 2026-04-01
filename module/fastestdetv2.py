from pathlib import Path
import sys
import torch
import torch.nn as nn
_ROOT = str(Path(__file__).resolve().parents[1])
if not _ROOT in sys.path:
    sys.path.append(_ROOT)
from module.layers import DetectHead, SPP, SPP
from module.qamobileone.qamobileone import QAMobileOne
from module.repconv import QARepConv

class FastestDetV2(nn.Module):
    def __init__(self, num_classes: int, backbone_blocks: list[int]=[4, 10, 4],
        backbone_channels: list[int]=[24, 48, 96, 192], head_channels: int=96,
        inference_mode: bool=False):
        super().__init__()
        self.inference_mode = inference_mode
        self.is_detach_backbone = False
        self.is_detach_agm = False
        self.backbone = QAMobileOne(backbone_blocks, backbone_channels,
            inference_mode=inference_mode)
        channels = sum(backbone_channels[-3:])
        self.upsample = nn.Upsample(scale_factor=2)
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.spp = SPP(channels, head_channels, inference_mode)
        self.det = DetectHead(head_channels, num_classes, inference_mode)
        # auxiliary guidance branch
        # https://github.com/RangiLyu/nanodet/blob/be9b4a9/nanodet/model/arch/nanodet_plus.py#L35
        if not self.inference_mode:
            self.aux_spp = SPP(channels, channels//2, inference_mode)
            self.aux_det = nn.Sequential(
                QARepConv(channels//2, channels//2, 3,
                    stride=1, padding=1, inference_mode=False),
                nn.SiLU(),
                QARepConv(channels//2, channels//2, 3,
                    stride=1, padding=1, inference_mode=False),
                nn.SiLU(),
                DetectHead(channels//2, num_classes, inference_mode=False))

    def forward(self, x):
        p2, p3, p4 = self.backbone(x)
        if not self.inference_mode and self.is_detach_backbone:
            p2, p3, p4 = p2.detach(), p3.detach(), p4.detach()
        p = torch.cat((self.avg_pool(p2), p3, self.upsample(p4)), dim=1)
        y = self.spp(p)
        y = self.det(y)
        if not self.inference_mode:
            aux_x = p.detach() if self.is_detach_agm else p
            aux_y = self.aux_spp(aux_x)
            aux_y = self.aux_det(aux_y)
            return y, aux_y
        return y

    def reparameterize(self):
        """Re-parameterization for inference."""
        if self.inference_mode:
            return
        self.inference_mode = True
        del self.aux_spp, self.aux_det

if __name__ == "__main__":
    model = FastestDetV2(80)
    x = torch.rand(1, 3, 352, 352)
    model.eval()
    with torch.no_grad():
        y, aux_y = model(x)
        print(y.shape, aux_y.shape)
