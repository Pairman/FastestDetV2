from pathlib import Path
import sys
import torch
import torch.nn as nn
_ROOT = str(Path(__file__).resolve().parents[1])
if not _ROOT in sys.path:
    sys.path.append(_ROOT)
from module.detector import DetectHead, SPP
from module.qamobileone import QAMobileOne
from module.repconv import QARepConv

class FastestDetV2(nn.Module):
    def __init__(self, num_classes: int=80, backbone_blocks: list[int]=[4, 8, 4],
        backbone_channels: list[int]=[24, 48, 96, 192],
        inference_mode: bool=False, enable_agm: bool=True):
        super().__init__()
        self.inference_mode = inference_mode
        self.enable_agm = not inference_mode and enable_agm
        self.is_detach_backbone = False
        self.is_detach_agm = False
        self.backbone = QAMobileOne(backbone_blocks, backbone_channels,
            inference_mode=inference_mode)
        channels = sum(backbone_channels[-3:])
        head_channels = backbone_channels[2]
        self.upsample = nn.Upsample(scale_factor=2)
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.spp = SPP(channels, head_channels, inference_mode)
        self.det = DetectHead(head_channels, num_classes, inference_mode)
        # assign guidance module
        # https://github.com/RangiLyu/nanodet/blob/be9b4a9/nanodet/model/arch/nanodet_plus.py#L35
        if self.enable_agm:
            aux_channels = head_channels * 3 // 2
            self.aux_spp = SPP(channels, aux_channels, inference_mode)
            self.aux_det = nn.Sequential(
                # use qarepconv since it's faster at training-time
                QARepConv(aux_channels, aux_channels, 3, stride=1, padding=1),
                nn.SiLU(),
                QARepConv(aux_channels, aux_channels, 3, stride=1, padding=1),
                nn.SiLU(),
                DetectHead(aux_channels, num_classes))

    def forward(self, x):
        p2, p3, p4 = self.backbone(x)
        if self.training and self.is_detach_backbone:
            p2, p3, p4 = p2.detach(), p3.detach(), p4.detach()
        p = torch.cat((self.avg_pool(p2), p3, self.upsample(p4)), dim=1)
        y = self.spp(p)
        y = self.det(y)
        if self.enable_agm:
            aux_x = p.detach() if self.is_detach_agm else p
            aux_y = self.aux_spp(aux_x)
            aux_y = self.aux_det(aux_y)
            return y, aux_y
        return y

    def reparameterize(self):
        """Re-parameterization for inference."""
        if self.training or self.inference_mode:
            return
        if self.enable_agm:
            del self.aux_spp, self.aux_det
        self.inference_mode = True

if __name__ == "__main__":
    model = FastestDetV2(80)
    x = torch.rand(1, 3, 352, 352)
    model.eval()
    with torch.no_grad():
        y, aux_y = model(x)
        print(y.shape, aux_y.shape)
