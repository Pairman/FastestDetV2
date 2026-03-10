from pathlib import Path
import sys
import torch
import torch.nn as nn
_ROOT = str(Path(__file__).resolve().parents[1])
if not _ROOT in sys.path:
    sys.path.append(_ROOT)
from module.layers import DetectHead, SPP
from module.qamobileone.qamobileone import QAMobileOne
from module.repconv import QARepConv

class FastestDetV2(nn.Module):

    def __init__(self, num_classes: int, backbone_blocks: list[int]=[4, 6, 8, 3],
        backbone_channels: list[int]=[24, 48, 64, 128], head_channels: int=80,
        load_weights: bool=False, inference_mode: bool=False):
        super().__init__()
        self.inference_mode = inference_mode
        self.is_detach_backbone = False
        self.is_detach_agm = False
        self.backbone = QAMobileOne(backbone_blocks, backbone_channels,
            load_weights=not load_weights, inference_mode=inference_mode)
        channels = sum(backbone_channels[-3:])
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.spp = SPP(channels, head_channels, inference_mode)
        self.det = DetectHead(head_channels, num_classes, inference_mode)
        # auxiliary guidance branch
        # https://github.com/RangiLyu/nanodet/blob/be9b4a9/nanodet/model/arch/nanodet_plus.py#L35
        if not self.inference_mode:
            aux_channels = head_channels
            self.aux_spp = SPP(channels, aux_channels, inference_mode)
            self.aux_det = nn.Sequential(
                QARepConv(aux_channels, aux_channels, 3,
                    stride=1, padding=1, inference_mode=False),
                nn.SiLU(),
                QARepConv(aux_channels, aux_channels, 3,
                    stride=1, padding=1, inference_mode=False),
                nn.SiLU(),
                DetectHead(aux_channels, num_classes, inference_mode=False))

    @classmethod
    def from_config(cls, cfg, load_weights: bool=False, inference_mode: bool=False):
        return cls(cfg.num_classes,
            backbone_blocks=cfg.backbone_blocks,
            backbone_channels=cfg.backbone_channels,
            head_channels=cfg.head_channels,
            load_weights=load_weights,
            inference_mode=inference_mode)

    def forward(self, x):
        p1, p2, p3 = self.backbone(x)
        if not self.inference_mode and self.is_detach_backbone:
            p1, p2, p3 = p1.detach(), p2.detach(), p3.detach()
        p1 = self.avg_pool(p1)
        p3 = torch.nn.functional.interpolate(p3, scale_factor=2, mode="nearest")
        p = torch.cat((p1, p2, p3), dim=1)
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
    model = FastestDetV2(80, load_weights=False, inference_mode=False)
    x = torch.rand(1, 3, 352, 352)
    model.eval()
    with torch.no_grad():
        y, aux_y = model(x)
        print(y.shape, aux_y.shape)
