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
from utils.reparam import reparameterize_model

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
    # from mobileone.mobileone import reparameterize_model
    # model = FastestDet(80, load_weights=False, inference_mode=False, is_expr=True)
    # x = torch.rand(1, 3, 352, 352)
    # model.eval()
    # with torch.no_grad():
    #     print(model(x).shape)
    #     print(reparameterize_model(model)(x).shape)

    from timeit import default_timer as time
    import numpy as np
    import torch
    device = torch.device("cpu")
    n_warmup, n_run = 10, 1000
    configs = [[]]
    models = [
        FastestDetV2(80, True, True).eval().to(device)
    ]
    times = [[] for _ in configs]
    with torch.no_grad():
        for i in range(n_warmup + n_run):
            x = torch.rand(1, 3, 352, 352, device=device)
            _t = []
            for model, ts in zip(models, times):
                t = time()
                _ = model(x)
                t = time() - t
                ts.append(t)
                _t.append(t)
            print(f"i={i}, time={_t}")

    times = [ts[n_warmup:] for ts in times]
    stats = [{
        "avg": float(arr.mean()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "std": float(arr.std()),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    } for arr in [np.array(ts) for ts in times]]
    for config, stat in zip(configs, stats):
        print(f"config={config}, stats={stat}", flush=True)
