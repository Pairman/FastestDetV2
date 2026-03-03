from copy import deepcopy
import torch

class EMA:
    """Exponential Moving Average for model parameters."""
    def __init__(self, model: torch.nn.Module, decay=0.9998, device="cpu"):
        self.decay = decay
        self.model = deepcopy(model).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def update(self, model: torch.nn.Module):
        with torch.no_grad():
            sd = model.state_dict()
            for k, v in self.model.state_dict().items():
                if v.dtype.is_floating_point:
                    v.mul_(self.decay).add_(sd[k].detach(), alpha=1 - self.decay)
