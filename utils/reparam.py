from copy import deepcopy
import torch.nn as nn

# https://github.com/apple/ml-mobileone/blob/b7f4e6d/mobileone.py#L408
def reparameterize_model(model: nn.Module) -> nn.Module:
    """Re-parameterize a train-time multi-branched model recursively
    into a single-branched model for inference."""
    # avoid editing original model
    model = deepcopy(model)
    # re-parameterize recursively
    def _reparam(module: nn.Module):
        if isinstance(model, nn.Module):
            for child in module.children():
                _reparam(child)
            if hasattr(module, "reparameterize"):
                module.reparameterize()
    _reparam(model)
    return model
