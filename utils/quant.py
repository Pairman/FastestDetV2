import torch

def print_quant_stats(model: torch.nn.Module):
    """Print int8/fp32 parameter counts by top-level module prefix."""
    groups = ("backbone", "spp", "det", "other")
    stats = {k: {"int8": 0, "fp32": 0, "other": 0} for k in groups}
    total = {"int8": 0, "fp32": 0, "other": 0}
    for n, t in model.state_dict().items():
        if n.startswith("backbone."):
            g = "backbone"
        elif n.startswith("spp."):
            g = "spp"
        elif n.startswith("det."):
            g = "det"
        else:
            g = "other"
        n = t.numel()
        if t.dtype == torch.int8:
            k = "int8"
        elif t.dtype == torch.float32:
            k = "fp32"
        else:
            k = "other"
        stats[g][k] += n
        total[k] += n
    print("Quantization stats (numel by dtype):")
    for g in groups:
        s = stats[g]
        print(f"  {g:<8} int8={s['int8']:<10} fp32={s['fp32']:<10} other={s['other']}")
    print(f"  {'total':<8} int8={total['int8']:<10} fp32={total['fp32']:<10} other={total['other']}")
