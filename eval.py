import argparse
from os import cpu_count
from pathlib import Path
import sys
import torch
_ROOT = str(Path(__file__).resolve().parent)
if not _ROOT in sys.path:
    sys.path.append(_ROOT)
from module.fastestdetv2 import FastestDetV2
from utils.config import Config
from utils.datasets import collate_fn, Dataset
from utils.evaluator import COCODetectionEvaluator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda", help="device")
    parser.add_argument("--weights", type=str, default=str(Path(_ROOT)/"weights/fastestdetv2_coco_best.pth"), help=".pt weights")
    parser.add_argument("--configs", type=str, default=str(Path(_ROOT)/"configs/coco.yaml"), help=".yaml configs")
    opt = parser.parse_args()
    cfg = Config(opt.configs)
    # data loaders
    num_workers = max(4, cpu_count())
    val_dataset = Dataset(cfg.val_txt, cfg.input_size, False)
    val_loader = torch.utils.data.DataLoader(val_dataset, cfg.batch_size,
        shuffle=False, collate_fn=collate_fn, drop_last=False,
        num_workers=num_workers, persistent_workers=True)
    # model
    model = FastestDetV2(num_classes=cfg.num_classes,
        backbone_blocks=cfg.backbone_blocks,
        backbone_channels=cfg.backbone_channels,
        inference_mode=True).to(opt.device)
    model.load_state_dict(torch.load(opt.weights))
    print(f"Loaded detector weights {opt.weights}")
    model.eval()
    print("Starting evaluation")
    stats = COCODetectionEvaluator(cfg.names, opt.device).eval(
        val_loader, model, colour="green")
    for k, v in stats.items():
        print(f"{k}: {v:.6f}")
