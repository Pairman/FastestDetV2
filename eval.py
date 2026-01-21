import argparse
from os import cpu_count
from pathlib import Path
import sys
import torch
from torch.utils.data import DataLoader
_ROOT = str(Path(__file__).resolve())
if not _ROOT in sys.path:
    sys.path.append(_ROOT)
from module.fastestdetv2 import FastestDetV2
from utils.config import Config
from utils.datasets import collate_fn, Dataset
from utils.evaluator import COCODetectionEvaluator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda", help="device")
    parser.add_argument("--weights", type=str, default=None, help=".pt weights")
    parser.add_argument("--yaml", type=str, default="", help=".yaml config")
    opt = parser.parse_args()
    cfg = Config(opt.yaml)
    # data loaders
    num_workers = max(4, cpu_count())
    val_dataset = Dataset(cfg.val_txt, cfg.input_width, cfg.input_height, False)
    val_loader = DataLoader(val_dataset, cfg.batch_size,
        shuffle=False, collate_fn=collate_fn, drop_last=False,
        num_workers=num_workers, persistent_workers=True)
    # model
    model = FastestDetV2(cfg.num_classes, load_weights=True, inference_mode=True).to(opt.device)
    model.load_state_dict(torch.load(opt.weights))
    print(f"Loaded detector weights {opt.weights}")
    model.eval()
    print("Starting evaluation")
    map50 = COCODetectionEvaluator(cfg.names, opt.device).eval(
        val_loader, model, colour="green")
