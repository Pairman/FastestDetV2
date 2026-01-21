import argparse
from os import cpu_count
from pathlib import Path
from shutil import get_terminal_size
import sys
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
_ROOT = str(Path(__file__).resolve())
if not _ROOT in sys.path:
    sys.path.append(_ROOT)
from module.fastestdetv2 import FastestDetV2
from module.loss import DetectorLoss
from utils.config import Config
from utils.datasets import collate_fn, Dataset
from utils.evaluator import COCODetectionEvaluator
from utils.lr import MultiStepCosineLR
from utils.reparam import reparameterize_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda", help="device")
    parser.add_argument("--weights", type=str, default=None, help=".pt weights")
    parser.add_argument("--yaml", type=str, default="", help=".yaml config")
    parser.add_argument("--enable-wandb", action="store_true", help="log to wandb")
    opt = parser.parse_args()
    cfg = Config(opt.yaml)
    cfg_name = Path(opt.yaml).stem
    savedir = Path(__file__).resolve().parent/"checkpoints"
    savedir.mkdir(exist_ok=True)
    ncols = get_terminal_size().columns
    # data loaders
    num_workers = max(4, cpu_count() // 4)
    train_dataset = Dataset(cfg.train_txt, cfg.input_width, cfg.input_height, True)
    val_dataset = Dataset(cfg.val_txt, cfg.input_width, cfg.input_height, False)
    train_loader = DataLoader(train_dataset, cfg.batch_size,
        shuffle=True, collate_fn=collate_fn, drop_last=True,
        num_workers=num_workers, persistent_workers=True)
    val_loader = DataLoader(val_dataset, cfg.batch_size,
        shuffle=False, collate_fn=collate_fn, drop_last=False,
        num_workers=num_workers, persistent_workers=True)
    # model
    if opt.weights is not None:
        model = FastestDetV2(cfg.num_classes,
            load_weights=True, inference_mode=False).to(opt.device)
        model.load_state_dict(torch.load(opt.weights))
        print(f"Loaded detector weights {opt.weights}")
    else:
        model = FastestDetV2(cfg.num_classes,
            load_weights=False, inference_mode=False).to(opt.device)
    proj_name = f"{type(model).__name__.lower()}_{type(model.backbone).__name__.lower()}_{cfg_name}"
    # optimizer
    criterion = DetectorLoss(opt.device)
    params = [{"params": [], "weight_decay": 5e-4}, {"params": [], "weight_decay": 0.0}]
    for n, p in model.named_parameters():                                              params[1 if p.ndim == 1 or n.endswith(".bias") else 0]["params"].append(p)
    optimizer = torch.optim.SGD(params, lr=cfg.learning_rate, momentum=0.949)
    scheduler = MultiStepCosineLR(optimizer, milestones=cfg.milestones)
    if opt.enable_wandb:
        import wandb
        wandb.init(project=proj_name, config={
            "dataset": cfg_name, "epochs": cfg.end_epoch,
            "batch_size": cfg.batch_size, "learning_rate": cfg.learning_rate})
    # train & eval
    step = 0
    warmup_steps = cfg.warmup_epoch * len(train_loader)
    best_ap50 = 0.0
    print(f"Start training for {cfg.end_epoch} epochs")
    for epoch in range(cfg.end_epoch):
        # train
        model.train()
        pbar = tqdm(train_loader, ncols=ncols)
        avg_iou, avg_obj, avg_cls, avg_total, = 0.0, 0.0, 0.0, 0.0
        for ib, (imgs, labels) in enumerate(pbar):
            imgs, labels = imgs.to(opt.device).float() / 255.0, labels.to(opt.device)
            outputs = model(imgs)
            iou, obj, cls, total = criterion(outputs, labels)
            optimizer.zero_grad()
            total.backward()
            optimizer.step()
            # warmup
            if step < warmup_steps:
                curr_lr = cfg.learning_rate * step / warmup_steps
                for g in optimizer.param_groups:
                    g["lr"] = curr_lr
            step += 1
            avg_iou += iou.item()
            avg_obj += obj.item()
            avg_cls += cls.item()
            avg_total += total.item()
            pbar.set_description(f"{epoch}: LR{optimizer.param_groups[0]['lr']:.2} "
                f"IoU{avg_iou/(ib+1):.2f} Obj{avg_obj/(ib+1):.2f} "
                f"Cls{avg_cls/(ib+1):.2f} Tot{avg_total/(ib+1):.2f}")
        if opt.enable_wandb:
            wandb.log({"train/lr": optimizer.param_groups[0]["lr"],
                "train/iou": avg_iou/(ib+1), "train/obj": avg_obj/(ib+1),
                "train/cls": avg_cls/(ib+1), "train/loss": avg_total/(ib+1)}, step=epoch)
        scheduler.step()
        # eval
        if (epoch % 10 != 0 or epoch == 0) and epoch != cfg.end_epoch - 1:
            continue
        with torch.no_grad():
            model.eval()
            model_eval = reparameterize_model(model)
            stats = COCODetectionEvaluator(cfg.names, opt.device).eval(
                val_loader, model_eval, ncols=ncols, colour="green")
            if stats["coco/AP50"] > best_ap50:
                best_ap50 = stats["coco/AP50"]
                torch.save(model_eval.state_dict(), str(savedir/
                    f"{proj_name}_ap50,{best_ap50}_ep{epoch}.pth"))
                torch.save(model.state_dict(), str(savedir/
                    f"{proj_name}_ap50,{best_ap50}_ep{epoch}_unfused.pth"))
            if opt.enable_wandb:
                wandb.log(stats, step=epoch)
    if opt.enable_wandb:
        wandb.finish()
