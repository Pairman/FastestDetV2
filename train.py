import argparse
from os import cpu_count
from pathlib import Path
from shutil import get_terminal_size
import sys
import torch
from tqdm import tqdm
_ROOT = str(Path(__file__).resolve())
if not _ROOT in sys.path:
    sys.path.append(_ROOT)
from module.fastestdetv2 import FastestDetV2
from utils.config import Config
from utils.datasets import collate_fn, Dataset
from utils.ema import EMA
from utils.evaluator import COCODetectionEvaluator
from utils.loss import DetectorLoss
from utils.lr import MultiStepCosineLR
from utils.reparam import reparameterize_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda", help="device")
    parser.add_argument("--weights", type=str, default=None, help=".pt weights")
    parser.add_argument("--configs", type=str, default=str(Path(__file__).parent/"configs/coco-h.yaml"), help=".yaml configs")
    parser.add_argument("--enable-wandb", action="store_true", help="log to wandb")
    opt = parser.parse_args()
    cfg = Config(opt.configs)
    cfg_name = Path(opt.configs).stem
    savedir = Path(__file__).resolve().parent/"checkpoints"
    savedir.mkdir(exist_ok=True)
    ncols = get_terminal_size().columns
    # data loaders
    num_workers = max(4, cpu_count() // 4)
    train_dataset = Dataset(cfg.train_txt, cfg.input_size, aug=True)
    val_dataset = Dataset(cfg.val_txt, cfg.input_size, aug=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, cfg.batch_size,
        shuffle=True, collate_fn=collate_fn, drop_last=True,
        num_workers=num_workers, persistent_workers=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, cfg.batch_size,
        shuffle=False, collate_fn=collate_fn, drop_last=False,
        num_workers=num_workers, persistent_workers=True)
    # model
    if opt.weights is not None:
        model = FastestDetV2.from_config(cfg,
            load_weights=True, inference_mode=False).to(opt.device)
        model.load_state_dict(torch.load(opt.weights))
        print(f"Loaded detector weights {opt.weights}")
    else:
        model = FastestDetV2.from_config(cfg,
            load_weights=False, inference_mode=False).to(opt.device)
    ema = EMA(model, decay=0.9999, device=opt.device)
    proj_name = f"{type(model).__name__.lower()}_{type(model.backbone).__name__.lower()}_{cfg_name}"
    # optimizer
    criterion = DetectorLoss(opt.device)
    params = [{"params": [], "weight_decay": 5e-4}, {"params": [], "weight_decay": 0.0}]
    for n, p in model.named_parameters():
        params[1 if p.ndim == 1 or n.endswith(".bias") else 0]["params"].append(p)
    optimizer = torch.optim.SGD(params, lr=cfg.learning_rate, momentum=0.949)
    scheduler = MultiStepCosineLR(optimizer, milestones=cfg.milestones)
    # wandb logger
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
        model.is_detach_backbone = epoch < cfg.warmup_epoch
        model.is_detach_agm = epoch >= (cfg.milestones[-1]
            if len(cfg.milestones) > 0 else 0.8 * cfg.end_epoch)
        pbar = tqdm(train_loader, ncols=ncols)
        avg_iou, avg_obj, avg_cls, avg_loss, = 0.0, 0.0, 0.0, 0.0
        for ib, (imgs, labels) in enumerate(pbar):
            imgs, labels = imgs.to(opt.device).float() / 255.0, labels.to(opt.device)
            optimizer.zero_grad()
            outputs = model(imgs)
            iou, obj, cls, loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            ema.update(model)
            # warmup
            if step < warmup_steps:
                curr_lr = cfg.learning_rate * step / warmup_steps
                for g in optimizer.param_groups:
                    g["lr"] = curr_lr
            step += 1
            avg_iou += iou.item()
            avg_obj += obj.item()
            avg_cls += cls.item()
            avg_loss += loss.item()
            pbar.set_description(f"{epoch}: "
                f"iou{avg_iou/(ib+1):.2f} obj{avg_obj/(ib+1):.2f} "
                f"cls{avg_cls/(ib+1):.2f} loss{avg_loss/(ib+1):.2f}")
        if opt.enable_wandb:
            wandb.log({"train/lr": optimizer.param_groups[0]["lr"],
                "train/iou": avg_iou/(ib+1), "train/obj": avg_obj/(ib+1),
                "train/cls": avg_cls/(ib+1), "train/loss": avg_loss/(ib+1)},
                step=epoch)
        scheduler.step()
        # eval
        if (epoch % 10 != 0 or epoch == 0) and epoch != cfg.end_epoch - 1:
            continue
        with torch.no_grad():
            ema.model.eval()
            model_eval = reparameterize_model(ema.model)
            stats = COCODetectionEvaluator(cfg.names, opt.device).eval(
                val_loader, model_eval, ncols=ncols, colour="green")
            if stats["coco/AP50"] > best_ap50:
                best_ap50 = stats["coco/AP50"]
                torch.save(model_eval.state_dict(), str(savedir/
                    f"{proj_name}_ap50,{best_ap50:.6f}_ep{epoch}.pth"))
                torch.save(ema.model.state_dict(), str(savedir/
                    f"{proj_name}_ap50,{best_ap50:.6f}_ep{epoch}_unfused.pth"))
            if opt.enable_wandb:
                wandb.log(stats, step=epoch)
    if opt.enable_wandb:
        wandb.finish()
