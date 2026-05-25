import argparse
from os import cpu_count
from pathlib import Path
from prefetch_generator import BackgroundGenerator
from shutil import get_terminal_size
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
_ROOT = str(Path(__file__).resolve().parents[1])
if not _ROOT in sys.path:
    sys.path.append(_ROOT)
from module.qamobileone import QAMobileOneClassifier
from utils.ema import EMA
from utils.reparam import reparameterize_model

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

def get_train_transform(epoch: int, end_epoch: int):
    imgsz = int(160 + 64 * min(1, epoch / (0.8 * end_epoch)) ** 0.5)
    return transforms.Compose([
        transforms.RandomResizedCrop(imgsz, (0.6, 1.0), (0.7, 1.3)),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.02),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class ImageNetEvaluator:
    """Track Top-1 / Top-5 accuracy."""
    def __init__(self):
        self.top1, self.top5, self.total = 0, 0, 0

    @torch.no_grad()
    def update(self, outputs: torch.Tensor, targets: torch.Tensor):
        maxk = (1, 5)
        batch_size = targets.size(0)
        _, pred = outputs.topk(maxk[-1], dim=1, largest=True, sorted=True)
        pred = pred.t()  # (5, B)
        correct = pred.eq(targets.view(1, -1).expand_as(pred))
        self.top1 += correct[:1].reshape(-1).float().sum().item()
        self.top5 += correct[:5].reshape(-1).float().sum().item()
        self.total += batch_size

    def get(self):
        if self.total == 0:
            return 0.0, 0.0
        return self.top1 / self.total, self.top5 / self.total

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", type=str, default=str(Path("/data_ssd/datasets/imagenet")), help="imagenet-1k dataset root")
    parser.add_argument("--device", type=str, default="cuda", help="device")
    parser.add_argument("--enable-wandb", action="store_true", help="log to wandb")
    cfg = {"name": "imagenet1k", "num_blocks_per_stage": [4, 8, 4], "base_channels": [24, 48, 96, 192],
        "end_epoch": 300, "batch_size": 256, "learning_rate": 0.1, "warmup_epoch": 5}
    opt = parser.parse_args()
    ncols = get_terminal_size().columns
    savedir = Path(_ROOT).resolve()/"weights"
    savedir.mkdir(exist_ok=True)
    # dataloaders
    num_workers = max(6, cpu_count() // 2)
    train_dataset = datasets.ImageNet(root=opt.datadir, split="train")
    val_dataset = datasets.ImageNet(root=opt.datadir, split="val", transform=val_transform)
    train_loader = DataLoaderX(train_dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoaderX(val_dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=num_workers, pin_memory=True)
    # model
    model = QAMobileOneClassifier(num_blocks_per_stage=cfg["num_blocks_per_stage"], base_channels=cfg["base_channels"]).to(opt.device)
    ema = EMA(model, decay=0.9998, device=opt.device)
    proj_name = f"{type(model).__name__.lower()}_{cfg['name']}"
    # optimizer
    train_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    val_criterion = nn.CrossEntropyLoss()
    params = [{"params": [], "weight_decay": 1.2e-4}, {"params": [], "weight_decay": 0.0}]
    for n, p in model.named_parameters():
        params[1 if p.ndim == 1 or n.endswith(".bias") else 0]["params"].append(p)
    optimizer = torch.optim.SGD(params, lr=cfg["learning_rate"], momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["end_epoch"])
    scaler = torch.amp.GradScaler("cuda")
    if opt.enable_wandb:
        import wandb
        wandb.init(project=proj_name, config={
            "dataset": cfg["name"], "epochs": cfg["end_epoch"],
            "batch_size": cfg["batch_size"], "learning_rate": cfg["learning_rate"]})
    # train and eval
    step = 0
    warmup_steps = cfg["warmup_epoch"] * len(train_loader)
    best_top1 = 0.0
    for epoch in range(1, cfg["end_epoch"] + 1):
        # train
        model.train()
        train_loader.dataset.transform = get_train_transform(epoch, cfg["end_epoch"])
        meter = ImageNetEvaluator()
        sum_loss = 0.0
        pbar = tqdm(train_loader, ncols=ncols)
        avg_loss, top1, top5 = 0, 0, 0
        for imgs, labels in pbar:
            imgs, labels = imgs.to(opt.device, non_blocking=True), labels.to(opt.device, non_blocking=True)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                outputs = model(imgs)
                loss = train_criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)
            # warmup
            if step < warmup_steps:
                curr_lr = cfg["learning_rate"] * step / warmup_steps
                for g in optimizer.param_groups:
                    g["lr"] = curr_lr
            step += 1
            meter.update(outputs, labels)
            sum_loss += loss.item() * imgs.size(0)
            avg_loss = sum_loss / meter.total
            top1, top5 = meter.get()
            pbar.set_description(f"{epoch}: "
                f"loss{avg_loss:.2f} topk{top1:.3f},{top5:.3f}")
        if opt.enable_wandb:
            wandb.log({"train/lr": optimizer.param_groups[0]["lr"],
                "train/loss": avg_loss, "train/top1": top1, "train/top5": top5}, step=epoch)
        scheduler.step()
        # eval
        torch.save(ema.model.state_dict(), str(savedir/f"{proj_name}_last_unfused.pth"))
        if epoch % 5 != 0 and epoch != cfg["end_epoch"]:
            continue
        with torch.no_grad():
            ema.model.eval()
            model_eval = reparameterize_model(ema.model)
            meter = ImageNetEvaluator()
            sum_loss = 0.0
            pbar = tqdm(val_loader, ncols=ncols, colour="green")
            for imgs, labels in pbar:
                imgs, labels = imgs.to(opt.device, non_blocking=True), labels.to(opt.device, non_blocking=True)
                outputs = model_eval(imgs)
                loss = val_criterion(outputs, labels)
                meter.update(outputs, labels)
                sum_loss += loss.item() * imgs.size(0)
                avg_loss = sum_loss / meter.total
                top1, top5 = meter.get()
                pbar.set_description(f"[Eval] "
                    f"loss{avg_loss:.2f} topk{top1:.3f},{top5:.3f}")
        if opt.enable_wandb:
            wandb.log({"val/loss": avg_loss, "val/top1": top1, "val/top5": top5}, step=epoch)
        bu_path = savedir/f"{proj_name}_best_unfused.pth"
        if top1 > best_top1:
            best_top1 = top1
            torch.save(ema.model.state_dict(), str(bu_path))
            if opt.enable_wandb:
                wandb.save(str(bu_path), policy="now")
    if opt.enable_wandb:
        wandb.finish()
