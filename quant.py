import argparse
from copy import deepcopy
from os import cpu_count
from pathlib import Path
from shutil import get_terminal_size
import sys
import warnings
import torch
from torch.export import export, export_for_training
from torch.ao.quantization import move_exported_model_to_eval
from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e
from tqdm import tqdm
_ROOT = str(Path(__file__).resolve().parent)
if not _ROOT in sys.path:
    sys.path.append(_ROOT)
from module.fastestdetv2 import FastestDetV2
from utils.config import Config
from utils.datasets import collate_fn, Dataset
from utils.evaluator import COCODetectionEvaluator
from utils.loss import DetectorLoss
from utils.quant import print_quant_stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda", help="device")
    parser.add_argument("--weights", type=str, default=str(Path(_ROOT)/"checkpoints/fastestdetv2_coco_best.pth"), help=".pt weights, reparameterized")
    parser.add_argument("--configs", type=str, default=str(Path(_ROOT)/"configs/coco.yaml"), help=".yaml configs")
    parser.add_argument("--target", type=str, default="arm", help="target platform, arm or x86")
    opt = parser.parse_args()
    cfg = Config(opt.configs)
    cfg_name = Path(opt.configs).stem
    savedir = Path(__file__).resolve().parent/"checkpoints"
    savedir.mkdir(exist_ok=True)
    ncols = get_terminal_size().columns
    warnings.filterwarnings("ignore", message=".*erase_node(.*) on an already erased node.*")
    # data loaders
    num_workers = max(4, cpu_count() // 4)
    calib_dataset = Dataset(cfg.train_txt, cfg.input_size, aug=False)
    # calib_dataset = torch.utils.data.random_split(calib_dataset, [
    #     cfg.batch_size, len(calib_dataset) - cfg.batch_size])[0]
    val_dataset = Dataset(cfg.val_txt, cfg.input_size, aug=False)
    calib_loader = torch.utils.data.DataLoader(calib_dataset,
        shuffle=True, collate_fn=collate_fn, drop_last=True,
        num_workers=num_workers, persistent_workers=True)
    val_loader = torch.utils.data.DataLoader(val_dataset,
        shuffle=False, collate_fn=collate_fn, drop_last=True,
        num_workers=num_workers, persistent_workers=True)
    # model
    model = FastestDetV2(num_classes=cfg.num_classes, inference_mode=True).to(opt.device).eval()
    model.load_state_dict(torch.load(opt.weights))
    print(f"Loaded detector weights {opt.weights}")
    proj_name = f"{type(model).__name__.lower()}_{type(model.backbone).__name__.lower()}_{cfg_name}"
    # quantizer
    dummy_inputs = (torch.randn(1, 3, cfg.input_size[1], cfg.input_size[0],
        device=opt.device),)
    model = export_for_training(model, dummy_inputs).module()
    # quantizer
    if opt.target == "x86":
        import torch.ao.quantization.quantizer.x86_inductor_quantizer as xiq
        qconfig = xiq.get_default_x86_inductor_quantization_config()
        quantizer = xiq.X86InductorQuantizer()
        quantizer.set_module_name_qconfig("backbone", qconfig)
    else:
        import torch.ao.quantization.quantizer.xnnpack_quantizer as xpq
        # XNNPACKQuantizer.set_module_name() is unreliable in torch 2.6.
        # .set_global(), then exclude non-backbone blocks.
        class _SelectiveXNNPACKQuantizer(xpq.XNNPACKQuantizer):
            def set_module_name(self, module_name: str, quantization_config):
                self.module_name_config[module_name] = quantization_config
                return self
        qconfig = xpq.get_symmetric_quantization_config()
        quantizer = _SelectiveXNNPACKQuantizer()
        quantizer.set_global(qconfig)
        for module_name in ("det", "spp", "avg_pool", "upsample"):
            quantizer.set_module_name(module_name, None)
    model = prepare_pt2e(model, quantizer)
    # calibration
    criterion = DetectorLoss(opt.device)
    move_exported_model_to_eval(model)
    print("Start calibration")
    with torch.no_grad():
        pbar = tqdm(calib_loader, ncols=ncols)
        avg_iou, avg_obj, avg_cls, avg_loss, = 0.0, 0.0, 0.0, 0.0
        for ib, (imgs, labels) in enumerate(pbar):
            imgs, labels = imgs.to(opt.device).float() / 255.0, labels.to(opt.device)
            outputs = model(imgs)
            iou, obj, cls, loss = criterion(outputs, labels)
            avg_iou += iou.item()
            avg_obj += obj.item()
            avg_cls += cls.item()
            avg_loss += loss.item()
            pbar.set_description(f"iou{avg_iou/(ib+1):.2f} obj{avg_obj/(ib+1):.2f} "
                f"cls{avg_cls/(ib+1):.2f} loss{avg_loss/(ib+1):.2f}")
    model_quant = convert_pt2e(deepcopy(model.cpu()), fold_quantize=True)
    model_quant = export(model_quant, dummy_inputs)
    print_quant_stats(model_quant.module())
    stats = COCODetectionEvaluator(cfg.names).eval(
        val_loader, model_quant.module(), ncols=ncols, colour="green")
    torch.export.save(model_quant, str(savedir/
        f"{proj_name}_ap50,{stats['coco/AP50']:.6f}_ptq,{opt.target}.pt"))
