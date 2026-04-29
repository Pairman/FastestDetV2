import argparse
from pathlib import Path
import sys
import time
import cv2
import torch
_ROOT = str(Path(__file__).resolve().parent)
if not _ROOT in sys.path:
    sys.path.append(_ROOT)
from module.fastestdetv2 import FastestDetV2
from utils.config import Config
from utils.postproc import decode_preds, apply_nms

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda", help="device")
    parser.add_argument("--weights", type=str, default=str(Path(_ROOT)/"weights/fastestdetv2_coco_best.pth"), help=".pt weights")
    parser.add_argument("--configs", type=str, default=str(Path(_ROOT)/"configs/coco.yaml"), help=".yaml configs")
    parser.add_argument("--image", type=str, default=None, help="input image if not exporting")
    parser.add_argument("--result", type=str, default="result.png", help="input image")
    parser.add_argument("--conf-thres", type=float, default=0.65, help="confidence threshold")
    parser.add_argument("--export", action="store_true", help="export to onnx and torchscript")
    opt = parser.parse_args()
    cfg = Config(opt.configs)
    # model
    model = FastestDetV2(num_classes=cfg.num_classes, inference_mode=True).to(opt.device)
    model.load_state_dict(torch.load(opt.weights))
    print(f"Loaded detector weights {opt.weights}")
    model.eval()
    if opt.export:
        model.to("cpu")
        dummy = torch.randn(1, 3, cfg.input_size[1], cfg.input_size[0], device="cpu")
        onnx_path = Path(opt.weights).with_suffix(".onnx")
        ts_path = Path(opt.weights).with_suffix(".pt")
        with torch.no_grad():
            # onnx
            torch.onnx.export(model, dummy, str(onnx_path),
                input_names=["input"], output_names=["output"],
                opset_version=11, do_constant_folding=True)
            # torchscript
            traced = torch.jit.trace(model, dummy)
            traced.save(str(ts_path))
        print(f"Saved to {onnx_path} and {ts_path}")
        sys.exit(0)
    # preproc
    print(f"Processing image {opt.image}")
    img0 = cv2.imread(opt.image)
    img = cv2.resize(img0, cfg.input_size)
    img = torch.from_numpy(img).permute(2,0,1).unsqueeze(0) # HWC->BCHW
    img = img.float().div(255.0).to(opt.device) # norm
    # warmup
    with torch.no_grad():
        model(torch.randn(img.shape, device=opt.device))
    # inference
    print("Starting inference")
    t1 = time.perf_counter()
    with torch.no_grad():
        preds = model(img)
    t2 = time.perf_counter()
    print(f"Inference time: {(t2 - t1) * 1000}ms")
    preds = apply_nms(decode_preds(preds), conf_thres=opt.conf_thres)
    # visualize
    names = cfg.names
    h0, w0, _ = img0.shape
    for box in preds[0]:
        conf, cls = box[4], int(box[5])
        x1, y1 = int(box[0] * w0), int(box[1] * h0)
        x2, y2 = int(box[2] * w0), int(box[3] * h0)
        cv2.rectangle(img0, (x1, y1), (x2, y2), (255, 255, 0), 2)
        label = f"{cls} {names[cls]} {conf:.2f}"
        cv2.putText(img0, label, (x1, y1 - 5), 0, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(img0, label, (x1 + 1, y1 - 5), 0, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imwrite(opt.result, img0)
    print(f"Saved result to {opt.result}")
