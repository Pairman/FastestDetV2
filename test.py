import argparse
from pathlib import Path
import sys
import time
import cv2
import torch
_ROOT = str(Path(__file__).resolve())
if not _ROOT in sys.path:
    sys.path.append(_ROOT)
from module.fastestdetv2 import FastestDetV2
from utils.config import Config
from utils.postproc import process_preds

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda", help="device")
    parser.add_argument("--weights", type=str, default=None, help=".pt weights")
    parser.add_argument("--yaml", type=str, required=True, help=".yaml config")
    parser.add_argument("--image", type=str, required=True, help="input image")
    parser.add_argument("--result", type=str, default="result.png", help="input image")
    parser.add_argument("--conf-thres", type=float, default=0.65, help="confidence threshold")
    opt = parser.parse_args()
    cfg = Config(opt.yaml)
    # model
    model = FastestDetV2(cfg.num_classes, load_weights=True, inference_mode=True).to(opt.device)
    model.load_state_dict(torch.load(opt.weights))
    print(f"Loaded detector weights {opt.weights}")
    model.eval()
    # preproc
    print(f"Processing image {opt.image}")
    img0 = cv2.imread(opt.image)
    img = cv2.resize(img0, (cfg.input_width, cfg.input_height))
    img = torch.from_numpy(img).permute(2,0,1).unsqueeze(0)  # HWC->BCHW
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
    preds = process_preds(preds, conf_thres=opt.conf_thres)
    # visualize
    with open(cfg.names) as f:
        names = [l.strip() for l in f.readlines() if l.strip()]
    h0, w0, _ = img0.shape
    h1, h1 = h0 / cfg.input_height, w0 / cfg.input_width
    for box in preds[0]:
        conf, cls = box[4], int(box[5])
        x1, y1 = int(box[0] * w0), int(box[1] * h0)
        x2, y2 = int(box[2] * w0), int(box[3] * h0)
        cv2.rectangle(img0, (x1, y1), (x2, y2), (255, 255, 0), 1)
        cv2.putText(img0, f"{cls} {names[cls]} {conf:.2f}", (x1, y1 - 5), 0, 0.6, (0, 255, 0), 1)
    print(f"Saving result to {opt.result}")
    cv2.imwrite(opt.result, img0)
