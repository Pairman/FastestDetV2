import torch
import numpy as np
from tqdm import tqdm
from utils.postproc import *

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

_stat_names = [
    "coco/AP", "coco/AP50", "coco/AP75",
    "coco/AP_small", "coco/AP_medium", "coco/AP_large",
    "coco/AR_max1", "coco/AR_max10", "coco/AR_max100",
    "coco/AR_small", "coco/AR_medium", "coco/AR_large"
]

class COCODetectionEvaluator():
    def __init__(self, names, device="cuda"):
        self.device = device
        self.names = []
        with open(names) as f:
            for line in f.readlines():
                self.names.append(line.strip())
    
    def _eval(self, gts, dts):
        # create ground truths
        coco_gt = COCO()
        coco_gt.dataset = {
            "images": [{"id": i} for i in range(len(gts))],
            "annotations": [
                {
                    "image_id": i, "category_id": int(gts[i][j, 0]),
                    "bbox": [
                        float(gts[i][j, 1]),
                        float(gts[i][j, 2]),
                        float(gts[i][j, 3] - gts[i][j, 1]),
                        float(gts[i][j, 4] - gts[i][j, 2])
                    ],
                    "area": float(
                        (gts[i][j, 3] - gts[i][j, 1]) *
                        (gts[i][j, 4] - gts[i][j, 2])
                    ),
                    "id": k + 1, "iscrowd": 0,
                } for k, (i, j) in enumerate(
                    (i, j) for i, gt in enumerate(gts)
                    for j in range(gt.shape[0])
                )
            ],
            "categories": [
                {"id": i, "supercategory": c, "name": c}
                for i, c in enumerate(self.names)
            ]
        }
        coco_gt.createIndex()
        # create detections
        coco_dt = COCO()
        coco_dt.dataset = {
            "images": [{"id": i} for i in range(len(dts))],
            "annotations": [
                {
                    "image_id": i, "category_id": int(dts[i][j, 0]),
                    "score": float(dts[i][j, 1]),
                    "bbox": [
                        float(dts[i][j, 2]),
                        float(dts[i][j, 3]),
                        float(dts[i][j, 4] - dts[i][j, 2]),
                        float(dts[i][j, 5] - dts[i][j, 3])
                    ],
                    "area": float(
                        (dts[i][j, 4] - dts[i][j, 2]) *
                        (dts[i][j, 5] - dts[i][j, 3])
                    ),
                    "id": k + 1,
                } for k, (i, j) in enumerate(
                    (i, j) for i, dt in enumerate(dts)
                    for j in range(dt.shape[0])
                )
            ],
            "categories": [
                {"id": i, "supercategory": c, "name": c}
                for i, c in enumerate(self.names)
            ]
        }
        coco_dt.createIndex()
        # eval
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        return dict(zip(_stat_names, coco_eval.stats))

    def eval(self, val_loader, model, **kwargs):
        gts, dts = [], []
        pbar = tqdm(val_loader, **kwargs)
        for imgs, labels in pbar:
            imgs = imgs.to(self.device).float() / 255.0
            with torch.no_grad():
                preds = model(imgs)
                output = process_preds(preds, conf_thres=0.001)
            n, _, h, w = imgs.shape
            # detections
            for p in output:
                dtboxes = []
                for b in p:
                    b = b.cpu().numpy()
                    score = b[4]
                    category = b[5]
                    x1, y1, x2, y2 = b[:4] * [w, h, w, h]
                    dtboxes.append([category, score, x1, y1, x2, y2])
                dts.append(np.array(dtboxes))
            # ground truths
            for i in range(n):
                gtboxes = []
                for l in labels:
                    if l[0] == i:
                        l = l.cpu().numpy()
                        category = l[1]
                        bcx, bcy, bw, bh = l[2:] * [w, h, w, h]
                        x1, y1 = bcx - 0.5 * bw, bcy - 0.5 * bh
                        x2, y2 = bcx + 0.5 * bw, bcy + 0.5 * bh
                        gtboxes.append([category, x1, y1, x2, y2])
                gts.append(np.array(gtboxes))
        return self._eval(gts, dts)
