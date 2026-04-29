import torch
import torchvision

def decode_preds(preds):
    N, _, H, W = preds.shape
    preds = preds.permute(0, 2, 3, 1).contiguous()  # NCHW -> NHWC
    pobj = preds[..., 0]
    preg = preds[..., 1:5]
    pcls = preds[..., 5:]
    # conf & cls
    cls_scores, cls = pcls.max(-1)
    conf = (pobj ** 0.6) * (cls_scores ** 0.4)
    # grid
    gy, gx = torch.meshgrid(
        torch.arange(H, device=preds.device, dtype=preds.dtype),
        torch.arange(W, device=preds.device, dtype=preds.dtype),
        indexing="ij")
    # decode box
    bcx = (preg[..., 0].tanh() + gx) / W
    bcy = (preg[..., 1].tanh() + gy) / H
    bw = preg[..., 2].sigmoid()
    bh = preg[..., 3].sigmoid()
    # coords
    x1 = bcx - 0.5 * bw
    y1 = bcy - 0.5 * bh
    x2 = bcx + 0.5 * bw
    y2 = bcy + 0.5 * bh
    # boxes
    boxes = torch.stack([x1, y1, x2, y2, conf, cls.to(preds.dtype)], dim=-1)
    boxes = boxes.view(N, H * W, 6)
    return boxes

def apply_nms(boxes, conf_thres=0.25, iou_thres=0.45):
    out = []
    for p in boxes:
        pb = p[p[:, 4] > conf_thres]
        if pb.numel() == 0:
            out.append(torch.zeros((0, 6), device=p.device, dtype=p.dtype))
            continue
        keep = torchvision.ops.batched_nms(pb[:,:4], pb[:,4], pb[:,5].long(), iou_thres)
        out.append(pb[keep])
    return out
