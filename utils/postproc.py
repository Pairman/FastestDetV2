import torch
import torchvision

def process_preds(preds, conf_thres=0.25, iou_thres=0.45):
    N, _, H, W = preds.shape
    preds = preds.permute(0, 2, 3, 1) # NCHW->NHWC
    pobj = preds[...,0:1]
    preg = preds[:, :, :, 1:5]
    pcls = preds[:, :, :, 5:]
    # conf, cls
    conf = (pobj.squeeze(-1)**0.6) * (pcls.max(-1)[0]**0.4)
    cls = pcls.argmax(-1)
    # coords
    gy, gx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    gx, gy = gx.to(preds.device), gy.to(preds.device)
    bcx = (preg[...,0].tanh() + gx) / W
    bcy = (preg[...,1].tanh() + gy) / H
    bw, bh = preg[...,2].sigmoid(), preg[...,3].sigmoid()
    x1, y1 = bcx - 0.5*bw, bcy - 0.5*bh
    x2, y2 = bcx + 0.5*bw, bcy + 0.5*bh
    boxes = torch.stack([x1, y1, x2, y2, conf, cls.float()], dim=-1)
    boxes = boxes.reshape(N, H*W, 6)
    # nms
    out = []
    for p in boxes:
        pb = p[p[:,4] > conf_thres]
        if len(pb) == 0:
            out.append(torch.zeros((0,6), device=preds.device))
            continue
        keep = torchvision.ops.batched_nms(pb[:,:4], pb[:,4], pb[:,5].long(), iou_thres)
        out.append(pb[keep])
    return out
