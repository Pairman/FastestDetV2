import math
import torch
import torch.nn as nn

class DetectorLoss(nn.Module):
    def __init__(self, device="cpu"):    
        super().__init__()
        self.device = device
        # obj and cls loss functions
        self.BCEcls = nn.NLLLoss() 
        self.BCEobj = nn.SmoothL1Loss(reduction="none")

    def bbox_iou(self, box1, box2, eps=1e-7):
        """Compute IoU-based loss (SIoU)"""
        box1, box2 = box1.t(), box2.t()
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2
        # iou
        inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
                (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
        union = w1 * h1 + w2 * h2 - inter + eps
        iou = inter / union
        # convex box
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
        # siou loss from https://arxiv.org/pdf/2205.12740.pdf
        s_cw = (b2_x1 + b2_x2 - b1_x1 - b1_x2) * 0.5
        s_ch = (b2_y1 + b2_y2 - b1_y1 - b1_y2) * 0.5
        sigma = torch.pow(s_cw ** 2 + s_ch ** 2, 0.5) + eps
        sin_alpha_1 = torch.abs(s_cw) / sigma
        sin_alpha_2 = torch.abs(s_ch) / sigma
        threshold = pow(2, 0.5) / 2
        sin_alpha = torch.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)
        angle_cost = torch.cos(torch.arcsin(sin_alpha) * 2 - math.pi / 2)
        rho_x = (s_cw / cw) ** 2
        rho_y = (s_ch / ch) ** 2
        gamma = angle_cost - 2
        distance_cost = 2 - torch.exp(gamma * rho_x) - torch.exp(gamma * rho_y)
        omiga_w = torch.abs(w1 - w2) / torch.max(w1, w2)
        omiga_h = torch.abs(h1 - h2) / torch.max(h1, h2)
        shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + \
                     torch.pow(1 - torch.exp(-1 * omiga_h), 4)
        siou_score = iou - 0.5 * (distance_cost + shape_cost)
        return siou_score, iou
        
    def build_target(self, preds, targets):
        """Build ground truths"""
        H = preds.shape[2]
        # gt containers
        gt_box, gt_cls, ps_index = [], [], []
        # grid offsets
        quadrant = torch.tensor([[0, 0], [1, 0], 
                                 [0, 1], [1, 1]], device=self.device)
        if targets.shape[0] > 0:
            # scale to feature map
            scale = torch.ones(6).to(self.device)
            scale[2:] = torch.tensor(preds.shape)[[3, 2, 3, 2]]
            gt = targets * scale
            # repeat for neighbors
            gt = gt.repeat(4, 1, 1)
            # boundary check
            quadrant = quadrant.repeat(gt.size(1), 1, 1).permute(1, 0, 2)
            gij = gt[..., 2:4].long() + quadrant
            j = torch.where(gij < H, gij, 0).min(dim=-1)[0] > 0 
            # positive indices
            gi, gj = gij[j].T
            batch_index = gt[..., 0].long()[j]
            ps_index.append((batch_index, gi, gj))
            # gt
            gbox = gt[..., 2:][j]
            gt_box.append(gbox)
            gt_cls.append(gt[..., 1].long()[j])
        return gt_box, gt_cls, ps_index

    def get_loss(self, preds, gt_info, eps=1e-9):
        """Compute single head detector loss."""
        gt_box, gt_cls, ps_index = gt_info
        pred = preds.permute(0, 2, 3, 1)
        # objectness
        pobj = pred[:, :, :, 0]
        # box regression
        preg = pred[:, :, :, 1:5]
        # class regression
        pcls = pred[:, :, :, 5:]
        B, H, W, _ = pred.shape
        tobj = torch.zeros_like(pobj) 
        factor = torch.ones_like(pobj) * 0.75
        l_iou, l_obj, l_cls = torch.zeros(1, device=self.device), \
            torch.zeros(1, device=self.device), torch.zeros(1, device=self.device)
        if len(gt_box) > 0:
            # box regression loss
            b, gx, gy = ps_index[0]
            ptbox = torch.ones((preg[b, gy, gx].shape)).to(self.device)
            ptbox[:, 0] = preg[b, gy, gx][:, 0].tanh() + gx
            ptbox[:, 1] = preg[b, gy, gx][:, 1].tanh() + gy
            ptbox[:, 2] = preg[b, gy, gx][:, 2].sigmoid() * W
            ptbox[:, 3] = preg[b, gy, gx][:, 3].sigmoid() * H
            # iou loss
            siou_score, raw_iou = self.bbox_iou(ptbox, gt_box[0])
            f = siou_score > siou_score.mean()
            if f.any():
                b_f, gy_f, gx_f = b[f], gy[f], gx[f]
                l_iou =  (1.0 - siou_score[f]).mean()
                # classification loss
                ps = torch.log(pcls[b_f, gy_f, gx_f] + 1e-9)
                l_cls = self.BCEcls(ps, gt_cls[0][f])
                # iou-aware objectness
                tobj[b_f, gy_f, gx_f] = raw_iou[f].float().detach()
            # positive sample balancing
            n = torch.bincount(b, minlength=B)
            factor[b, gy, gx] = (1.0 / (n[b].float() / (H * W) + eps)) * 0.25
        # objectness loss
        l_obj = (self.BCEobj(pobj, tobj) * factor).mean()               
        return l_iou, l_obj, l_cls

    def forward(self, preds, targets):
        if isinstance(preds, tuple):
            preds, aux_preds = preds
        else:
            preds, aux_preds = preds, None
        # assign guidance
        guidance = preds.detach() if aux_preds is None else aux_preds.detach()
        gt_info = self.build_target(guidance, targets)
        # main branch loss
        l_iou_m, l_obj_m, l_cls_m = self.get_loss(preds, gt_info)
        # auxiliary branch loss
        if aux_preds is None:
            l_iou, l_obj, l_cls = l_iou_m, l_obj_m, l_cls_m
        else:
            l_iou_a, l_obj_a, l_cls_a = self.get_loss(aux_preds, gt_info)
            l_iou = l_iou_m + l_iou_a
            l_obj = l_obj_m + l_obj_a
            l_cls = l_cls_m + l_cls_a
        # total loss
        loss = (l_iou * 8.0) + (l_obj * 16.0) + l_cls
        return l_iou, l_obj, l_cls, loss
