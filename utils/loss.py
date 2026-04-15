import math
import torch
import torch.nn as nn

def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert boxes from center format to corner format."""
    half_wh = boxes[..., 2:4] * 0.5
    return torch.cat((boxes[..., :2] - half_wh, boxes[..., :2] + half_wh), dim=-1)

def pairwise_iou(box1: torch.Tensor, box2: torch.Tensor, eps=1e-7) -> torch.Tensor:
    """Compute pairwise IoU matrix between two box sets."""
    # convert boxes to corner format
    box1 = box_cxcywh_to_xyxy(box1)[:, None, :]
    box2 = box_cxcywh_to_xyxy(box2)[None, :, :]
    # intersection area
    lt = torch.maximum(box1[..., :2], box2[..., :2])
    rb = torch.minimum(box1[..., 2:], box2[..., 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    # union area
    area1 = (box1[..., 2] - box1[..., 0]).clamp(min=0) * \
        (box1[..., 3] - box1[..., 1]).clamp(min=0)
    area2 = (box2[..., 2] - box2[..., 0]).clamp(min=0) * \
        (box2[..., 3] - box2[..., 1]).clamp(min=0)
    return inter / (area1 + area2 - inter + eps)

def bbox_iou(box1, box2, eps=1e-7):
    """Compute SIoU score and raw IoU for matched boxes."""
    # split box coordinates
    box1, box2 = box1.t(), box2.t()
    b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
    b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
    b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
    b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2
    # base iou
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
        (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union
    # distance penalty
    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
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
    # shape penalty
    omiga_w = torch.abs(w1 - w2) / torch.max(w1, w2)
    omiga_h = torch.abs(h1 - h2) / torch.max(h1, h2)
    shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + \
        torch.pow(1 - torch.exp(-1 * omiga_h), 4)
    siou_score = iou - 0.5 * (distance_cost + shape_cost)
    return siou_score, iou

def decode_preds(preds):
    """Decode predictions into objectness, class, and boxes."""
    pred = preds.permute(0, 2, 3, 1)
    # split prediction branches
    pobj = pred[..., 0].clamp(min=1e-4, max=1 - 1e-4)
    preg = pred[..., 1:5]
    pcls = pred[..., 5:].clamp(min=1e-9)
    _, H, W, _ = pred.shape
    # build feature map grid
    gy, gx = torch.meshgrid(
        torch.arange(H, device=pred.device),
        torch.arange(W, device=pred.device),
        indexing="ij")
    # decode box prediction
    pbox = torch.empty((*pred.shape[:3], 4),
        device=pred.device, dtype=pred.dtype)
    pbox[..., 0] = preg[..., 0].tanh() + gx
    pbox[..., 1] = preg[..., 1].tanh() + gy
    pbox[..., 2] = preg[..., 2].sigmoid() * W
    pbox[..., 3] = preg[..., 3].sigmoid() * H
    return pobj, pcls, pbox, gx, gy

class DetectorLoss(nn.Module):
    def __init__(self, device="cpu", center_radius=2.5,
        candidate_topk=10, pre_candidate_topk=32, iou_weight=3.0):
        """Initialize detector loss and matching hyper-parameters."""
        super().__init__()
        self.device = device
        self.center_radius = center_radius
        self.candidate_topk = candidate_topk
        self.pre_candidate_topk = pre_candidate_topk
        self.iou_weight = iou_weight
        self.BCEcls = nn.NLLLoss()
        self.BCEobj = nn.SmoothL1Loss(reduction="none")

    def get_candidate_mask(self, gt_box, points, W, H):
        """Select feature map points by box and center prior."""
        gt_xyxy = box_cxcywh_to_xyxy(gt_box)
        px = points[:, 0][None, :]
        py = points[:, 1][None, :]
        # candidate points in gt box
        in_boxes = (px >= gt_xyxy[:, 0:1]) & (px <= gt_xyxy[:, 2:3]) & \
            (py >= gt_xyxy[:, 1:2]) & (py <= gt_xyxy[:, 3:4])
        # center prior
        center_radius = gt_box.new_tensor(self.center_radius)
        center_xy = gt_box[:, :2]
        in_centers = (px >= center_xy[:, 0:1] - center_radius) & \
            (px <= center_xy[:, 0:1] + center_radius) & \
            (py >= center_xy[:, 1:2] - center_radius) & \
            (py <= center_xy[:, 1:2] + center_radius)
        candidate_mask = in_boxes.any(dim=0) | in_centers.any(dim=0)
        # fallback to gt center point
        if not candidate_mask.any():
            gi = gt_box[:, 0].long().clamp_(0, W - 1)
            gj = gt_box[:, 1].long().clamp_(0, H - 1)
            candidate_mask[(gj * W + gi).unique()] = True
        return candidate_mask, in_boxes, in_centers

    def filter_candidates(self, gt_box, points, candidate_index):
        """Keep only geometrically close candidates before cls cost."""
        if candidate_index.numel() <= self.pre_candidate_topk:
            return candidate_index
        candidate_points = points[candidate_index]
        gt_centers = gt_box[:, :2]
        # center distance filter
        center_dist = ((gt_centers[:, None, :] - candidate_points[None, :, :]) ** 2).sum(dim=-1)
        pre_topk = min(self.pre_candidate_topk, candidate_index.numel())
        filter_index = center_dist.topk(pre_topk, dim=1, largest=False).indices
        return candidate_index[filter_index.reshape(-1).unique()]

    def compute_matching_cost(self, gt_box, gt_cls, pred_box, pred_obj, pred_cls, in_boxes, in_centers):
        """Build SimOTA matching cost for filtered candidates."""
        # matching cost
        pairwise_ious = pairwise_iou(gt_box, pred_box).clamp(min=1e-7)
        cls_prob = pred_cls[:, gt_cls].permute(1, 0)
        obj_prob = pred_obj.unsqueeze(0)
        cls_cost = -(cls_prob * obj_prob).clamp(min=1e-7).log()
        iou_cost = -pairwise_ious.log()
        in_boxes_and_center = in_boxes & in_centers
        cost = cls_cost + self.iou_weight * iou_cost + \
            (~in_boxes_and_center).float() * 100000.0
        return pairwise_ious, cost

    def dynamic_k_matching(self, cost, pairwise_ious):
        """Assign positives with dynamic-k matching."""
        # dynamic k matching
        matching = torch.zeros_like(cost, dtype=torch.bool)
        topk = min(self.candidate_topk, pairwise_ious.size(1))
        dynamic_ks = pairwise_ious.topk(topk, dim=1).values.sum(dim=1).int().clamp(min=1)
        for gt_idx in range(pairwise_ious.size(0)):
            _, pos_idx = torch.topk(cost[gt_idx], k=int(dynamic_ks[gt_idx].item()), largest=False)
            matching[gt_idx, pos_idx] = True
        # resolve one-to-many conflicts
        matched_gts = matching.sum(dim=0)
        if (matched_gts > 1).any():
            conflict = matched_gts > 1
            min_cost_gt = cost[:, conflict].argmin(dim=0)
            matching[:, conflict] = False
            matching[min_cost_gt, conflict] = True
        fg_mask = matching.any(dim=0)
        # fallback to lowest cost match
        if not fg_mask.any():
            min_cost_idx = cost.argmin(dim=1)
            matching[torch.arange(pairwise_ious.size(0), device=cost.device), min_cost_idx] = True
            fg_mask = matching.any(dim=0)
        return matching, fg_mask

    @torch.no_grad()
    def build_target(self, preds, targets):
        """Assign ground truths to feature map points."""
        device = preds.device
        B, _, H, W = preds.shape
        # flatten prediction map
        pobj, pcls, pbox, gx, gy = decode_preds(preds)
        pbox = pbox.reshape(B, H * W, 4)
        pobj = pobj.reshape(B, H * W)
        pcls = pcls.reshape(B, H * W, -1)
        points = torch.stack((gx.reshape(-1).float(), gy.reshape(-1).float()), dim=-1)
        empty_long = torch.zeros(0, dtype=torch.long, device=device)
        empty_box = torch.zeros((0, 4), dtype=preds.dtype, device=device)
        if targets.numel() == 0:
            return empty_box, empty_long, (empty_long, empty_long, empty_long)
        batch_index, grid_x, grid_y, gt_box, gt_cls = [], [], [], [], []
        for ib in range(B):
            gt = targets[targets[:, 0].long() == ib]
            if gt.numel() == 0:
                continue
            gt_cls_i = gt[:, 1].long()
            scale = gt.new_tensor([W, H, W, H])
            gt_box_i = gt[:, 2:6] * scale
            # simota
            # https://github.com/open-mmlab/mmyolo/blob/8c4d9dc/mmyolo/models/task_modules/assigners/batch_yolov7_assigner.py#L301
            # geometric filter before cost calculation
            candidate_mask, in_boxes, in_centers = self.get_candidate_mask(gt_box_i, points, W, H)
            candidate_index = candidate_mask.nonzero(as_tuple=False).squeeze(1)
            candidate_index = self.filter_candidates(gt_box_i, points, candidate_index)
            pairwise_ious, cost = self.compute_matching_cost(gt_box_i, gt_cls_i,
                pbox[ib, candidate_index], pobj[ib, candidate_index], pcls[ib, candidate_index],
                in_boxes[:, candidate_index], in_centers[:, candidate_index])
            # dynamic k matching
            matching, fg_mask = self.dynamic_k_matching(cost, pairwise_ious)
            matched_gt_idx = matching[:, fg_mask].float().argmax(dim=0)
            matched_index = candidate_index[fg_mask]
            batch_index.append(torch.full((matched_index.numel(),), ib, dtype=torch.long, device=device))
            grid_x.append((matched_index % W).long())
            grid_y.append((matched_index // W).long())
            gt_box.append(gt_box_i[matched_gt_idx])
            gt_cls.append(gt_cls_i[matched_gt_idx])
        if len(gt_box) == 0:
            return empty_box, empty_long, (empty_long, empty_long, empty_long)
        return torch.cat(gt_box, dim=0), torch.cat(gt_cls, dim=0), \
            (torch.cat(batch_index, dim=0), torch.cat(grid_x, dim=0), torch.cat(grid_y, dim=0))

    def get_loss(self, preds, gt_info, eps=1e-9):
        """Compute losses for one prediction branch."""
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
        l_iou = torch.zeros(1, device=preds.device)
        l_obj = torch.zeros(1, device=preds.device)
        l_cls = torch.zeros(1, device=preds.device)
        if gt_box.numel() > 0:
            b, gx, gy = ps_index
            ptbox = torch.empty_like(gt_box)
            ptbox[:, 0] = preg[b, gy, gx][:, 0].tanh() + gx
            ptbox[:, 1] = preg[b, gy, gx][:, 1].tanh() + gy
            ptbox[:, 2] = preg[b, gy, gx][:, 2].sigmoid() * W
            ptbox[:, 3] = preg[b, gy, gx][:, 3].sigmoid() * H
            # siou loss
            siou_score, raw_iou = bbox_iou(ptbox, gt_box)
            l_iou = (1.0 - siou_score).mean()
            # classification loss
            l_cls = self.BCEcls(torch.log(pcls[b, gy, gx].clamp(min=1e-9)), gt_cls)
            # iou-aware objectness
            tobj[b, gy, gx] = raw_iou.float().detach()
            n = torch.bincount(b, minlength=B)
            factor[b, gy, gx] = (1.0 / (n[b].float() / (H * W) + eps)) * 0.25
        # objectness loss
        l_obj = (self.BCEobj(pobj, tobj) * factor).mean()
        return l_iou, l_obj, l_cls

    def forward(self, preds, targets):
        """Compute total detector loss with optional auxiliary branch."""
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
