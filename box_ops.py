"""
box_ops.py

Utilitas bbox: konversi format, IoU, CIoU, dan loss.
Format bbox model: (cx, cy, w, h) ternormalisasi [0,1].

Fungsi:
    cxcywh_to_xyxy(b)         : ubah ke (x1,y1,x2,y2)
    box_iou(a, b)             : IoU per pasangan (elementwise), return [N]
    ciou_loss(pred, tgt)      : 1 - CIoU per pasangan, return [N]
    bbox_regression_loss(...) : gabungan L1 + CIoU (dengan mask slot terisi)
    mean_iou(...)             : IoU rata-rata (metrik monitoring, bukan loss)
"""

import torch


def cxcywh_to_xyxy(b):
    cx, cy, w, h = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    x1 = cx - w / 2; y1 = cy - h / 2
    x2 = cx + w / 2; y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


def box_iou(a, b, eps=1e-7):
    """IoU elementwise antara a dan b (format xyxy), shape [...,4] -> [...]."""
    ax1, ay1, ax2, ay2 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx1, by1, bx2, by2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]

    inter_x1 = torch.max(ax1, bx1); inter_y1 = torch.max(ay1, by1)
    inter_x2 = torch.min(ax2, bx2); inter_y2 = torch.min(ay2, by2)
    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

    area_a = (ax2 - ax1).clamp(min=0) * (ay2 - ay1).clamp(min=0)
    area_b = (bx2 - bx1).clamp(min=0) * (by2 - by1).clamp(min=0)
    union = area_a + area_b - inter + eps
    return inter / union


def ciou_loss(pred, tgt, eps=1e-7):
    """1 - CIoU per pasangan. pred,tgt format cxcywh [...,4] -> loss [...]."""
    p = cxcywh_to_xyxy(pred); t = cxcywh_to_xyxy(tgt)
    iou = box_iou(p, t, eps)

    # jarak antar pusat (pakai cxcywh langsung)
    pcx, pcy = pred[..., 0], pred[..., 1]
    tcx, tcy = tgt[..., 0], tgt[..., 1]
    center_dist = (pcx - tcx) ** 2 + (pcy - tcy) ** 2

    # diagonal kotak pembungkus terkecil
    x1 = torch.min(p[..., 0], t[..., 0]); y1 = torch.min(p[..., 1], t[..., 1])
    x2 = torch.max(p[..., 2], t[..., 2]); y2 = torch.max(p[..., 3], t[..., 3])
    diag = (x2 - x1) ** 2 + (y2 - y1) ** 2 + eps

    # konsistensi rasio aspek
    pw, ph = pred[..., 2].clamp(min=eps), pred[..., 3].clamp(min=eps)
    tw, th = tgt[..., 2].clamp(min=eps), tgt[..., 3].clamp(min=eps)
    v = (4 / (3.14159265 ** 2)) * (torch.atan(tw / th) - torch.atan(pw / ph)) ** 2
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)

    return 1 - iou + center_dist / diag + alpha * v


def bbox_regression_loss(pred, tgt, slot_mask, w_l1=1.0, w_ciou=1.0):
    """
    pred, tgt : [B,M,4] cxcywh
    slot_mask : [B,M] 1=slot terisi
    Return: (loss_total, l1_val, ciou_val)  -- val untuk logging
    """
    m = slot_mask.unsqueeze(-1)                       # [B,M,1]
    denom = slot_mask.sum().clamp(min=1.0)

    l1 = (torch.abs(pred - tgt) * m).sum() / denom / 4.0

    ciou = ciou_loss(pred, tgt)                       # [B,M]
    ciou = (ciou * slot_mask).sum() / denom

    total = w_l1 * l1 + w_ciou * ciou
    return total, l1.item(), ciou.item()


@torch.no_grad()
def mean_iou(pred, tgt, slot_mask, eps=1e-7):
    """IoU rata-rata di slot terisi (metrik monitoring). Return float."""
    p = cxcywh_to_xyxy(pred); t = cxcywh_to_xyxy(tgt)
    iou = box_iou(p, t, eps)                           # [B,M]
    denom = slot_mask.sum().clamp(min=1.0)
    return ((iou * slot_mask).sum() / denom).item()