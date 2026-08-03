"""
train_teacher_only.py

Train the student FULLY on the teacher depth (Teacher v2, degree-2 calibrated).
Ground-truth sensor depth is NOT used for the depth loss at all — the teacher
is the only depth target, on ALL valid teacher pixels (not just holes).

Detection / keypoint / light / height still train on their own GT labels
(only the DEPTH target is teacher-only).

Follows the same structure/conventions as train.py.
Checkpoints: checkpoints/train_teacheronly_{encoder}_{loss}_last.pt
Logs:        logs/train_teacheronly_{encoder}_{loss}.csv
"""

import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import CrossingDataset, collate_fn
from model import MultiTaskNet
from loss import SILogLoss, SoftDeltaLoss
from box_ops import bbox_regression_loss, mean_iou


def make_bool_mask(depth_mask):
    return depth_mask > 0.5


def depth_loss_teacher_only(depth_fn, pred, batch):
    """Depth loss computed ONLY against the teacher, on all valid teacher pixels.
    GT sensor depth is ignored. Returns (loss, gt_dummy, teacher_value)."""
    l_teacher = torch.tensor(0.0, device=pred.device)
    if batch["has_teacher"].sum() > 0:
        sel = batch["has_teacher"].view(-1) > 0.5
        if sel.any():
            teacher_sel = batch["teacher"][sel].clamp(min=1e-3)
            tmask = torch.ones_like(teacher_sel, dtype=torch.bool)   # ALL teacher pixels
            l_teacher = depth_fn(pred[sel], teacher_sel, mask=tmask, interpolate=False)
    return (l_teacher,
            0.0,
            float(l_teacher if isinstance(l_teacher, float) else l_teacher.detach()))


def detection_loss(out, batch, w_bbox_l1, w_bbox_ciou):
    obj_mask = batch["obj_mask"]
    l_obj = F.binary_cross_entropy_with_logits(out["objectness"], obj_mask)

    l_bbox, bbox_l1, bbox_ciou = bbox_regression_loss(
        out["bbox"], batch["bbox"], obj_mask, w_l1=w_bbox_l1, w_ciou=w_bbox_ciou)
    iou = mean_iou(out["bbox"], batch["bbox"], obj_mask)

    vis = batch["keypoints"][..., 2:3]
    l_kpt = (torch.abs(out["keypoints_reg"] - batch["keypoints"][..., :2]) * vis).sum() \
            / vis.sum().clamp(min=1.0) / 2.0

    l_light = F.cross_entropy(out["light"], batch["light"])
    return l_obj, l_bbox, l_kpt, l_light, bbox_l1, bbox_ciou, iou


def height_loss(out, batch):
    log_gt = torch.log(batch["height_cm"].clamp(min=1.0))
    return F.l1_loss(out["log_height"], log_gt)


def train(cfg):
    name = f"train_teacheronly_{cfg['encoder_name']}_{cfg['depth_loss_name']}"

    device = cfg["device"]
    min_depth, max_depth = 1e-3, 20.0
    train_ds = CrossingDataset(cfg["csv_path"], "train", cfg["coco_json"],
                               use_teacher=True, max_objects=cfg["max_objects"],
                               depth_min=min_depth, depth_max=max_depth)

    print(f"train: {len(train_ds)} sampel | TEACHER-ONLY depth (no GT) | "
          f"encoder={cfg['encoder_name']} | depth_loss={cfg['depth_loss_name']} | "
          f"max_objects={cfg['max_objects']}")

    dl = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                    num_workers=cfg["num_workers"], collate_fn=collate_fn, drop_last=True)

    model = MultiTaskNet(encoder_name=cfg["encoder_name"], pretrained=cfg["pretrained"],
                         max_objects=cfg["max_objects"]).to(device)

    if cfg["depth_loss_name"] == "softdelta":
        print("depth loss: SoftDeltaLoss")
        depth_fn = SoftDeltaLoss().to(device)
    elif cfg["depth_loss_name"] == "silog":
        print("depth loss: SILogLoss")
        depth_fn = SILogLoss().to(device)
    else:
        raise ValueError(f"depth_loss_name {cfg['depth_loss_name']} belum diimplementasikan")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg["lr"] * 10, epochs=cfg["epochs"],
        steps_per_epoch=len(dl), pct_start=0.3)

    log_csv = f"logs/{name}.csv"
    os.makedirs(os.path.dirname(log_csv) or ".", exist_ok=True)
    log_f = open(log_csv, "w", newline="")
    log_w = csv.writer(log_f)
    log_w.writerow(["epoch", "loss_total", "depth_teacher",
                    "obj", "bbox", "bbox_l1", "bbox_ciou", "iou",
                    "kpt", "light", "height", "lr", "det_active"])

    for epoch in range(cfg["epochs"]):
        model.train()
        use_det = epoch >= cfg["detection_start_epoch"]

        agg = {k: 0.0 for k in ["total", "teacher", "obj", "bbox",
                                "bbox_l1", "bbox_ciou", "iou", "kpt", "light", "height"]}
        n = 0
        for batch in dl:
            for k in batch:
                if torch.is_tensor(batch[k]):
                    batch[k] = batch[k].to(device)

            out = model(batch["image"], height_cm=batch["height_cm"], use_geometry=False)
            l_depth, _, l_teacher = depth_loss_teacher_only(depth_fn, out["depth"], batch)

            loss = cfg["w_depth"] * l_depth
            agg["teacher"] += l_teacher

            if use_det:
                l_obj, l_bbox, l_kpt, l_light, b_l1, b_ciou, iou = detection_loss(
                    out, batch, cfg["w_bbox_l1"], cfg["w_bbox_ciou"])
                loss = (loss + cfg["w_obj"]*l_obj + cfg["w_bbox"]*l_bbox
                        + cfg["w_kpt"]*l_kpt + cfg["w_light"]*l_light)
                agg["obj"] += l_obj.item(); agg["bbox"] += l_bbox.item()
                agg["bbox_l1"] += b_l1; agg["bbox_ciou"] += b_ciou; agg["iou"] += iou
                agg["kpt"] += l_kpt.item(); agg["light"] += l_light.item()
                if cfg["predict_height"]:
                    l_h = height_loss(out, batch)
                    loss = loss + cfg["w_height"] * l_h
                    agg["height"] += l_h.item()

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            sched.step()
            agg["total"] += loss.item(); n += 1

        lr_now = opt.param_groups[0]["lr"]
        row = {k: agg[k]/n for k in agg}
        print(f"epoch {epoch:02d} | total {row['total']:.3f} | depth_teacher {row['teacher']:.3f}"
              + (f" | bbox {row['bbox']:.3f} (l1 {row['bbox_l1']:.3f} ciou {row['bbox_ciou']:.3f}) "
                 f"IoU {row['iou']:.3f} | obj {row['obj']:.3f} kpt {row['kpt']:.3f} "
                 f"light {row['light']:.3f} height {row['height']:.3f}" if use_det else "")
              + f" | lr {lr_now:.1e} det={use_det}")

        log_w.writerow([epoch, row["total"], row["teacher"],
                        row["obj"], row["bbox"], row["bbox_l1"], row["bbox_ciou"], row["iou"],
                        row["kpt"], row["light"], row["height"], lr_now, int(use_det)])
        log_f.flush()

        if cfg["ckpt_dir"]:
            os.makedirs(cfg["ckpt_dir"], exist_ok=True)
            torch.save(model.state_dict(), os.path.join(cfg["ckpt_dir"], f"{name}_last.pt"))

    log_f.close()
    return model


if __name__ == "__main__":
    loss = "softdelta"
    for name in [
                # 'efficientnet_b0',
                'efficientnet_b3',
                'efficientnet_b5',
            ]:

        cfg = {
            "csv_path":   "dataset_v2.csv",
            "coco_json":  "new_ds/train/_annotations_coco.json",

            "encoder_name": name,
            "pretrained": True,
            "predict_height": True,
            "max_objects": 4,
            "depth_loss_name": loss,

            "detection_start_epoch": 5,

            "epochs": 50,
            "batch_size": 32,
            "lr": 3e-4,
            "num_workers": 4,

            "device": "cuda",
            "ckpt_dir": "checkpoints",

            "w_depth": 1.0,
            "w_obj": 1.0,
            "w_bbox": 1.0,
            "w_bbox_l1": 1.0,
            "w_bbox_ciou": 1.0,
            "w_kpt": 5.0,
            "w_light": 1.0,
            "w_height": 1.0,
        }
        train(cfg)