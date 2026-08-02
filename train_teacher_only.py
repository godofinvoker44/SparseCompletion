"""
train_teacher_only.py

Train the student FULLY on the teacher depth (Teacher v2, degree-2 calibrated).
Ground-truth sensor depth is NOT used for the depth loss at all — the teacher
is the only depth target, on ALL valid teacher pixels (not just holes).

Rationale: Teacher v2 (degree-2 calibration) reaches delta1 ~0.84 vs GT, higher
than the GT-trained student (~0.737). This experiment tests whether imitating a
well-calibrated, dense teacher can beat sparse-GT supervision.

Detection / keypoint / light / height still train on their own GT labels
(only the DEPTH target is teacher-only).

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
from loss import DEPTH_LOSSES
from box_ops import bbox_regression_loss, mean_iou

# reuse the exact same helpers as train.py for consistency
from train import make_bool_mask, detection_loss, height_loss


def depth_loss_teacher_only(depth_fn, pred, batch):
    """Depth loss computed ONLY against the teacher, on all valid teacher pixels.
    GT is ignored. Returns (loss, gt_dummy, teacher_value)."""
    l_teacher = torch.tensor(0.0, device=pred.device)
    if batch["has_teacher"].sum() > 0:
        sel = batch["has_teacher"].view(-1) > 0.5
        if sel.any():
            teacher_sel = batch["teacher"][sel].clamp(min=1e-3)
            tmask = torch.ones_like(teacher_sel, dtype=torch.bool)   # ALL teacher pixels
            l_teacher = depth_fn(pred[sel], teacher_sel, mask=tmask, interpolate=False)
    val = float(l_teacher.detach() if torch.is_tensor(l_teacher) else l_teacher)
    return l_teacher, 0.0, val


def train_teacher_only(cfg):
    device = cfg["device"]
    name = f"train_teacheronly_{cfg['encoder_name']}_{cfg['depth_loss_name']}"

    train_ds = CrossingDataset(cfg["csv_path"], "train", cfg["coco_json"],
                                use_teacher=True, max_objects=cfg["max_objects"])
    print(f"train: {len(train_ds)} samples | TEACHER-ONLY depth | "
            f"encoder={cfg['encoder_name']} | depth_loss={cfg['depth_loss_name']}")
    dl = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                    num_workers=cfg["num_workers"], collate_fn=collate_fn, drop_last=True)

    model = MultiTaskNet(encoder_name=cfg["encoder_name"], pretrained=cfg["pretrained"],
                            max_objects=cfg["max_objects"]).to(device)
    depth_fn = DEPTH_LOSSES[cfg["depth_loss_name"]]().to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    steps = len(dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg["lr"] * 10, epochs=cfg["epochs"],
        steps_per_epoch=steps, pct_start=0.3)

    log_w = log_f = None
    if cfg["log_csv"]:
        os.makedirs(os.path.dirname(cfg["log_csv"]) or ".", exist_ok=True)
        log_f = open(cfg["log_csv"], "w", newline="")
        log_w = csv.writer(log_f)
        log_w.writerow(["epoch", "loss_total", "depth_teacher",
                        "obj", "bbox", "iou", "kpt", "light", "height", "lr", "det_active"])

    for epoch in range(cfg["epochs"]):
        model.train()
        use_det = epoch >= cfg["detection_start_epoch"]
        agg = {k: 0.0 for k in ["total", "teacher", "obj", "bbox", "iou",
                                "kpt", "light", "height"]}
        n = 0
        for batch in dl:
            for k in batch:
                if torch.is_tensor(batch[k]):
                    batch[k] = batch[k].to(device)

            out = model(batch["image"], height_cm=batch["height_cm"])
            l_depth, _, l_teacher = depth_loss_teacher_only(depth_fn, out["depth"], batch)
            loss = cfg["w_depth"] * l_depth
            agg["teacher"] += l_teacher

            if use_det:
                l_obj, l_bbox, l_kpt, l_light, b_l1, b_ciou, iou = detection_loss(
                    out, batch, cfg["w_bbox_l1"], cfg["w_bbox_ciou"])
                loss = (loss + cfg["w_obj"]*l_obj + cfg["w_bbox"]*l_bbox
                        + cfg["w_kpt"]*l_kpt + cfg["w_light"]*l_light)
                agg["obj"] += l_obj.item(); agg["bbox"] += l_bbox.item(); agg["iou"] += iou
                agg["kpt"] += l_kpt.item(); agg["light"] += l_light.item()
                if cfg["predict_height"]:
                    l_h = height_loss(out, batch)
                    loss = loss + cfg["w_height"] * l_h
                    agg["height"] += l_h.item()

            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step(); sched.step()
            agg["total"] += loss.item(); n += 1

        lr_now = opt.param_groups[0]["lr"]
        row = {k: agg[k]/n for k in agg}
        print(f"epoch {epoch:02d} | total {row['total']:.3f} | depth_teacher {row['teacher']:.3f}"
                + (f" | bbox {row['bbox']:.3f} IoU {row['iou']:.3f} kpt {row['kpt']:.3f} "
                    f"light {row['light']:.3f}" if use_det else "")
                + f" | lr {lr_now:.1e} det={use_det}")

        if log_w:
            log_w.writerow([epoch, row["total"], row["teacher"], row["obj"], row["bbox"],
                            row["iou"], row["kpt"], row["light"], row["height"],
                            lr_now, int(use_det)])
            log_f.flush()

        if cfg["ckpt_dir"]:
            os.makedirs(cfg["ckpt_dir"], exist_ok=True)
            torch.save(model.state_dict(), os.path.join(cfg["ckpt_dir"], f"{name}_last.pt"))

    if log_f:
        log_f.close()
    return model


if __name__ == "__main__":
    encoder = "efficientnet_b0"
    loss_name = "softdelta"
    cfg = {
        "csv_path":   "dataset.csv",           # make sure this points to teacher v2 paths
        "coco_json":  "new_ds/train/_annotations_coco.json",

        "encoder_name": encoder,
        "pretrained": True,
        "predict_height": True,
        "max_objects": 4,
        "depth_loss_name": loss_name,

        "detection_start_epoch": 5,
        "epochs": 50, "batch_size": 8, "lr": 3e-4, "num_workers": 4,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "ckpt_dir": "checkpoints",
        "log_csv": f"logs/train_teacheronly_{encoder}_{loss_name}.csv",

        "w_depth": 1.0,
        "w_obj": 1.0,
        "w_bbox": 1.0, "w_bbox_l1": 1.0, "w_bbox_ciou": 1.0,
        "w_kpt": 5.0, "w_light": 1.0, "w_height": 1.0,
    }
    train_teacher_only(cfg)