"""
train_two_phase.py

Training dua fase untuk depth (skema pretraining-finetune):
    FASE 1 (epoch 0..phase1_epochs-1): TEACHER SAJA.
        Model belajar struktur depth global yang mulus dari DAv2.
        GT sensor tidak dipakai. One Cycle #1 (max_lr tinggi).
        
    FASE 2 (sisanya): GT SAJA (Versi B).
        Model mempertajam & mengoreksi skala dengan GT sensor sparse.
        Teacher dimatikan total. One Cycle #2 (max_lr lebih rendah = finetune).

Task lain (bbox/keypoint/light/height) aktif di fase 2 saja (mulai
detection_start_epoch dihitung relatif ke awal fase 2), karena fase 1 fokus depth.

Impor helper dari train.py agar tidak duplikasi (detection_loss, height_loss, dll).
"""

import os
import csv
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import CrossingDataset, collate_fn
from model import MultiTaskNet
from loss import SILogLoss, SoftDeltaLoss
from train import make_bool_mask, detection_loss, height_loss


def depth_loss_phase(depth_fn, pred, batch, phase, w_teacher_fill=0.0):
    """
    phase==1: teacher saja (semua piksel teacher yg ada). GT diabaikan.
    phase==2: GT saja. teacher fill-hole opsional (w_teacher_fill>0).
    Return (loss, l_gt_val, l_teacher_val).
    """
    gt_mask = make_bool_mask(batch["depth_mask"])

    if phase == 1:
        # teacher saja
        l_teacher = torch.tensor(0.0, device=pred.device)
        if batch["has_teacher"].sum() > 0:
            sel = batch["has_teacher"].view(-1) > 0.5
            if sel.any():
                teacher_sel = batch["teacher"][sel].clamp(min=1e-3)
                tmask = torch.ones_like(teacher_sel, dtype=torch.bool)
                l_teacher = depth_fn(pred[sel], teacher_sel, mask=tmask, interpolate=False)
        return l_teacher, 0.0, float(l_teacher.detach() if torch.is_tensor(l_teacher) else l_teacher)

    # phase 2: GT saja (+ opsional teacher fill-hole)
    l_gt = depth_fn(pred, batch["depth"], mask=gt_mask, interpolate=False)
    l_teacher = torch.tensor(0.0, device=pred.device)
    if w_teacher_fill > 0 and batch["has_teacher"].sum() > 0:
        sel = batch["has_teacher"].view(-1) > 0.5
        if sel.any():
            teacher_sel = batch["teacher"][sel].clamp(min=1e-3)
            tmask = ~gt_mask[sel]                     # fill-hole
            if tmask.any():
                l_teacher = depth_fn(pred[sel], teacher_sel, mask=tmask, interpolate=False)
                
    return (l_gt + w_teacher_fill * l_teacher,
            float(l_gt.detach()),
            float(l_teacher.detach() if torch.is_tensor(l_teacher) else l_teacher))


def train_two_phase(cfg):
    device = cfg["device"]
    name = f"train2p_{cfg['encoder_name']}_{cfg['depth_loss_name']}"

    ds = CrossingDataset(cfg["csv_path"], "train", cfg["coco_json"],
                            use_teacher=True, max_objects=cfg["max_objects"])   # butuh teacher di fase 1
    print(f"train: {len(ds)} sampel | encoder={cfg['encoder_name']} | "
            f"fase1(teacher)={cfg['phase1_epochs']}ep, fase2(GT)={cfg['epochs']-cfg['phase1_epochs']}ep")
    dl = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True,
                    num_workers=cfg["num_workers"], collate_fn=collate_fn, drop_last=True)

    model = MultiTaskNet(cfg["encoder_name"], pretrained=cfg["pretrained"],
                            max_objects=cfg["max_objects"]).to(device)
    depth_fn = (SoftDeltaLoss() if cfg["depth_loss_name"] == "softdelta" else SILogLoss()).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)

    steps = len(dl)
    p1 = cfg["phase1_epochs"]
    p2 = cfg["epochs"] - p1
    # One Cycle #1 (fase teacher)
    sched1 = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg["lr"] * cfg["phase1_lr_mult"], epochs=p1,
        steps_per_epoch=steps, pct_start=0.3)
    sched2 = None

    os.makedirs("logs", exist_ok=True)
    log_f = open(f"logs/{name}.csv", "w", newline="")
    log_w = csv.writer(log_f)
    log_w.writerow(["epoch", "phase", "total", "depth_gt", "depth_teacher",
                    "obj", "bbox", "kpt", "light", "height", "lr"])

    for epoch in range(cfg["epochs"]):
        model.train()
        phase = 1 if epoch < p1 else 2

        # saat masuk fase 2 pertama kali: reset lr, buat One Cycle #2 (lebih rendah)
        if phase == 2 and sched2 is None:
            for g in opt.param_groups:
                g["lr"] = cfg["lr"]
            sched2 = torch.optim.lr_scheduler.OneCycleLR(
                opt, max_lr=cfg["lr"] * cfg["phase2_lr_mult"], epochs=p2,
                steps_per_epoch=steps, pct_start=0.3)
            print(f"--- masuk FASE 2 (GT saja) di epoch {epoch} ---")

        sched = sched1 if phase == 1 else sched2
        # deteksi aktif di fase 2, setelah beberapa epoch pemanasan depth-GT
        det_start_abs = p1 + cfg["detection_start_epoch"]
        use_det = epoch >= det_start_abs

        agg = {k: 0.0 for k in ["total", "gt", "teacher", "obj", "bbox", "kpt", "light", "height"]}
        n = 0
        for batch in dl:
            for k in batch:
                if torch.is_tensor(batch[k]):
                    batch[k] = batch[k].to(device)

            out = model(batch["image"], height_cm=batch["height_cm"])
            l_depth, l_gt, l_teach = depth_loss_phase(
                depth_fn, out["depth"], batch, phase,
                w_teacher_fill=cfg.get("phase2_teacher_fill", 0.0))
            loss = cfg["w_depth"] * l_depth
            agg["gt"] += l_gt; agg["teacher"] += l_teach

            if use_det:
                l_obj, l_bbox, l_kpt, l_light, _, _, _ = detection_loss(
                    out, batch, cfg["w_bbox_l1"], cfg["w_bbox_ciou"])
                loss = (loss + cfg["w_obj"]*l_obj + cfg["w_bbox"]*l_bbox
                        + cfg["w_kpt"]*l_kpt + cfg["w_light"]*l_light)
                agg["obj"] += l_obj.item(); agg["bbox"] += l_bbox.item()
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
        print(f"epoch {epoch:02d} [fase {phase}] | total {row['total']:.3f} "
                f"| gt {row['gt']:.3f} teach {row['teacher']:.3f}"
                + (f" | bbox {row['bbox']:.3f} kpt {row['kpt']:.3f} light {row['light']:.3f}" if use_det else "")
                + f" | lr {lr_now:.1e}")
        log_w.writerow([epoch, phase, row["total"], row["gt"], row["teacher"],
                        row["obj"], row["bbox"], row["kpt"], row["light"], row["height"], lr_now])
        log_f.flush()

        if cfg["ckpt_dir"]:
            os.makedirs(cfg["ckpt_dir"], exist_ok=True)
            torch.save(model.state_dict(), os.path.join(cfg["ckpt_dir"], f"{name}_last.pt"))

    log_f.close()
    return model


if __name__ == "__main__":
    cfg = {
        "csv_path": "dataset.csv",
        "coco_json": "new_ds/train/_annotations_coco.json",
        "encoder_name": "efficientnet_b0",
        "pretrained": True,
        "depth_loss_name": "softdelta",
        "max_objects": 4,
        "predict_height": True,

        "epochs": 50,
        "phase1_epochs": 25,          # 25 teacher, 25 GT
        "phase1_lr_mult": 10,         # One Cycle #1 max_lr = lr*10
        "phase2_lr_mult": 3,          # One Cycle #2 max_lr = lr*3 (finetune, lebih rendah)
        "phase2_teacher_fill": 0.0,   # Versi B: 0 = teacher mati di fase 2

        "detection_start_epoch": 5,   # relatif ke awal fase 2
        "lr": 3e-4, "batch_size": 32, "num_workers": 4,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "ckpt_dir": "checkpoints",

        "w_depth": 1.0, "w_obj": 1.0, "w_bbox": 1.0,
        "w_bbox_l1": 1.0, "w_bbox_ciou": 1.0, "w_kpt": 5.0, "w_light": 1.0, "w_height": 1.0,
    }
    train_two_phase(cfg)