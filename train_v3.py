"""
train_v3.py

Pola training v3 (dataset_v3.csv, 5000 frame). Grid 2 x 2:

    depth_source : "gt"     -> depth loss ke GT sensor (piksel valid saja)
                   "pseudo" -> depth loss ke pseudo label Depth-Anything-V2
                               (make_teacher_v3.py), di semua piksel; GT tidak dipakai
    kpt_order    : "raw"       -> keypoint urutan apa adanya dari anotasi
                   "canonical" -> keypoint diurutkan ke [TL, TR, BL, BR] (kpt_order.py)

Bbox / objectness / keypoint / light / height selalu dilatih dari label GT.
Setiap epoch dievaluasi di split val (depth vs GT sensor), setelah training
langsung dievaluasi di split test -> eval_results_v3.csv.

Nama: v3_{depth_source}_{kpt_order}_{encoder}_{loss}
    checkpoints/<nama>_last.pt , logs/<nama>.csv
"""

import os
import csv
import time
import torch
from torch.utils.data import DataLoader

from dataset import CrossingDataset, collate_fn
from model import MultiTaskNet
from loss import SILogLoss, SoftDeltaLoss
from train import depth_loss_fn, detection_loss, height_loss
from train_teacher_only import depth_loss_teacher_only
from evaluate_v3 import evaluate_model, evaluate_ckpt, print_table

VAL_KEYS = ["delta1", "AbsRel", "mean_iou", "light_bal_acc", "kpt_px_idx", "kpt_px_canon", "kpt_spread"]


def run_name(cfg):
    return f"v3_{cfg['depth_source']}_{cfg['kpt_order']}_{cfg['encoder_name']}_{cfg['depth_loss_name']}"


def train(cfg):
    assert cfg["depth_source"] in ("gt", "pseudo")
    name = run_name(cfg)
    device = cfg["device"]
    use_teacher = cfg["depth_source"] == "pseudo"

    ds_kw = dict(max_objects=cfg["max_objects"], depth_min=1e-3, depth_max=20.0,
                 kpt_order=cfg["kpt_order"])
    train_ds = CrossingDataset(cfg["csv_path"], "train", cfg["coco_json"], use_teacher=use_teacher, **ds_kw)
    val_ds = CrossingDataset(cfg["csv_path"], "val", cfg["coco_json"], use_teacher=False, **ds_kw)
    print(f"\n=== {name} ===\ntrain {len(train_ds)} | val {len(val_ds)} | "
          f"depth_source={cfg['depth_source']} | kpt_order={cfg['kpt_order']}")

    dl = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                    num_workers=cfg["num_workers"], collate_fn=collate_fn, drop_last=True,
                    persistent_workers=cfg["num_workers"] > 0)

    torch.manual_seed(cfg["seed"])
    model = MultiTaskNet(encoder_name=cfg["encoder_name"], pretrained=cfg["pretrained"],
                         max_objects=cfg["max_objects"]).to(device)
    depth_fn = {"softdelta": SoftDeltaLoss, "silog": SILogLoss}[cfg["depth_loss_name"]]().to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"] * 10, epochs=cfg["epochs"],
                                                steps_per_epoch=len(dl), pct_start=0.3)

    os.makedirs("logs", exist_ok=True); os.makedirs(cfg["ckpt_dir"], exist_ok=True)
    log_f = open(f"logs/{name}.csv", "w", newline="")
    log_w = csv.writer(log_f)
    log_w.writerow(["epoch", "loss_total", "depth", "obj", "bbox", "iou", "kpt", "light", "height",
                    "lr", "det_active"] + [f"val_{k}" for k in VAL_KEYS] + ["sec"])

    for epoch in range(cfg["epochs"]):
        t0 = time.time()
        model.train()
        use_det = epoch >= cfg["detection_start_epoch"]
        agg = {k: 0.0 for k in ["total", "depth", "obj", "bbox", "iou", "kpt", "light", "height"]}
        n = 0
        for batch in dl:
            for k in batch:
                if torch.is_tensor(batch[k]):
                    batch[k] = batch[k].to(device, non_blocking=True)
            out = model(batch["image"])

            if use_teacher:
                l_depth, _, _ = depth_loss_teacher_only(depth_fn, out["depth"], batch)
            else:
                l_depth, _, _ = depth_loss_fn(depth_fn, out["depth"], batch, w_teacher=0.0)
            loss = cfg["w_depth"] * l_depth
            agg["depth"] += float(l_depth.detach())

            if use_det:
                l_obj, l_bbox, l_kpt, l_light, _, _, iou = detection_loss(
                    out, batch, cfg["w_bbox_l1"], cfg["w_bbox_ciou"])
                l_h = height_loss(out, batch)
                loss = (loss + cfg["w_obj"] * l_obj + cfg["w_bbox"] * l_bbox + cfg["w_kpt"] * l_kpt
                        + cfg["w_light"] * l_light + cfg["w_height"] * l_h)
                agg["obj"] += l_obj.item(); agg["bbox"] += l_bbox.item(); agg["iou"] += iou
                agg["kpt"] += l_kpt.item(); agg["light"] += l_light.item(); agg["height"] += l_h.item()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step(); sched.step()
            agg["total"] += loss.item(); n += 1
            if cfg.get("max_steps") and n >= cfg["max_steps"]:
                break

        row = {k: agg[k] / max(n, 1) for k in agg}
        val = evaluate_model(model, val_ds, device, num_workers=cfg["num_workers"])
        sec = time.time() - t0
        print(f"ep {epoch:02d} | loss {row['total']:.3f} depth {row['depth']:.3f}"
              + (f" kpt {row['kpt']:.4f} iou {row['iou']:.3f}" if use_det else "")
              + f" || val d1 {val['delta1']:.3f} AbsRel {val['AbsRel']:.3f} IoU {val['mean_iou']:.3f}"
              f" kpt_canon {val['kpt_px_canon']:.1f}px spread {val['kpt_spread']:.2f} | {sec:.0f}s", flush=True)
        log_w.writerow([epoch, row["total"], row["depth"], row["obj"], row["bbox"], row["iou"], row["kpt"],
                        row["light"], row["height"], opt.param_groups[0]["lr"], int(use_det)]
                       + [val[k] for k in VAL_KEYS] + [round(sec, 1)])
        log_f.flush()
        torch.save(model.state_dict(), os.path.join(cfg["ckpt_dir"], f"{name}_last.pt"))

    log_f.close()
    return os.path.join(cfg["ckpt_dir"], f"{name}_last.pt")


BASE_CFG = {
    "csv_path": "dataset_v3.csv",
    "coco_json": "ds_v3/_annotations.coco.json",
    "encoder_name": "efficientnet_b0",
    "pretrained": True,
    "max_objects": 4,
    "depth_loss_name": "softdelta",
    "detection_start_epoch": 5,
    "epochs": 50,
    "batch_size": 32,
    "lr": 3e-4,
    "num_workers": 4,
    "seed": 42,
    "device": "cuda",
    "ckpt_dir": "checkpoints",
    "w_depth": 1.0, "w_obj": 1.0, "w_bbox": 1.0, "w_bbox_l1": 1.0, "w_bbox_ciou": 1.0,
    "w_kpt": 5.0, "w_light": 1.0, "w_height": 1.0,
}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth_source", nargs="+", default=["gt", "pseudo"])
    ap.add_argument("--kpt_order", nargs="+", default=["raw", "canonical"])
    ap.add_argument("--encoder", default=BASE_CFG["encoder_name"])
    ap.add_argument("--epochs", type=int, default=BASE_CFG["epochs"])
    ap.add_argument("--max_steps", type=int, default=0, help="smoke test: batasi step per epoch")
    ap.add_argument("--ckpt_dir", default=BASE_CFG["ckpt_dir"])
    args = ap.parse_args()

    results = []
    for ds_src in args.depth_source:          # 1. GT depth langsung, 2. pseudo label DepthAnything
        for kord in args.kpt_order:           # tanpa / dengan info sudut TL,TR,BL,BR
            cfg = {**BASE_CFG, "depth_source": ds_src, "kpt_order": kord,
                   "encoder_name": args.encoder, "epochs": args.epochs,
                   "max_steps": args.max_steps, "ckpt_dir": args.ckpt_dir,
                   "detection_start_epoch": min(BASE_CFG["detection_start_epoch"], args.epochs - 1)}
            ckpt = train(cfg)
            results.append(evaluate_ckpt(ckpt, cfg["csv_path"], cfg["coco_json"], "test"))
            print_table(results[-1:])

    out_csv = "eval_results_v3.csv" if args.ckpt_dir == BASE_CFG["ckpt_dir"] else \
        os.path.join(args.ckpt_dir, "eval_results_v3.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader(); w.writerows(results)
    print("\n=== HASIL TEST ==="); print_table(results); print("->", out_csv)
