import os
import re
import glob
import torch
import numpy as np
from torch.utils.data import DataLoader

from dataset import CrossingDataset, collate_fn
from model import MultiTaskNet
from box_ops import cxcywh_to_xyxy, box_iou

import time as _time


@torch.no_grad()
def measure_efficiency(model, device="cpu", img_size=(512, 256), n_warmup=5, n_runs=20):
    """Ukur efisiensi model: params, ukuran, FLOPs, latency, FPS.
    Tidak tergantung checkpoint (murni arsitektur), tapi butuh model di device."""
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())

    tmp = "_size_check_tmp.pt"
    torch.save(model.state_dict(), tmp)
    size_mb = os.path.getsize(tmp) / (1024 ** 2)
    os.remove(tmp)

    gflops = None
    try:
        from thop import profile
        dummy = torch.randn(1, 3, *img_size).to(device)
        macs, _ = profile(model, inputs=(dummy,), verbose=False)
        gflops = macs * 2 / 1e9
    except Exception:
        pass

    dummy = torch.randn(1, 3, *img_size).to(device)
    for _ in range(n_warmup):
        _ = model(dummy)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = _time.time()
    for _ in range(n_runs):
        _ = model(dummy)
    if device == "cuda":
        torch.cuda.synchronize()
    ms = (_time.time() - t0) / n_runs * 1000

    return {
        "params_M": n_params / 1e6,
        "size_MB": size_mb,
        "GFLOPs": gflops if gflops is not None else float("nan"),
        "latency_ms": ms,
        "FPS": 1000.0 / ms,
    }



KNOWN_ENCODERS = [
    "efficientnet_b0", "efficientnet_b3", "efficientnet_b5",
    "mobilenetv4_conv_small", "mobilenetv3_small_100", "mobilenetv3_large_100",
    "regnety_002", "regnetx_002", "resnet10t", "resnet18",
]

# zona jarak (meter), non-overlap: dekat / sedang / jauh
ZONES = [(0.0, 5.0), (5.0, 10.0), (10.0, 20.0)]


def parse_ckpt_name(fname):
    name = fname
    if name.endswith(".pt"):
        name = name[:-3]
    if name.startswith("train_"):
        name = name[len("train_"):]
    if name.endswith("_last"):
        name = name[:-len("_last")]
    encoder = None
    for enc in sorted(KNOWN_ENCODERS, key=len, reverse=True):
        if name.startswith(enc):
            encoder = enc
            rest = name[len(enc):].lstrip("_")
            break
    if encoder is None:
        return {"encoder": "unknown", "loss": "?", "teacher": "?", "weight": None}
    teacher = "no_teacher" not in rest
    rest = rest.replace("no_teacher", "").strip("_")
    weight = None
    m = re.search(r"w(\d+\.?\d*)", rest)
    if m:
        weight = float(m.group(1))
        rest = re.sub(r"w\d+\.?\d*", "", rest).strip("_")
    loss = rest.strip("_") if rest.strip("_") else "?"
    return {"encoder": encoder, "loss": loss, "teacher": teacher, "weight": weight}


@torch.no_grad()
def _metrics_from(p, g):
    """Hitung metrik dari pasangan pred(p) dan gt(g) yang sudah difilter. p,g 1D."""
    diff = p - g
    ratio = torch.max(p / g, g / p)
    return {
        "RMSE": torch.sqrt((diff ** 2).mean()).item(),
        "RMSE_log": torch.sqrt(((torch.log(p) - torch.log(g)) ** 2).mean()).item(),
        "AbsRel": (diff.abs() / g).mean().item(),
        "SqRel": ((diff ** 2) / g).mean().item(),
        "delta1": (ratio < 1.25).float().mean().item(),
        "delta2": (ratio < 1.25 ** 2).float().mean().item(),
        "delta3": (ratio < 1.25 ** 3).float().mean().item(),
    }


@torch.no_grad()
def compute_depth_metrics(pred, gt, mask, min_depth=0.5, max_depth=20.0):
    """Metrik GLOBAL (semua piksel valid dalam rentang)."""
    p = pred[mask]; g = gt[mask]
    valid = (g >= min_depth) & (g <= max_depth)
    p = p[valid].clamp(min=min_depth, max=max_depth); g = g[valid]
    if p.numel() == 0:
        return None
    return _metrics_from(p, g)


@torch.no_grad()
def compute_depth_metrics_zoned(pred, gt, mask, zones=ZONES, gmin=0.5, gmax=20.0):
    """Metrik per ZONA jarak (berdasarkan nilai GT tiap piksel)."""
    p_all = pred[mask]; g_all = gt[mask]
    results = {}
    for (zmin, zmax) in zones:
        sel = (g_all >= max(zmin, gmin)) & (g_all < min(zmax, gmax + 1e-6))
        p = p_all[sel].clamp(min=gmin, max=gmax); g = g_all[sel]
        label = f"{int(zmin)}-{int(zmax)}m"
        if p.numel() == 0:
            results[label] = None
        else:
            r = _metrics_from(p, g)
            r["n_pixel"] = int(p.numel())
            results[label] = r
    return results


@torch.no_grad()
def evaluate(cfg, verbose=True):
    device = cfg["device"]
    ds = CrossingDataset(cfg["csv_path"], "test", cfg["coco_json"],
                         use_teacher=False, max_objects=cfg["max_objects"])
    if verbose:
        print(f"test: {len(ds)} sampel")
    dl = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=False,
                    num_workers=cfg["num_workers"], collate_fn=collate_fn)

    model = MultiTaskNet(encoder_name=cfg["encoder_name"], pretrained=False,
                         max_objects=cfg["max_objects"]).to(device)
    state = torch.load(cfg["ckpt_path"], map_location=device, weights_only=False)
    model.load_state_dict(state, strict=False)
    model.eval()

    # metrik efisiensi (opsional, murni arsitektur)
    eff = None
    if cfg.get("measure_eff", True):
        eff = measure_efficiency(model, device=device)

    depth_sum = {}; n_depth = 0
    zone_sum = {f"{int(a)}-{int(b)}m": {} for (a, b) in ZONES}
    zone_n = {f"{int(a)}-{int(b)}m": 0 for (a, b) in ZONES}
    iou_sum = 0.0; iou_n = 0
    light_correct = 0; light_total = 0
    kpt_dist_sum = 0.0; kpt_n = 0

    for batch in dl:
        for k in batch:
            if torch.is_tensor(batch[k]):
                batch[k] = batch[k].to(device)
        out = model(batch["image"])
        mask = batch["depth_mask"] > 0.5

        # global
        m = compute_depth_metrics(out["depth"], batch["depth"], mask)
        if m is not None:
            for k, v in m.items():
                depth_sum[k] = depth_sum.get(k, 0.0) + v
            n_depth += 1

        # per zona
        zoned = compute_depth_metrics_zoned(out["depth"], batch["depth"], mask)
        for zlabel, zm in zoned.items():
            if zm is not None:
                for k, v in zm.items():
                    zone_sum[zlabel][k] = zone_sum[zlabel].get(k, 0.0) + v
                zone_n[zlabel] += 1

        # bbox iou
        p = cxcywh_to_xyxy(out["bbox"]); t = cxcywh_to_xyxy(batch["bbox"])
        iou = box_iou(p, t); sm = batch["obj_mask"]
        iou_sum += (iou * sm).sum().item(); iou_n += sm.sum().item()

        # light
        pred_light = out["light"].argmax(dim=1)
        light_correct += (pred_light == batch["light"]).sum().item()
        light_total += batch["light"].numel()

        # keypoint
        vis = batch["keypoints"][..., 2] > 0.5
        slot = batch["obj_mask"] > 0.5
        valid_kpt = vis & slot.unsqueeze(-1)
        dist = torch.sqrt(((out["keypoints_reg"] - batch["keypoints"][..., :2]) ** 2).sum(-1) + 1e-12)
        kpt_dist_sum += (dist * valid_kpt.float()).sum().item()
        kpt_n += valid_kpt.sum().item()

    if verbose:
        print("\n" + "=" * 50)
        print("DEPTH GLOBAL (meter, piksel valid GT)")
        print("=" * 50)
        for k in ["RMSE", "RMSE_log", "AbsRel", "SqRel", "delta1", "delta2", "delta3"]:
            print(f"  {k:10s}: {depth_sum[k]/n_depth:.4f}")

        print("\nDEPTH PER ZONA")
        for zlabel in zone_sum:
            if zone_n[zlabel] > 0:
                z = {k: zone_sum[zlabel][k] / zone_n[zlabel] for k in zone_sum[zlabel]}
                print(f"  {zlabel:8s}: RMSE={z['RMSE']:.3f}  delta1={z['delta1']:.3f}  "
                      f"AbsRel={z['AbsRel']:.3f}  (~{z['n_pixel']:.0f} px/batch)")
            else:
                print(f"  {zlabel:8s}: (tidak ada piksel GT di zona ini)")

        print("\nDETEKSI");  print(f"  mean IoU  : {iou_sum/max(iou_n,1):.4f}")
        print("\nLIGHT");    print(f"  accuracy  : {light_correct/max(light_total,1):.4f}")
        print("\nKEYPOINT"); print(f"  mean dist : {kpt_dist_sum/max(kpt_n,1):.4f}")
        if eff is not None:
            print("\nEFISIENSI")
            print(f"  params    : {eff['params_M']:.2f} M")
            print(f"  size      : {eff['size_MB']:.2f} MB")
            print(f"  GFLOPs    : {eff['GFLOPs']:.2f}")
            print(f"  latency   : {eff['latency_ms']:.1f} ms  ({eff['FPS']:.1f} FPS) [{device}]")
    else:
        out = {}
        for k in ["RMSE", "RMSE_log", "AbsRel", "SqRel", "delta1", "delta2", "delta3"]:
            out[k] = depth_sum[k] / n_depth
        # kolom per zona
        for zlabel in zone_sum:
            if zone_n[zlabel] > 0:
                for k in ["RMSE", "delta1", "delta2", "delta3", "AbsRel"]:
                    out[f"{zlabel}_{k}"] = zone_sum[zlabel][k] / zone_n[zlabel]
            else:
                for k in ["RMSE", "delta1", "delta2", "delta3", "AbsRel"]:
                    out[f"{zlabel}_{k}"] = float("nan")
        out["mean_iou"] = iou_sum / max(iou_n, 1)
        out["light_acc"] = light_correct / max(light_total, 1)
        out["kpt_dist"] = kpt_dist_sum / max(kpt_n, 1)
        if eff is not None:
            out.update(eff)
        return out


if __name__ == "__main__":
    MODE = "multi"     # "single" = 1 model print; "multi" = semua model -> CSV

    if MODE == "single":
        decoder_name = "efficientnet_b0"
        loss = "softdelta"
        use_teacher = "_no_teacher"      # "" jika pakai teacher, atau mis. "_w0.10"
        model_name = f"train_{decoder_name}_{loss}{use_teacher}_last"

        print("model_name:", model_name)
        cfg = {
            "csv_path":   "dataset.csv",
            "coco_json":  "new_ds/train/_annotations_coco.json",
            "ckpt_path":  f"checkpoints/{model_name}.pt",
            "encoder_name": decoder_name,
            "max_objects": 4, "batch_size": 16, "num_workers": 4,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        }
        evaluate(cfg, verbose=True)

    else:
        import pandas as pd
        ckpt_dir = "checkpoints"
        out_csv = "eval_results.csv"
        ckpt_files = sorted(glob.glob(os.path.join(ckpt_dir, "*.pt")))
        print(f"ditemukan {len(ckpt_files)} checkpoint")

        rows = []
        for path in ckpt_files:
            fname = os.path.basename(path)
            info = parse_ckpt_name(fname)
            if info["encoder"] == "unknown":
                print(f"  [skip] {fname}"); continue
            cfg = {
                "csv_path":   "dataset.csv",
                "coco_json":  "new_ds/train/_annotations_coco.json",
                "ckpt_path":  path, "encoder_name": info["encoder"],
                "max_objects": 4, "batch_size": 16, "num_workers": 4,
                "device": "cuda" if torch.cuda.is_available() else "cpu",
            }
            try:
                metrics = evaluate(cfg, verbose=False)
            except Exception as e:
                print(f"  [error] {fname}: {str(e)[:60]}"); continue
            rows.append({"encoder": info["encoder"], "loss": info["loss"],
                         "teacher": info["teacher"], "weight": info["weight"], **metrics})
            print(f"  done: {fname}  d1={metrics['delta1']:.3f} RMSE={metrics['RMSE']:.3f}")

        df = pd.DataFrame(rows)
        df.to_csv(out_csv, index=False)
        print(f"\nsimpan {len(df)} baris ke {out_csv}")
        print(df.to_string())