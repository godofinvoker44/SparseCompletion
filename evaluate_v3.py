"""
evaluate_v3.py

Evaluasi model v3 di split test (atau val). Depth SELALU dibandingkan ke GT sensor,
apa pun sumber depth saat training (gt / pseudo).

Metrik keypoint (koordinat dalam piksel gambar 256x512):
    kpt_px_idx     : jarak titik ke-j prediksi vs titik ke-j GT, dengan urutan GT
                     sama seperti saat model dilatih (raw / canonical).
    kpt_px_canon   : prediksi DAN GT sama-sama dikanonisasi ke [TL,TR,BL,BR] dulu.
                     -> adil untuk membandingkan model raw vs canonical.
    kpt_px_TL/TR/BL/BR : kpt_px_canon per sudut.
    kpt_pck10      : fraksi titik (canonical) dengan error < 10 px.
    kpt_spread     : rata-rata jarak antar 4 titik prediksi / jarak antar 4 titik GT.
                     ~1 = normal, << 1 = titik-titik mengkerut / collapse.

Jalankan: python evaluate_v3.py  -> semua checkpoint v3_* -> eval_results_v3.csv
"""

import os
import glob
import csv
import torch
from torch.utils.data import DataLoader

from dataset import CrossingDataset, collate_fn, IMG_SIZE
from model import MultiTaskNet
from box_ops import cxcywh_to_xyxy, box_iou
from evaluate import compute_depth_metrics, compute_depth_metrics_zoned, ZONES
from kpt_order import canonicalize_tensor, CANON_NAMES

DEPTH_KEYS = ["RMSE", "RMSE_log", "AbsRel", "SqRel", "delta1", "delta2", "delta3"]
LIGHT_NAMES = ["No-Light", "Green", "Red"]


def _pairwise_mean(xy):
    """xy [N,4,2] -> rata-rata jarak antar 6 pasang titik, [N]."""
    d = torch.cdist(xy, xy)                     # [N,4,4]
    iu = torch.triu_indices(4, 4, offset=1)
    return d[:, iu[0], iu[1]].mean(-1)


@torch.no_grad()
def evaluate_model(model, ds, device, batch_size=8, num_workers=4):
    model.eval()
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    num_workers=num_workers, collate_fn=collate_fn)
    H, W = IMG_SIZE
    scale = torch.tensor([W, H], dtype=torch.float32, device=device)

    dsum = {k: 0.0 for k in DEPTH_KEYS}; dn = 0
    zlabels = [f"{int(a)}-{int(b)}m" for a, b in ZONES]
    zsum = {z: {} for z in zlabels}; zn = {z: 0 for z in zlabels}
    iou_sum = iou_n = 0.0
    light_cm = torch.zeros(3, 3, dtype=torch.long)
    kidx_sum = kcan_sum = kn = pck = 0.0
    corner_sum = torch.zeros(4); spread_sum = spread_n = 0.0

    for batch in dl:
        for k in batch:
            if torch.is_tensor(batch[k]):
                batch[k] = batch[k].to(device)
        out = model(batch["image"])
        B = batch["image"].shape[0]

        # ---- depth vs GT sensor, per gambar ----
        for b in range(B):
            mask = batch["depth_mask"][b:b+1] > 0.5
            m = compute_depth_metrics(out["depth"][b:b+1], batch["depth"][b:b+1], mask)
            if m is not None:
                for k in DEPTH_KEYS:
                    dsum[k] += m[k]
                dn += 1
            zm = compute_depth_metrics_zoned(out["depth"][b:b+1], batch["depth"][b:b+1], mask)
            for z, v in zm.items():
                if v is not None:
                    for k, x in v.items():
                        zsum[z][k] = zsum[z].get(k, 0.0) + x
                    zn[z] += 1

        # ---- bbox ----
        slot = batch["obj_mask"] > 0.5
        iou = box_iou(cxcywh_to_xyxy(out["bbox"]), cxcywh_to_xyxy(batch["bbox"]))
        iou_sum += iou[slot].sum().item(); iou_n += slot.sum().item()

        # ---- light (confusion matrix) ----
        pl = out["light"].argmax(1).cpu(); gl = batch["light"].cpu()
        for g, p in zip(gl.tolist(), pl.tolist()):
            light_cm[g, p] += 1

        # ---- keypoint: hanya slot terisi dengan 4 titik visible ----
        vis_all = (batch["keypoints"][..., 2] > 0.5).all(-1)          # [B,M]
        sel = slot & vis_all
        if sel.any():
            pr = out["keypoints_reg"][sel] * scale                     # [N,4,2] px
            gt = batch["keypoints"][sel][..., :2] * scale
            d_idx = (pr - gt).norm(dim=-1)                             # [N,4]
            pr_c, gt_c = canonicalize_tensor(pr), canonicalize_tensor(gt)
            d_can = (pr_c - gt_c).norm(dim=-1)
            kidx_sum += d_idx.sum().item(); kcan_sum += d_can.sum().item()
            kn += d_can.numel(); pck += (d_can < 10).sum().item()
            corner_sum += d_can.sum(0).cpu()
            ratio = _pairwise_mean(pr) / _pairwise_mean(gt).clamp(min=1e-6)
            spread_sum += ratio.sum().item(); spread_n += ratio.numel()

    res = {k: dsum[k] / max(dn, 1) for k in DEPTH_KEYS}
    for z in zlabels:
        for k in ["RMSE", "delta1", "AbsRel"]:
            res[f"{z}_{k}"] = zsum[z][k] / zn[z] if zn[z] else float("nan")
    res["mean_iou"] = iou_sum / max(iou_n, 1)
    tot = light_cm.sum().item()
    res["light_acc"] = light_cm.diag().sum().item() / max(tot, 1)
    recalls = [light_cm[i, i].item() / light_cm[i].sum().item()
               for i in range(3) if light_cm[i].sum() > 0]
    res["light_bal_acc"] = sum(recalls) / max(len(recalls), 1)
    # per kelas: precision, recall (= akurasi per kelas), F1, support
    f1s = []
    for i, nm in enumerate(LIGHT_NAMES):
        tp = light_cm[i, i].item(); sup = light_cm[i].sum().item(); npred = light_cm[:, i].sum().item()
        prec = tp / npred if npred else 0.0
        rec = tp / sup if sup else float("nan")
        f1 = 2 * prec * rec / (prec + rec) if sup and (prec + rec) > 0 else 0.0
        res[f"light_{nm}_precision"] = prec; res[f"light_{nm}_recall"] = rec
        res[f"light_{nm}_f1"] = f1; res[f"light_{nm}_support"] = sup
        if sup:
            f1s.append(f1)
    res["light_macro_f1"] = sum(f1s) / max(len(f1s), 1)
    res["light_cm"] = light_cm.tolist()          # baris = GT, kolom = prediksi
    res["kpt_px_idx"] = kidx_sum / max(kn, 1)
    res["kpt_px_canon"] = kcan_sum / max(kn, 1)
    n_obj = kn / 4 if kn else 1
    for i, nm in enumerate(CANON_NAMES):
        res[f"kpt_px_{nm}"] = corner_sum[i].item() / max(n_obj, 1)
    res["kpt_pck10"] = pck / max(kn, 1)
    res["kpt_spread"] = spread_sum / max(spread_n, 1)
    return res


def parse_v3_name(fname):
    """v3_{gt|pseudo}_{raw|canonical}_{encoder}_{loss}_last.pt"""
    stem = os.path.basename(fname).replace("_last.pt", "").replace(".pt", "")
    parts = stem.split("_")
    depth_source, kpt_order = parts[1], parts[2]
    loss = parts[-1]
    encoder = "_".join(parts[3:-1])
    return {"name": stem, "depth_source": depth_source, "kpt_order": kpt_order,
            "encoder": encoder, "loss": loss}


def evaluate_ckpt(path, csv_path="dataset_v3.csv", coco_json="ds_v3/_annotations.coco.json",
                  split="test", device=None, max_objects=4):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    info = parse_v3_name(path)
    ds = CrossingDataset(csv_path, split, coco_json, use_teacher=False,
                         max_objects=max_objects, kpt_order=info["kpt_order"])
    model = MultiTaskNet(encoder_name=info["encoder"], pretrained=False,
                         max_objects=max_objects).to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=False))
    res = evaluate_model(model, ds, device)
    return {**info, "split": split, "n_samples": len(ds), **res}


def print_table(rows):
    cols = ["name", "delta1", "AbsRel", "RMSE", "mean_iou", "light_acc", "light_bal_acc",
            "kpt_px_idx", "kpt_px_canon", "kpt_pck10", "kpt_spread"]
    print("  ".join(f"{c:>13s}" if c != "name" else f"{c:45s}" for c in cols))
    for r in rows:
        print("  ".join(f"{r[c]:45s}" if c == "name" else f"{r[c]:13.4f}" for c in cols))


if __name__ == "__main__":
    ckpts = sorted(glob.glob("checkpoints/v3_*_last.pt"))
    print(f"ditemukan {len(ckpts)} checkpoint v3")
    rows = []
    for p in ckpts:
        r = evaluate_ckpt(p)
        rows.append(r)
        print(f"  done {r['name']}")
    if rows:
        with open("eval_results_v3.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print_table(rows)
        print("-> eval_results_v3.csv")
