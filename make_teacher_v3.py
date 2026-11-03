"""
make_teacher_v3.py

Pseudo label depth dari Depth-Anything-V2-Large (relative) untuk dataset v3.
Metode SAMA dengan Teacher2 di eda_teacher.ipynb:
    pred relatif -> resize ke 512x256 -> kalibrasi polinomial derajat-2 per frame
    ke GT sensor (piksel valid) -> clip [0.001, 20] m -> simpan float32 .npy

Frame dengan piksel GT valid < 50 dilewati (tidak bisa dikalibrasi) -> has_teacher=0.
Resume: file yang sudah ada dilewati.

Jalankan: python make_teacher_v3.py   (env shimo_ml)
"""

import os
import csv
import json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from numpy.polynomial import Polynomial
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

CSV_PATH = "dataset_v3.csv"
MODEL_ID = "depth-anything/Depth-Anything-V2-Large-hf"
IMG_SIZE = (512, 256)   # H, W
DMIN, DMAX = 0.001, 20.0
DEPTH_SCALE = 1.0 / 1000.0


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    proc = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID).to(device).eval()

    rows = list(csv.DictReader(open(CSV_PATH)))
    calib_log, skipped = {}, []
    log_path = os.path.join(os.path.dirname(rows[0]["teacher_path"]), "_calib_log.json")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    if os.path.exists(log_path):
        calib_log = json.load(open(log_path))

    for k, r in enumerate(rows):
        out_path = r["teacher_path"]
        if os.path.exists(out_path):
            continue

        gt = np.load(r["depth_path"]).astype(np.float32)
        if gt.shape != IMG_SIZE:
            gt = np.asarray(Image.fromarray(gt).resize((IMG_SIZE[1], IMG_SIZE[0]), Image.NEAREST))
        gt = gt * DEPTH_SCALE
        valid = (gt > DMIN) & (gt < DMAX)
        if valid.sum() < 50:
            skipped.append(r["frame_id"]); continue

        pil = Image.open(r["img_path"]).convert("RGB")
        inp = proc(images=pil, return_tensors="pt").to(device)
        with torch.no_grad():
            pred = model(**inp).predicted_depth
        pred = F.interpolate(pred.unsqueeze(1), size=IMG_SIZE, mode="bilinear",
                             align_corners=False)[0, 0].cpu().numpy()

        poly = Polynomial.fit(pred[valid], gt[valid], deg=2)
        teacher = np.clip(poly(pred), DMIN, DMAX).astype(np.float32)
        np.save(out_path, teacher)

        mae = float(np.abs(teacher[valid] - gt[valid]).mean())
        calib_log[r["frame_id"]] = {"coef": [float(c) for c in poly.convert().coef], "mae_m": mae}

        if k % 250 == 0:
            print(f"  {k}/{len(rows)}  {r['frame_id']}  MAE vs GT = {mae:.3f} m", flush=True)
            json.dump(calib_log, open(log_path, "w"))

    json.dump(calib_log, open(log_path, "w"))
    maes = [v["mae_m"] for v in calib_log.values()]
    print(f"selesai: {len(calib_log)} teacher | skip {len(skipped)} | "
          f"MAE kalibrasi rata-rata {np.mean(maes):.3f} m (median {np.median(maes):.3f})")
    if skipped:
        print("  skip:", skipped[:20], "..." if len(skipped) > 20 else "")


if __name__ == "__main__":
    main()
