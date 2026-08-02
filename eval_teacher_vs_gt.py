"""
eval_teacher_vs_gt.py

Mengukur seberapa akurat TEACHER (hasil kalibrasi DAv2) terhadap GT sensor,
di split test. Ini menentukan "plafon": student tidak akan jauh melampaui
teacher lewat distillation, jadi kalau teacher sendiri kurang akurat,
memperbaiki distillation tak akan banyak menolong.

Metrik sama seperti evaluasi student: global + per-zona.
Membandingkan teacher (kolom teacher_path) vs GT (depth_path) di piksel valid GT.
"""

import csv
import numpy as np
import torch

from evaluate import compute_depth_metrics, compute_depth_metrics_zoned, ZONES
from dataset import IMG_SIZE, DEPTH_SCALE, DEPTH_MIN, DEPTH_MAX
from PIL import Image


def load_depth(path, scale):
    d = np.load(path).astype(np.float32)
    Ht, Wt = IMG_SIZE
    if d.shape != (Ht, Wt):
        d = np.asarray(Image.fromarray(d).resize((Wt, Ht), Image.NEAREST), dtype=np.float32)
    return d * scale


def main(csv_path="dataset_v2.csv", split="test"):
    rows = [r for r in csv.DictReader(open(csv_path)) if r["split"] == split]
    rows = [r for r in rows if r.get("teacher_path")]   # yang punya teacher
    print(f"{len(rows)} sampel test dengan teacher")

    depth_sum = {}; n = 0
    zone_sum = {f"{int(a)}-{int(b)}m": {} for (a, b) in ZONES}
    zone_n = {f"{int(a)}-{int(b)}m": 0 for (a, b) in ZONES}

    for r in rows:
        gt = load_depth(r["depth_path"], DEPTH_SCALE)          # GT sensor (meter)
        teacher = np.load(r["teacher_path"]).astype(np.float32)  # teacher sudah meter
        Ht, Wt = IMG_SIZE
        if teacher.shape != (Ht, Wt):
            teacher = np.asarray(Image.fromarray(teacher).resize((Wt, Ht), Image.NEAREST), dtype=np.float32)

        gt_t = torch.from_numpy(gt).unsqueeze(0).unsqueeze(0)
        te_t = torch.from_numpy(teacher).clamp(min=1e-3).unsqueeze(0).unsqueeze(0)
        mask = (gt_t >= DEPTH_MIN) & (gt_t < DEPTH_MAX)

        # teacher sebagai "prediksi", GT sebagai target
        m = compute_depth_metrics(te_t, gt_t, mask)
        if m is not None:
            for k, v in m.items():
                depth_sum[k] = depth_sum.get(k, 0.0) + v
            n += 1
        z = compute_depth_metrics_zoned(te_t, gt_t, mask)
        for zl, zm in z.items():
            if zm is not None:
                for k, v in zm.items():
                    zone_sum[zl][k] = zone_sum[zl].get(k, 0.0) + v
                zone_n[zl] += 1

    print("\n" + "=" * 50)
    print("TEACHER vs GT (plafon distillation)")
    print("=" * 50)
    for k in ["RMSE", "RMSE_log", "AbsRel", "SqRel", "delta1", "delta2", "delta3"]:
        print(f"  {k:10s}: {depth_sum[k]/n:.4f}")
    print("\nPER ZONA:")
    for zl in zone_sum:
        if zone_n[zl] > 0:
            z = {k: zone_sum[zl][k]/zone_n[zl] for k in zone_sum[zl]}
            print(f"  {zl:8s}: RMSE={z['RMSE']:.3f}  delta1={z['delta1']:.3f}  AbsRel={z['AbsRel']:.3f}")


if __name__ == "__main__":
    main()