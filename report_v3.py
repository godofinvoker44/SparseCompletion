"""
report_v3.py

Laporan detail untuk 4 model grid v3 di split test:
    1. Metrik lampu per kelas (precision, recall/akurasi per kelas, F1, support),
       macro-F1, confusion matrix                 -> eval_light_v3.csv, fig_v3_light.png
    2. Visualisasi depth: gambar | GT sensor | pseudo label | prediksi 4 model
                                                   -> fig_v3_depth.png

Jalankan: python report_v3.py
"""

import csv
import glob
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import CrossingDataset
from model import MultiTaskNet
from evaluate_v3 import evaluate_ckpt, parse_v3_name, LIGHT_NAMES

CSV_PATH, COCO = "dataset_v3.csv", "ds_v3/_annotations.coco.json"
ORDER = ["gt_raw", "gt_canonical", "pseudo_raw", "pseudo_canonical"]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]     # palet kategorikal tervalidasi
DMAX = 20.0


def short(name):
    p = parse_v3_name(name + "_last.pt")
    return f"{p['depth_source']}_{p['kpt_order']}"


def light_report(results):
    rows = []
    print("\nLAMPU LALU LINTAS (test)")
    print(f"{'model':18s} {'kelas':9s} {'support':>7s} {'recall':>7s} {'precision':>9s} {'F1':>6s}")
    for r in results:
        for nm in LIGHT_NAMES:
            rows.append({"model": short(r["name"]), "class": nm,
                         "support": r[f"light_{nm}_support"],
                         "recall_acc": r[f"light_{nm}_recall"],
                         "precision": r[f"light_{nm}_precision"],
                         "f1": r[f"light_{nm}_f1"]})
            print(f"{short(r['name']):18s} {nm:9s} {r[f'light_{nm}_support']:7d} "
                  f"{100*r[f'light_{nm}_recall']:6.1f}% {100*r[f'light_{nm}_precision']:8.1f}% "
                  f"{r[f'light_{nm}_f1']:6.3f}")
        print(f"{short(r['name']):18s} {'(semua)':9s} {'':7s} acc {100*r['light_acc']:5.1f}%  "
              f"bal-acc {100*r['light_bal_acc']:5.1f}%  macro-F1 {r['light_macro_f1']:.3f}\n")
    with open("eval_light_v3.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # ---- figure: F1 per kelas (bar) + confusion matrix per model ----
    fig = plt.figure(figsize=(16, 8.5))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.05], hspace=0.45, wspace=0.35)
    ax = fig.add_subplot(gs[0, :])
    x = np.arange(len(LIGHT_NAMES) + 1); bw = 0.2
    for k, r in enumerate(results):
        vals = [r[f"light_{nm}_f1"] for nm in LIGHT_NAMES] + [r["light_macro_f1"]]
        bars = ax.bar(x + (k - 1.5) * bw, vals, bw - 0.02, color=COLORS[k], label=short(r["name"]))
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=8, color="#333")
    sup = results[0]
    ax.set_xticks(x, [f"{nm}\n(n={sup[f'light_{nm}_support']})" for nm in LIGHT_NAMES] + ["Macro-F1"])
    ax.set_ylim(0, 1.1); ax.set_ylabel("F1 (test)")
    ax.set_title("F1 lampu lalu lintas per kelas (split test)", loc="left", fontsize=12, pad=28)
    ax.legend(ncol=4, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0))
    ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", color="#e5e5e5"); ax.set_axisbelow(True)

    for k, r in enumerate(results):
        a = fig.add_subplot(gs[1, k])
        cm = np.array(r["light_cm"], dtype=float)
        cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
        a.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                a.text(j, i, f"{100*cmn[i,j]:.0f}%\n({int(cm[i,j])})", ha="center", va="center",
                       fontsize=9, color="white" if cmn[i, j] > 0.55 else "#222")
        a.set_xticks(range(3), LIGHT_NAMES, fontsize=8); a.set_yticks(range(3), LIGHT_NAMES, fontsize=8)
        a.set_xlabel("prediksi"); a.set_ylabel("GT" if k == 0 else "")
        a.set_title(short(r["name"]), fontsize=10, color=COLORS[k], fontweight="bold")
    fig.suptitle("Confusion matrix dinormalisasi per baris (= recall per kelas)", y=0.47, fontsize=10, x=0.5)
    fig.savefig("fig_v3_light.png", dpi=110, bbox_inches="tight"); plt.close(fig)
    print("-> eval_light_v3.csv, fig_v3_light.png")


@torch.no_grad()
def depth_figure(ckpts, n=6, device="cuda"):
    ds = CrossingDataset(CSV_PATH, "test", COCO, use_teacher=True, kpt_order="raw")
    # sampel tersebar: ambil frame dari sesi berbeda, campur tinggi kamera
    rng = np.random.default_rng(0)
    by_sess = {}
    for i, r in enumerate(ds.rows):
        by_sess.setdefault(r["session"], []).append(i)
    sess = sorted(by_sess); rng.shuffle(sess)
    idxs = [by_sess[s][len(by_sess[s]) // 2] for s in sess[:n]]

    models = []
    for p in ckpts:
        info = parse_v3_name(p)
        m = MultiTaskNet(encoder_name=info["encoder"], pretrained=False).to(device).eval()
        m.load_state_dict(torch.load(p, map_location=device, weights_only=False))
        models.append((short(info["name"]), m))

    cols = ["gambar", "GT sensor", "pseudo label\n(DepthAnything)"] + [nm for nm, _ in models]
    fig, axes = plt.subplots(n, len(cols), figsize=(2.0 * len(cols), 3.6 * n))
    for r, i in enumerate(idxs):
        s = ds[i]
        img = s["image"].permute(1, 2, 0).numpy()
        gt = s["depth"][0].numpy(); valid = s["depth_mask"][0].numpy() > 0.5
        panels = [img, np.where(valid, gt, np.nan), s["teacher"][0].numpy()]
        x = s["image"].unsqueeze(0).to(device)
        for _, m in models:
            panels.append(m(x)["depth"][0, 0].cpu().numpy())
        for c, pnl in enumerate(panels):
            a = axes[r, c]
            if c == 0:
                a.imshow(pnl)
                a.set_ylabel(f"{s['frame_id'][9:15]}\n{int(s['height_cm'])} cm", fontsize=8)
            else:
                cm = plt.get_cmap("turbo").copy(); cm.set_bad("#bbbbbb")
                a.imshow(pnl, cmap=cm, vmin=0, vmax=DMAX)
            if c >= 3:
                pm = panels[c][valid]; g = gt[valid]
                sel = (g > 0.5)
                if sel.any():
                    ratio = np.maximum(pm[sel] / g[sel], g[sel] / np.clip(pm[sel], 1e-3, None))
                    a.text(4, 505, f"d1={np.mean(ratio < 1.25):.2f}", fontsize=8, color="white",
                           bbox=dict(facecolor="black", alpha=0.6, pad=1, lw=0))
            a.set_xticks([]); a.set_yticks([])
            if r == 0:
                a.set_title(cols[c], fontsize=9)
    sm = plt.cm.ScalarMappable(cmap="turbo", norm=plt.Normalize(0, DMAX))
    cb = fig.colorbar(sm, ax=axes, fraction=0.015, pad=0.01); cb.set_label("depth (m)  | abu-abu = GT invalid")
    fig.savefig("fig_v3_depth.png", dpi=90, bbox_inches="tight"); plt.close(fig)
    print("-> fig_v3_depth.png")


@torch.no_grad()
def hole_report(ckpts, device="cuda"):
    """Metrik standar hanya dihitung di piksel GT valid. Di sini: bagaimana prediksi
    di piksel GT INVALID (lubang sensor: langit, objek jauh, permukaan mengkilap)?
    Referensi = pseudo label DepthAnything (hanya sebagai acuan kasar)."""
    ds = CrossingDataset(CSV_PATH, "test", COCO, use_teacher=True, kpt_order="raw")
    out = []
    for p in ckpts:
        info = parse_v3_name(p)
        m = MultiTaskNet(encoder_name=info["encoder"], pretrained=False).to(device).eval()
        m.load_state_dict(torch.load(p, map_location=device, weights_only=False))
        n_hole = n_near = 0; d1_sum = absrel_sum = 0.0
        for i in range(0, len(ds), 2):            # setiap frame ke-2 (cukup untuk statistik)
            s = ds[i]
            hole = (s["depth_mask"][0] < 0.5).numpy() & (s["has_teacher"].item() > 0)
            if hole.sum() < 100:
                continue
            pr = m(s["image"].unsqueeze(0).to(device))["depth"][0, 0].cpu().numpy()[hole]
            te = s["teacher"][0].numpy()[hole]
            ratio = np.maximum(pr / te, te / np.clip(pr, 1e-3, None))
            d1_sum += (ratio < 1.25).sum(); absrel_sum += (np.abs(pr - te) / te).sum()
            n_near += (pr < 2.0).sum(); n_hole += hole.sum()
        out.append({"model": short(info["name"]), "hole_px_frac": n_hole,
                    "hole_d1_vs_pseudo": d1_sum / n_hole, "hole_absrel_vs_pseudo": absrel_sum / n_hole,
                    "hole_pred_lt_2m": n_near / n_hole})
    print("\nDEPTH DI AREA GT INVALID (lubang sensor), acuan = pseudo label")
    print(f"{'model':18s} {'d1 vs pseudo':>12s} {'AbsRel vs pseudo':>17s} {'prediksi < 2 m':>15s}")
    for r in out:
        print(f"{r['model']:18s} {r['hole_d1_vs_pseudo']:12.3f} {r['hole_absrel_vs_pseudo']:17.3f} "
              f"{100*r['hole_pred_lt_2m']:14.1f}%")
    return out


if __name__ == "__main__":
    ckpts = sorted(glob.glob("checkpoints/v3_*_last.pt"), key=lambda p: ORDER.index(short(parse_v3_name(p)["name"])))
    results = [evaluate_ckpt(p, CSV_PATH, COCO, "test") for p in ckpts]
    light_report(results)
    depth_figure(ckpts)
    hole_report(ckpts)
