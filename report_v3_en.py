"""
report_v3_en.py

Detailed English report for the v3 2x2 grid (test split):
    results_v3_en.md          all metric tables (depth global / per zone / sensor holes,
                              teacher reference, detection, keypoints, traffic light)
    fig_v3_en_depth.png       image | GT sensor | pseudo label | 4 model predictions
    fig_v3_en_error.png       AbsRel error map vs GT sensor (valid pixels only)
    fig_v3_en_light.png       per-class F1 + confusion matrices
    fig_v3_en_detection.png   zebra-crossing images: GT vs predicted bbox + keypoints

Run: python report_v3_en.py   (env shimo_ml)
"""

import glob
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from dataset import CrossingDataset
from model import MultiTaskNet
from evaluate import compute_depth_metrics, compute_depth_metrics_zoned, ZONES
from evaluate_v3 import evaluate_ckpt, parse_v3_name, LIGHT_NAMES, DEPTH_KEYS
from kpt_order import canonicalize_tensor, CANON_NAMES

CSV_PATH, COCO = "dataset_v3.csv", "ds_v3/_annotations.coco.json"
ORDER = ["gt_raw", "gt_canonical", "pseudo_raw", "pseudo_canonical"]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]      # validated categorical palette
KPT_COLORS = ["#e74c3c", "#f1c40f", "#2ecc71", "#3498db"]  # TL, TR, BL, BR
DMAX = 20.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ZLABELS = [f"{int(a)}-{int(b)}m" for a, b in ZONES]


def short(name):
    p = parse_v3_name(name + "_last.pt")
    return f"{p['depth_source']}_{p['kpt_order']}"


def load_models(ckpts):
    models = []
    for p in ckpts:
        info = parse_v3_name(p)
        m = MultiTaskNet(encoder_name=info["encoder"], pretrained=False).to(DEVICE).eval()
        m.load_state_dict(torch.load(p, map_location=DEVICE, weights_only=False))
        models.append((short(info["name"]), m))
    return models


def pick_frames(ds, n, need_object=False, seed=0):
    """One frame per session (middle of the session), sessions shuffled."""
    by_sess = {}
    for i, r in enumerate(ds.rows):
        if need_object and int(r["n_object"]) == 0:
            continue
        by_sess.setdefault(r["session"], []).append(i)
    sess = sorted(by_sess)
    np.random.default_rng(seed).shuffle(sess)
    return [by_sess[s][len(by_sess[s]) // 2] for s in sess[:n]]


# ------------------------------------------------------------------ metrics
@torch.no_grad()
def teacher_reference(ds):
    """Pseudo label itself vs GT sensor. NOTE: the teacher is calibrated per frame
    with that frame's own GT, so this is an optimistic (oracle-scale) reference."""
    s_g = {k: 0.0 for k in DEPTH_KEYS}; n_g = 0
    s_z = {z: {} for z in ZLABELS}; n_z = {z: 0 for z in ZLABELS}
    for i in range(len(ds)):
        s = ds[i]
        if s["has_teacher"].item() == 0:
            continue
        t, g, m = s["teacher"][None], s["depth"][None], s["depth_mask"][None] > 0.5
        r = compute_depth_metrics(t, g, m)
        if r:
            for k in DEPTH_KEYS:
                s_g[k] += r[k]
            n_g += 1
        for z, v in compute_depth_metrics_zoned(t, g, m).items():
            if v:
                for k, x in v.items():
                    s_z[z][k] = s_z[z].get(k, 0.0) + x
                n_z[z] += 1
    out = {k: s_g[k] / n_g for k in DEPTH_KEYS}
    for z in ZLABELS:
        for k in ["RMSE", "delta1", "AbsRel"]:
            out[f"{z}_{k}"] = s_z[z][k] / n_z[z] if n_z[z] else float("nan")
    return out


@torch.no_grad()
def hole_metrics(models, ds):
    """Prediction quality on GT-INVALID pixels (sensor holes: sky, far objects,
    reflective surfaces). Reference = pseudo label (rough). 'pred < 2 m' does not
    depend on the reference: such values are implausible in those regions."""
    out = {}
    for name, m in models:
        n = near = 0; d1 = ar = 0.0
        for i in range(0, len(ds), 2):
            s = ds[i]
            hole = (s["depth_mask"][0] < 0.5).numpy() & (s["has_teacher"].item() > 0)
            if hole.sum() < 100:
                continue
            pr = m(s["image"][None].to(DEVICE))["depth"][0, 0].cpu().numpy()[hole]
            te = s["teacher"][0].numpy()[hole]
            ratio = np.maximum(pr / te, te / np.clip(pr, 1e-3, None))
            d1 += (ratio < 1.25).sum(); ar += (np.abs(pr - te) / te).sum()
            near += (pr < 2.0).sum(); n += hole.sum()
        out[name] = {"d1": d1 / n, "absrel": ar / n, "lt2m": near / n}
    return out


def md_table(header, rows):
    s = "| " + " | ".join(header) + " |\n|" + "|".join("---" for _ in header) + "|\n"
    for r in rows:
        s += "| " + " | ".join(r) + " |\n"
    return s


def write_markdown(results, teacher, holes, n_test):
    f3 = lambda x: f"{x:.3f}"
    pct = lambda x: f"{100*x:.1f}%"
    md = "# v3 experiment grid: evaluation report (test split)\n\n"
    md += (f"Encoder `efficientnet_b0`, depth loss SoftDelta, 50 epochs, single seed (42). "
           f"Test split: {n_test} frames from 51 unseen sessions. Depth is always evaluated "
           f"against the GT sensor depth (valid pixels, 0.5 to 20 m) unless stated otherwise.\n\n")
    md += ("Runs: `gt_*` = depth loss on GT sensor depth (valid pixels only); "
           "`pseudo_*` = depth loss on Depth-Anything-V2 pseudo labels (all pixels, no GT); "
           "`*_raw` = keypoints in annotation order; `*_canonical` = keypoints reordered to TL, TR, BL, BR.\n\n")

    md += "## 1. Depth vs GT sensor (valid pixels)\n\n"
    hdr = ["Model", "δ1 ↑", "δ2 ↑", "δ3 ↑", "AbsRel ↓", "SqRel ↓", "RMSE (m) ↓", "RMSE log ↓"]
    rows = [[short(r["name"])] + [f3(r[k]) for k in ["delta1", "delta2", "delta3", "AbsRel", "SqRel", "RMSE", "RMSE_log"]]
            for r in results]
    rows.append(["*Teacher (pseudo label) ‡*"] + [f3(teacher[k]) for k in ["delta1", "delta2", "delta3", "AbsRel", "SqRel", "RMSE", "RMSE_log"]])
    md += md_table(hdr, rows)
    md += ("\n‡ The pseudo label is calibrated per frame with that frame's own GT, so its numbers are an "
           "optimistic reference. The pseudo-trained students land close to it: they reproduce the "
           "teacher, including its errors.\n\n")

    md += "## 2. Depth per distance zone (GT-valid pixels)\n\n"
    hdr = ["Model"] + [f"{z} {k}" for z in ZLABELS for k in ["δ1", "AbsRel", "RMSE"]]
    src = [(short(r["name"]), r) for r in results] + [("*Teacher ‡*", teacher)]
    rows = [[nm] + [f3(r[f"{z}_{k}"]) for z in ZLABELS for k in ["delta1", "AbsRel", "RMSE"]] for nm, r in src]
    md += md_table(hdr, rows) + "\n"

    md += "## 3. Depth in GT sensor holes (GT-invalid pixels)\n\n"
    md += ("The standard metrics above ignore these pixels. Reference = pseudo label (rough), "
           "so δ1/AbsRel favour the pseudo runs; the last column does not depend on the reference.\n\n")
    hdr = ["Model", "δ1 vs pseudo ↑", "AbsRel vs pseudo ↓", "Predicted < 2 m (implausible) ↓"]
    rows = [[nm, f3(v["d1"]), f3(v["absrel"]), pct(v["lt2m"])] for nm, v in holes.items()]
    md += md_table(hdr, rows) + "\n"

    md += "## 4. Detection and keypoints\n\n"
    md += ("Keypoint error in pixels on the 256×512 image. *Canonical* error: prediction and GT are both "
           "reordered to TL/TR/BL/BR first, so raw and canonical runs are compared fairly. "
           "Spread = mean distance between the 4 predicted points ÷ same for GT (≈1 means no collapse).\n\n")
    hdr = ["Model", "bbox IoU ↑", "kpt err (px) ↓", "TL", "TR", "BL", "BR", "PCK@10px ↑", "Spread"]
    rows = [[short(r["name"]), f3(r["mean_iou"]), f"{r['kpt_px_canon']:.1f}"]
            + [f"{r[f'kpt_px_{c}']:.1f}" for c in CANON_NAMES] + [f3(r["kpt_pck10"]), f"{r['kpt_spread']:.2f}"]
            for r in results]
    md += md_table(hdr, rows) + "\n"

    md += "## 5. Traffic light classification\n\n"
    sup = results[0]
    md += ("Support (test): " + ", ".join(f"{nm} = {sup[f'light_{nm}_support']}" for nm in LIGHT_NAMES)
           + ". Recall = per-class accuracy. With only 39 Green frames, one frame ≈ 2.6 points.\n\n")
    hdr = ["Model"] + [f"{nm} {k}" for nm in LIGHT_NAMES for k in ["P", "R", "F1"]] + ["Accuracy", "Balanced acc", "Macro-F1"]
    rows = [[short(r["name"])] + [f3(r[f"light_{nm}_{k}"]) for nm in LIGHT_NAMES for k in ["precision", "recall", "f1"]]
            + [f3(r["light_acc"]), f3(r["light_bal_acc"]), f3(r["light_macro_f1"])] for r in results]
    md += md_table(hdr, rows) + "\n"
    md += "Confusion matrices (rows = GT, columns = prediction, order No-Light / Green / Red):\n\n"
    for r in results:
        md += f"- `{short(r['name'])}`: {r['light_cm']}\n"

    md += "\n## 6. Figures\n\n"
    md += ("- `fig_v3_en_depth.png`: image, GT sensor (grey = invalid), pseudo label, 4 predictions; "
           "δ1 vs GT in the corner of each prediction.\n"
           "- `fig_v3_en_error.png`: per-pixel AbsRel vs GT on valid pixels.\n"
           "- `fig_v3_en_light.png`: per-class F1 and row-normalised confusion matrices.\n"
           "- `fig_v3_en_detection.png`: GT (white dashed) vs predicted bbox and keypoints per model.\n")
    open("results_v3_en.md", "w", encoding="utf-8").write(md)
    print("-> results_v3_en.md")


# ------------------------------------------------------------------ figures
@torch.no_grad()
def fig_depth_and_error(models, ds, n=6):
    idxs = pick_frames(ds, n)
    cmap = plt.get_cmap("turbo").copy(); cmap.set_bad("#bdbdbd")
    ecmap = plt.get_cmap("Reds").copy(); ecmap.set_bad("#bdbdbd")
    cols = ["Image", "GT sensor", "Pseudo label\n(Depth-Anything-V2)"] + [nm for nm, _ in models]
    f1, ax1 = plt.subplots(n, len(cols), figsize=(2.0 * len(cols), 3.7 * n))
    f2, ax2 = plt.subplots(n, len(models) + 2, figsize=(2.0 * (len(models) + 2), 3.7 * n))
    for r, i in enumerate(idxs):
        s = ds[i]
        img = s["image"].permute(1, 2, 0).numpy()
        gt = s["depth"][0].numpy(); valid = (s["depth_mask"][0].numpy() > 0.5) & (gt >= 0.5)
        preds = [m(s["image"][None].to(DEVICE))["depth"][0, 0].cpu().numpy() for _, m in models]
        label = f"{s['frame_id'][:8]}\n{s['frame_id'][9:15]}, {int(s['height_cm'])} cm"
        panels = [img, np.where(valid, gt, np.nan), s["teacher"][0].numpy()] + preds
        for c, p in enumerate(panels):
            a = ax1[r, c]
            a.imshow(p) if c == 0 else a.imshow(p, cmap=cmap, vmin=0, vmax=DMAX)
            if c >= 2 and valid.any():
                ratio = np.maximum(p[valid] / gt[valid], gt[valid] / np.clip(p[valid], 1e-3, None))
                a.text(4, 506, f"δ1={np.mean(ratio < 1.25):.2f}", fontsize=8, color="white",
                       bbox=dict(facecolor="black", alpha=0.65, pad=1.5, lw=0))
            a.set_xticks([]); a.set_yticks([])
            if c == 0: a.set_ylabel(label, fontsize=8)
            if r == 0: a.set_title(cols[c], fontsize=9)
        # error maps
        ax2[r, 0].imshow(img); ax2[r, 0].set_ylabel(label, fontsize=8)
        err_panels = [s["teacher"][0].numpy()] + preds
        for c, p in enumerate(err_panels):
            a = ax2[r, c + 1]
            err = np.where(valid, np.abs(p - gt) / np.clip(gt, 1e-3, None), np.nan)
            a.imshow(err, cmap=ecmap, vmin=0, vmax=0.5)
            a.text(4, 506, f"AbsRel={np.nanmean(err):.2f}", fontsize=8, color="white",
                   bbox=dict(facecolor="black", alpha=0.65, pad=1.5, lw=0))
        for c in range(len(models) + 2):
            ax2[r, c].set_xticks([]); ax2[r, c].set_yticks([])
            if r == 0:
                ax2[r, c].set_title((["Image", "Pseudo label"] + [nm for nm, _ in models])[c], fontsize=9)
    cb = f1.colorbar(plt.cm.ScalarMappable(cmap="turbo", norm=plt.Normalize(0, DMAX)), ax=ax1, fraction=0.015, pad=0.01)
    cb.set_label("Depth (m); grey = GT invalid")
    f1.subplots_adjust(top=1 - 0.55 / (3.7 * n)); f1.suptitle("Depth predictions on unseen test sessions (δ1 vs GT sensor, valid pixels)", y=1 - 0.15 / (3.7 * n), fontsize=12)
    f1.savefig("fig_v3_en_depth.png", dpi=90, bbox_inches="tight"); plt.close(f1)
    cb = f2.colorbar(plt.cm.ScalarMappable(cmap="Reds", norm=plt.Normalize(0, 0.5)), ax=ax2, fraction=0.02, pad=0.01)
    cb.set_label("|pred − GT| / GT  (clipped at 0.5); grey = GT invalid")
    f2.subplots_adjust(top=1 - 0.55 / (3.7 * n)); f2.suptitle("Relative error vs GT sensor (valid pixels only)", y=1 - 0.15 / (3.7 * n), fontsize=12)
    f2.savefig("fig_v3_en_error.png", dpi=90, bbox_inches="tight"); plt.close(f2)
    print("-> fig_v3_en_depth.png, fig_v3_en_error.png")


def fig_light(results):
    fig = plt.figure(figsize=(16, 8.8))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.05], hspace=0.5, wspace=0.35)
    ax = fig.add_subplot(gs[0, :])
    x = np.arange(4); bw = 0.2
    for k, r in enumerate(results):
        vals = [r[f"light_{nm}_f1"] for nm in LIGHT_NAMES] + [r["light_macro_f1"]]
        bars = ax.bar(x + (k - 1.5) * bw, vals, bw - 0.02, color=COLORS[k], label=short(r["name"]))
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color="#333")
    sup = results[0]
    ax.set_xticks(x, [f"{nm}\n(n={sup[f'light_{nm}_support']})" for nm in LIGHT_NAMES] + ["Macro-F1"])
    ax.set_ylim(0, 1.1); ax.set_ylabel("F1 score (test)")
    ax.set_title("Traffic light classification: per-class F1 (test split)", loc="left", fontsize=12, pad=28)
    ax.legend(ncol=4, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0))
    ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", color="#e5e5e5"); ax.set_axisbelow(True)
    for k, r in enumerate(results):
        a = fig.add_subplot(gs[1, k])
        cm = np.array(r["light_cm"], dtype=float); cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
        a.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                a.text(j, i, f"{100*cmn[i,j]:.0f}%\n({int(cm[i,j])})", ha="center", va="center", fontsize=9,
                       color="white" if cmn[i, j] > 0.55 else "#222")
        a.set_xticks(range(3), LIGHT_NAMES, fontsize=8); a.set_yticks(range(3), LIGHT_NAMES, fontsize=8)
        a.set_xlabel("Predicted"); a.set_ylabel("Ground truth" if k == 0 else "")
        a.set_title(short(r["name"]), fontsize=10, color=COLORS[k], fontweight="bold")
    fig.text(0.5, 0.47, "Confusion matrices, row-normalised (diagonal = per-class recall)", ha="center", fontsize=10)
    fig.savefig("fig_v3_en_light.png", dpi=110, bbox_inches="tight"); plt.close(fig)
    print("-> fig_v3_en_light.png")


def _draw_dets(a, bboxes, kpts, color, dashed=False, labels=True):
    H, W = 512, 256
    for (cx, cy, w, h), kp in zip(bboxes, kpts):
        a.add_patch(patches.Rectangle(((cx - w / 2) * W, (cy - h / 2) * H), w * W, h * H, fill=False,
                                      edgecolor=color, lw=1.6, ls="--" if dashed else "-"))
        pts = kp * torch.tensor([W, H])
        order = [0, 1, 3, 2, 0]                                   # TL TR BR BL TL outline
        a.plot(pts[order, 0], pts[order, 1], color=color, lw=0.9, ls="--" if dashed else "-", alpha=0.9)
        for j in range(4):
            a.scatter(pts[j, 0], pts[j, 1], s=22, c=KPT_COLORS[j], edgecolors="black", linewidths=0.6, zorder=5)


@torch.no_grad()
def fig_detection(models, ds_canon, n=8):
    idxs = pick_frames(ds_canon, n, need_object=True, seed=1)
    cols = ["Ground truth"] + [nm for nm, _ in models]
    fig, axes = plt.subplots(n, len(cols), figsize=(2.3 * len(cols), 4.0 * n))
    from box_ops import cxcywh_to_xyxy, box_iou
    for r, i in enumerate(idxs):
        s = ds_canon[i]
        img = s["image"].permute(1, 2, 0).numpy()
        om = s["obj_mask"] > 0.5
        gt_b, gt_k = s["bbox"][om], s["keypoints"][om][..., :2]
        for c in range(len(cols)):
            a = axes[r, c]; a.imshow(img); a.set_xticks([]); a.set_yticks([])
            if r == 0: a.set_title(cols[c], fontsize=10, color="black" if c == 0 else COLORS[c - 1], fontweight="bold")
        _draw_dets(axes[r, 0], gt_b, gt_k, "white", dashed=True)
        axes[r, 0].set_ylabel(f"{s['frame_id'][:15]}\n{int(s['height_cm'])} cm, {int(om.sum())} obj", fontsize=8)
        for c, (nm, m) in enumerate(models, start=1):
            out = m(s["image"][None].to(DEVICE))
            keep = torch.sigmoid(out["objectness"][0]).cpu() > 0.5
            pb = out["bbox"][0].cpu()[keep]
            pk = canonicalize_tensor(out["keypoints_reg"][0].cpu()[keep]) if keep.any() else out["keypoints_reg"][0].cpu()[keep]
            a = axes[r, c]
            _draw_dets(a, gt_b, gt_k, "white", dashed=True)
            _draw_dets(a, pb, pk, COLORS[c - 1])
            # IoU of slot-matched boxes (same convention as the metric)
            slot_iou = box_iou(cxcywh_to_xyxy(out["bbox"][0].cpu()), cxcywh_to_xyxy(s["bbox"]))[om]
            light = LIGHT_NAMES[out["light"][0].argmax().item()]
            a.text(3, 508, f"IoU {slot_iou.mean():.2f} | {int(keep.sum())} det\nlight: {light}", fontsize=7.5,
                   color="white", bbox=dict(facecolor="black", alpha=0.65, pad=1.5, lw=0), va="bottom")
        axes[r, 0].text(3, 508, f"light: {LIGHT_NAMES[int(s['light'])]}", fontsize=7.5, color="white",
                        bbox=dict(facecolor="black", alpha=0.65, pad=1.5, lw=0), va="bottom")
    handles = [plt.Line2D([], [], color="white", ls="--", lw=1.6, label="GT (dashed)")]
    handles += [plt.Line2D([], [], marker="o", ls="", markerfacecolor=KPT_COLORS[j], markeredgecolor="black",
                           label=CANON_NAMES[j]) for j in range(4)]
    leg = fig.legend(handles=handles, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1 - 0.45 / (4.0 * n)), frameon=True,
                     facecolor="#555", labelcolor="white", fontsize=9)
    fig.suptitle("Zebra crossing detection on test sessions: GT vs predicted bbox + keypoints "
                 "(objectness > 0.5; predicted keypoints canonicalised for display)", y=1 - 0.1 / (4.0 * n), fontsize=11)
    fig.subplots_adjust(top=1 - 1.0 / (4.0 * n))
    fig.savefig("fig_v3_en_detection.png", dpi=85, bbox_inches="tight"); plt.close(fig)
    print("-> fig_v3_en_detection.png")


if __name__ == "__main__":
    ckpts = sorted(glob.glob("checkpoints/v3_*_last.pt"),
                   key=lambda p: ORDER.index(short(parse_v3_name(p)["name"])))
    results = [evaluate_ckpt(p, CSV_PATH, COCO, "test") for p in ckpts]
    ds_t = CrossingDataset(CSV_PATH, "test", COCO, use_teacher=True, kpt_order="raw")
    ds_c = CrossingDataset(CSV_PATH, "test", COCO, use_teacher=False, kpt_order="canonical")
    models = load_models(ckpts)
    teacher = teacher_reference(ds_t)
    holes = hole_metrics(models, ds_t)
    write_markdown(results, teacher, holes, len(ds_t))
    fig_depth_and_error(models, ds_t)
    fig_light(results)
    fig_detection(models, ds_c)
