"""
visualize_pred.py — Visualisasi prediksi model di test set vs GT (5 kolom).
  kolom 1: RGB polos
  kolom 2: RGB + bbox (GT putih, pred kuning + skor) + info lampu (GT vs pred)
  kolom 3: depth GT (m, turbo) + colorbar
  kolom 4: depth teacher (m, jet) + colorbar   [hanya jika use_teacher=True]
  kolom 5: depth prediksi (m, turbo) + colorbar

Set cfg["use_teacher"] = True untuk menampilkan kolom teacher (butuh teacher_path
di CSV). Jika False, kolom teacher dilewati (jadi 4 kolom).
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.data import DataLoader

from dataset import CrossingDataset, collate_fn
from model import MultiTaskNet

LIGHT_NAMES = {0: "No-Light", 1: "Green", 2: "Red"}


def draw_boxes(ax, bbox, kpts, scores, W, H, box_color, thresh=0.5, show_score=False):
    M = bbox.shape[0]
    for si in range(M):
        if scores[si] < thresh:
            continue
        cx, cy, w, h = bbox[si].tolist()
        ax.add_patch(patches.Rectangle(((cx - w / 2) * W, (cy - h / 2) * H), w * W, h * H,
                     fill=False, edgecolor=box_color, lw=2))
        if show_score:
            ax.text((cx - w / 2) * W, (cy - h / 2) * H - 2, f"{scores[si]:.2f}",
                    color=box_color, fontsize=7, weight="bold")
        for j in range(kpts.shape[1]):
            kx, ky = kpts[si, j, 0].item(), kpts[si, j, 1].item()
            v = kpts[si, j, 2].item() if kpts.shape[2] > 2 else 1.0
            if v > 0 and (kx > 0 or ky > 0):
                ax.plot(kx * W, ky * H, "o", color=box_color, ms=5,
                        markeredgecolor="black", markeredgewidth=0.5)


@torch.no_grad()
def visualize(cfg):
    device = cfg["device"]
    use_teacher = cfg.get("use_teacher", False)

    ds = CrossingDataset(cfg["csv_path"], "test", cfg["coco_json"],
                         use_teacher=use_teacher, max_objects=cfg["max_objects"])
    dl = DataLoader(ds, batch_size=cfg["n_samples"], shuffle=cfg["shuffle"],
                    num_workers=0, collate_fn=collate_fn)

    model = MultiTaskNet(encoder_name=cfg["encoder_name"], pretrained=False,
                         max_objects=cfg["max_objects"]).to(device)
    state = torch.load(cfg["ckpt_path"], map_location=device, weights_only=False)
    model.load_state_dict(state, strict=False)
    model.eval()

    batch = next(iter(dl))
    for k in batch:
        if torch.is_tensor(batch[k]):
            batch[k] = batch[k].to(device)
    out = model(batch["image"])

    ncol = 5 if use_teacher else 4
    n = batch["image"].shape[0]
    fig, axes = plt.subplots(n, ncol, figsize=(3.4 * ncol, 3.5 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    for i in range(n):
        img = batch["image"][i].permute(1, 2, 0).cpu().numpy().clip(0, 1)
        H, W = img.shape[:2]
        gt_d = batch["depth"][i, 0].cpu().numpy()
        mask = batch["depth_mask"][i, 0].cpu().numpy() > 0.5
        pr_d = out["depth"][i, 0].cpu().numpy()
        gt_vis = np.where(mask, gt_d, np.nan)
        vmax = np.nanmax(gt_vis) if np.isfinite(gt_vis).any() else 20

        # kolom 0: RGB polos
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(batch["frame_id"][i], fontsize=8)
        axes[i, 0].axis("off")

        # kolom 1: RGB + bbox + lampu
        ax = axes[i, 1]
        ax.imshow(img)
        draw_boxes(ax, batch["bbox"][i].cpu(), batch["keypoints"][i].cpu(),
                   batch["obj_mask"][i].cpu(), W, H, "white", thresh=0.5)
        pred_scores = torch.sigmoid(out["objectness"][i]).cpu()
        draw_boxes(ax, out["bbox"][i].cpu(), out["keypoints_reg"][i].cpu(),
                   pred_scores, W, H, "lime", thresh=cfg["obj_thresh"], show_score=True)
        pl = LIGHT_NAMES[out["light"][i].argmax().item()]
        gl = LIGHT_NAMES[batch["light"][i].item()]
        ok = "OK" if pl == gl else "X"
        ax.set_title("bbox + light\nGT={} pred={} [{}]".format(gl, pl, ok), fontsize=8)
        ax.axis("off")

        # kolom 2: depth GT (turbo)
        im2 = axes[i, 2].imshow(gt_vis, cmap="turbo", vmin=0, vmax=vmax)
        axes[i, 2].set_title("depth GT (m)", fontsize=8)
        axes[i, 2].axis("off")
        fig.colorbar(im2, ax=axes[i, 2], fraction=0.046)

        if use_teacher:
            # kolom 3: depth teacher (jet)
            teacher = batch["teacher"][i, 0].cpu().numpy()
            has_t = batch["has_teacher"][i].item() > 0.5
            t_vis = np.where(teacher > 1e-3, teacher, np.nan) if has_t else np.full_like(teacher, np.nan)
            im_t = axes[i, 3].imshow(t_vis, cmap="jet", vmin=0, vmax=vmax)
            axes[i, 3].set_title("depth teacher (m)" if has_t else "teacher (none)", fontsize=8)
            axes[i, 3].axis("off")
            fig.colorbar(im_t, ax=axes[i, 3], fraction=0.046)
            pred_col = 4
        else:
            pred_col = 3

        # kolom terakhir: depth pred (turbo)
        im3 = axes[i, pred_col].imshow(pr_d, cmap="turbo", vmin=0, vmax=vmax)
        axes[i, pred_col].set_title("depth pred (m)", fontsize=8)
        axes[i, pred_col].axis("off")
        fig.colorbar(im3, ax=axes[i, pred_col], fraction=0.046)

    plt.tight_layout()
    if cfg["save_path"]:
        plt.savefig(cfg["save_path"], dpi=110, bbox_inches="tight")
        print("saved", cfg["save_path"])
    plt.show()


if __name__ == "__main__":
    decoder_name, loss = "efficientnet_b5", "softdelta"
    # use_teacher = "_w0.80" # "_no_teacher"
    # model_name = f"train_v2_{decoder_name}_{loss}{use_teacher}_last"
    # model_name = "train2p_efficientnet_b0_softdelta_last"
    model_name = "train_teacheronly_efficientnet_b5_softdelta_last"
     
    cfg = {
        "csv_path":   "dataset_v2.csv",
        "coco_json":  "new_ds/train/_annotations_coco.json",
        "ckpt_path":  f"checkpoints/{model_name}.pt",
        "encoder_name": decoder_name,
        "max_objects": 4,
        "use_teacher": True,          # True = tampilkan kolom teacher (jet)
        "n_samples": 5, 
        "shuffle": False, 
        "obj_thresh": 0.3,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "save_path": "",
    }
    visualize(cfg)