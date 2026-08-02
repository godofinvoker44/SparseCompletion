"""
make_split_csv.py

Membuat dataset.csv yang mendaftar semua frame + info + pembagian train/test.
Split berdasarkan SESI: seluruh frame dari satu sesi rekaman masuk ke split yang
sama, sehingga scene di test benar-benar tak pernah terlihat saat training.

Sesi = bagian 'tanggal_waktu_tinggi' dari frame_id.
  frame_id 20260715_091851_160_000014 -> sesi '20260715_091851_160'

Pembagian ~80:20 di level sesi (bukan frame), diurutkan agar deterministik &
reproducible. Jalankan ulang kapan pun data bertambah -> CSV ter-update otomatis.

Kolom CSV:
  frame_id, session, height_cm, split,
  img_path, depth_path, teacher_path,
  has_object, light, n_kpt_visible

Atur parameter di blok if __name__ == "__main__".
"""

import os
import re
import csv
import glob
import json


def parse_frame_id(name):
    m = re.match(r"(\d{8}_\d{6}_\d+_\d+)", os.path.basename(name))
    return m.group(1) if m else None


def session_of(fid):
    """'20260715_091851_160_000014' -> '20260715_091851_160'"""
    parts = fid.split("_")
    return "_".join(parts[:3]) if len(parts) >= 4 else fid


def height_of(fid):
    parts = fid.split("_")
    return int(parts[2]) if len(parts) >= 3 and parts[2].isdigit() else -1


def build_csv(img_dir, depth_dir, coco_json, teacher_dir,
              out_csv, test_ratio=0.2, seed=42):
    # --- baca anotasi coco untuk info objek/light/keypoint ---
    with open(coco_json) as f:
        coco = json.load(f)
    obj_cat = next((c["id"] for c in coco["categories"] if "keypoints" in c), 1)

    anns_by_img = {}
    for a in coco["annotations"]:
        anns_by_img.setdefault(a["image_id"], []).append(a)

    # frame_id -> info dari coco
    info = {}
    for im in coco["images"]:
        fid = parse_frame_id(im["file_name"])
        if not fid:
            continue
        anns = [a for a in anns_by_img.get(im["id"], []) if a["category_id"] == obj_cat]
        tags = im.get("extra", {}).get("user_tags", [])
        light = next((t for t in tags if t in ("No-Light", "Green", "Red")), "No-Light")
        # total keypoint visible dari SEMUA objek di gambar ini
        n_kpt = sum(sum(1 for i in range(2, len(a.get("keypoints", [])), 3) if a["keypoints"][i] > 0)
                    for a in anns)
        info[fid] = {"file_name": im["file_name"], "has_object": int(bool(anns)),
                     "n_object": len(anns), "light": light, "n_kpt": n_kpt}

    # --- indeks depth & teacher ---
    depth_idx = {parse_frame_id(f): f for f in glob.glob(os.path.join(depth_dir, "*.npy"))
                 if parse_frame_id(f)}
    teacher_idx = {}
    if teacher_dir and os.path.isdir(teacher_dir):
        teacher_idx = {parse_frame_id(f): f for f in glob.glob(os.path.join(teacher_dir, "*.npy"))
                       if parse_frame_id(f)}

    # --- kumpulkan frame yang punya image + depth ---
    rows = []
    for fid, meta in info.items():
        if fid not in depth_idx:
            continue
        img_path = os.path.join(img_dir, meta["file_name"])
        if not os.path.exists(img_path):
            # coba resolusi nama (dots vs underscores)
            cands = glob.glob(os.path.join(img_dir, f"{fid}*"))
            if not cands:
                continue
            img_path = cands[0]
        rows.append({
            "frame_id": fid,
            "session": session_of(fid),
            "height_cm": height_of(fid),
            "img_path": img_path,
            "depth_path": depth_idx[fid],
            "teacher_path": teacher_idx.get(fid, ""),
            "has_object": meta["has_object"],
            "n_object": meta["n_object"],
            "light": meta["light"],
            "n_kpt_visible": meta["n_kpt"],
        })

    # --- split per SESI (deterministik) ---
    sessions = sorted(set(r["session"] for r in rows))
    # urutkan sesi, ambil ~test_ratio terakhir sbg test (deterministik & stabil
    # saat data nambah). Alternatif acak: pakai seed, tapi urutan lebih reproducible.
    n_test = max(1, round(len(sessions) * test_ratio))
    test_sessions = set(sessions[-n_test:])

    for r in rows:
        r["split"] = "test" if r["session"] in test_sessions else "train"

    # --- tulis CSV ---
    rows.sort(key=lambda r: r["frame_id"])
    fields = ["frame_id", "session", "height_cm", "split",
              "img_path", "depth_path", "teacher_path",
              "has_object", "n_object", "light", "n_kpt_visible"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # --- ringkasan ---
    n_train = sum(1 for r in rows if r["split"] == "train")
    n_test = sum(1 for r in rows if r["split"] == "test")
    print(f"total frame     : {len(rows)}")
    print(f"total sesi      : {len(sessions)}  (test: {len(test_sessions)} sesi)")
    print(f"train / test    : {n_train} / {n_test} frame "
          f"({100*n_test/max(len(rows),1):.1f}% test)")
    print(f"sesi test       : {sorted(test_sessions)}")
    print(f"CSV disimpan    : {out_csv}")
    return rows


if __name__ == "__main__":
    build_csv(
        img_dir     = "new_ds/train",
        depth_dir   = "new_ds/Depth",
        coco_json   = "new_ds/train/_annotations_coco.json",
        teacher_dir = "new_ds/Teacher",       # boleh kosong "" jika belum ada
        out_csv     = "dataset.csv",
        test_ratio  = 0.2,
        seed        = 42,
    )