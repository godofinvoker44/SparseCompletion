"""
prepare_v3.py

Siapkan dataset v3 dari 2 zip baru (Nov 2026):
    "Zebra Crossing Annotation Ver.2.v1i.coco.zip"   (5000 gambar + COCO, semua di valid/)
    "depth&imu data baru november 3.zip"             (depth/*.npy, imu/ -> TIDAK dipakai)

Output:
    ds_v3/images/*.jpg
    ds_v3/depth/*.npy                (uint16 mm, 65535 = invalid -> terbuang oleh depth_max)
    ds_v3/_annotations.coco.json
    dataset_v3.csv                   (kolom sama dengan dataset_v2.csv)

Split per SESI (tanggal_jam_tinggi), acak dengan seed -> train / val / test.
Seluruh frame dari satu sesi masuk ke split yang sama (tidak ada kebocoran scene).
"""

import os
import re
import csv
import json
import random
import zipfile
import collections

COCO_ZIP = "Zebra Crossing Annotation Ver.2.v1i.coco.zip"
DEPTH_ZIP = "depth&imu data baru november 3.zip"
OUT_DIR = "ds_v3"
OUT_CSV = "dataset_v3.csv"
RATIOS = {"train": 0.7, "val": 0.1, "test": 0.2}
SEED = 42


def parse_frame_id(name):
    m = re.match(r"(\d{8}_\d{6}_\d+_\d+)", os.path.basename(name))
    return m.group(1) if m else None


def main():
    img_dir = os.path.join(OUT_DIR, "images")
    depth_dir = os.path.join(OUT_DIR, "depth")
    teacher_dir = os.path.join(OUT_DIR, "teacher")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)

    # ---- COCO: gambar + json ----
    zc = zipfile.ZipFile(COCO_ZIP)
    coco = json.loads(zc.read("valid/_annotations.coco.json"))
    json_out = os.path.join(OUT_DIR, "_annotations.coco.json")
    with open(json_out, "w") as f:
        json.dump(coco, f)
    for im in coco["images"]:
        dst = os.path.join(img_dir, im["file_name"])
        if not os.path.exists(dst):
            with open(dst, "wb") as f:
                f.write(zc.read("valid/" + im["file_name"]))

    # ---- depth: hanya frame yang punya gambar ----
    zd = zipfile.ZipFile(DEPTH_ZIP)
    depth_members = {parse_frame_id(n): n for n in zd.namelist()
                     if n.startswith("depth/") and n.endswith(".npy")}

    obj_cat = next(c["id"] for c in coco["categories"] if "keypoints" in c)
    anns_by_img = collections.defaultdict(list)
    for a in coco["annotations"]:
        if a["category_id"] == obj_cat:
            anns_by_img[a["image_id"]].append(a)

    rows = []
    for im in coco["images"]:
        fid = parse_frame_id(im["file_name"])
        if fid not in depth_members:
            print("  [skip, tanpa depth]", fid)
            continue
        dpath = os.path.join(depth_dir, fid + ".npy")
        if not os.path.exists(dpath):
            with open(dpath, "wb") as f:
                f.write(zd.read(depth_members[fid]))
        anns = anns_by_img.get(im["id"], [])
        tags = im.get("extra", {}).get("user_tags", [])
        light = next((t for t in tags if t in ("No-Light", "Green", "Red")), "No-Light")
        n_kpt = sum(1 for a in anns for i in range(2, len(a["keypoints"]), 3) if a["keypoints"][i] > 0)
        session = "_".join(fid.split("_")[:3])
        rows.append({
            "frame_id": fid, "session": session, "height_cm": int(fid.split("_")[2]),
            "img_path": os.path.join(img_dir, im["file_name"]),
            "depth_path": dpath,
            "teacher_path": os.path.join(teacher_dir, fid + ".npy"),   # dibuat make_teacher_v3.py
            "has_object": int(bool(anns)), "n_object": len(anns),
            "light": light, "n_kpt_visible": n_kpt,
        })

    # ---- split per sesi ----
    sessions = sorted({r["session"] for r in rows})
    random.Random(SEED).shuffle(sessions)
    n = len(sessions)
    n_test = round(n * RATIOS["test"]); n_val = round(n * RATIOS["val"])
    split_of = {s: ("test" if i < n_test else "val" if i < n_test + n_val else "train")
                for i, s in enumerate(sessions)}
    for r in rows:
        r["split"] = split_of[r["session"]]

    rows.sort(key=lambda r: r["frame_id"])
    fields = ["frame_id", "session", "height_cm", "split", "img_path", "depth_path",
              "teacher_path", "has_object", "n_object", "light", "n_kpt_visible"]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)

    # ---- ringkasan ----
    print(f"total frame {len(rows)} | sesi {n} | CSV -> {OUT_CSV}")
    for sp in ("train", "val", "test"):
        rs = [r for r in rows if r["split"] == sp]
        print(f"  {sp:5s}: {len(rs):5d} frame, {sum(split_of[s] == sp for s in sessions):3d} sesi | "
              f"objek {sum(r['n_object'] for r in rs):4d} | "
              f"light {dict(collections.Counter(r['light'] for r in rs))} | "
              f"tinggi {dict(sorted(collections.Counter(r['height_cm'] for r in rs).items()))}")


if __name__ == "__main__":
    main()
