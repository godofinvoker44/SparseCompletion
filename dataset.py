"""
dataset.py

Satu dataset multi-task, baca daftar frame dari dataset.csv (kolom 'split').
Menghasilkan (per sampel):
    image        : [3,H,W] float [0,1]  (mentah; normalisasi/augment via transform)
    depth        : [1,H,W] meter (GT sensor, 0 = invalid)
    depth_mask   : [1,H,W] 1=valid (dalam [DEPTH_MIN,DEPTH_MAX)), else 0
    teacher      : [1,H,W] meter (jika use_teacher & file ada; else nol)
    has_teacher  : [1] 1 jika teacher ada
    bbox         : [M,4] cx,cy,w,h (0..1) per slot objek
    obj_mask     : [M]   1 = slot terisi
    keypoints    : [M,4,3] (x,y,visible) per slot
    light        : [] long (0=No-Light,1=Green,2=Red)
    height_cm    : [1] tinggi kamera dari nama file
    frame_id     : str

M = max_objects. Objek diurutkan dari TERDEKAT (paling bawah gambar) -> slot 0.
Anotasi bbox/keypoint dibaca dari coco_json (dicocokkan lewat frame_id).
"""

import os
import re
import csv
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

DEPTH_SCALE = 1.0 / 1000.0
DEPTH_MIN, DEPTH_MAX = 0.001, 20.0
IMG_SIZE = (512, 256)          # H, W
LIGHT_MAP = {"No-Light": 0, "Green": 1, "Red": 2}


def parse_frame_id(name):
    m = re.match(r"(\d{8}_\d{6}_\d+_\d+)", os.path.basename(name))
    return m.group(1) if m else None


class CrossingDataset(Dataset):
    def __init__(self, csv_path, split, coco_json,
                    use_teacher=False, max_objects=4, img_size=IMG_SIZE,
                    depth_scale=DEPTH_SCALE, depth_min=0.001, depth_max=20.0,
                    transform=None):
        self.use_teacher = use_teacher
        self.max_objects = max_objects
        self.img_size = img_size
        self.depth_scale = depth_scale
        self.depth_min, self.depth_max = depth_min, depth_max
        self.transform = transform

        # --- anotasi dari coco (bbox/keypoint), diindeks per frame_id ---
        with open(coco_json) as f:
            coco = json.load(f)
        self.obj_cat_id = next((c["id"] for c in coco["categories"] if "keypoints" in c), 1)
        id2fid = {im["id"]: parse_frame_id(im["file_name"]) for im in coco["images"]}
        self.wh = {parse_frame_id(im["file_name"]): (im["width"], im["height"])
                    for im in coco["images"]}
        self.anns_by_fid = {}
        for a in coco["annotations"]:
            fid = id2fid.get(a["image_id"])
            if fid:
                self.anns_by_fid.setdefault(fid, []).append(a)

        # --- daftar frame untuk split ini ---
        self.rows = []
        with open(csv_path) as f:
            for r in csv.DictReader(f):
                if r["split"] == split:
                    self.rows.append(r)

    def __len__(self):
        return len(self.rows)

    def _load_depth_npy(self, path):
        d = np.load(path).astype(np.float32)
        Ht, Wt = self.img_size
        return np.asarray(Image.fromarray(d).resize((Wt, Ht), Image.NEAREST), dtype=np.float32)

    def __getitem__(self, i):
        r = self.rows[i]
        fid = r["frame_id"]
        Ht, Wt = self.img_size
        ow, oh = self.wh.get(fid, (Wt, Ht))

        # image [0,1]
        img = Image.open(r["img_path"]).convert("RGB").resize((Wt, Ht), Image.BILINEAR)
        img = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)

        # GT sensor depth + mask
        depth = self._load_depth_npy(r["depth_path"]) * self.depth_scale
        depth = np.clip(depth, self.depth_min, self.depth_max)
        mask = ((depth >= self.depth_min) & (depth < self.depth_max)).astype(np.float32)
        depth = torch.from_numpy(depth).unsqueeze(0)
        depth_mask = torch.from_numpy(mask).unsqueeze(0)

        # teacher depth (opsional)
        teacher = torch.zeros(1, Ht, Wt, dtype=torch.float32)
        has_teacher = torch.zeros(1, dtype=torch.float32)
        if self.use_teacher and r.get("teacher_path"):
            tpath = r["teacher_path"]
            if tpath and os.path.exists(tpath):
                t = np.load(tpath).astype(np.float32)
                if t.shape != (Ht, Wt):
                    t = np.asarray(Image.fromarray(t).resize((Wt, Ht), Image.NEAREST), dtype=np.float32)
                teacher = torch.from_numpy(t).unsqueeze(0)
                has_teacher[0] = 1.0

        # deteksi multi-slot
        M = self.max_objects
        bbox = torch.zeros(M, 4, dtype=torch.float32)
        kpts = torch.zeros(M, 4, 3, dtype=torch.float32)
        obj_mask = torch.zeros(M, dtype=torch.float32)
        obj_anns = [a for a in self.anns_by_fid.get(fid, []) if a["category_id"] == self.obj_cat_id]

        def cy_of(a):
            x, y, w, h = a["bbox"]
            return (y + h/2) / oh
        
        obj_anns = sorted(obj_anns, key=cy_of, reverse=True)[:M]

        for si, a in enumerate(obj_anns):
            obj_mask[si] = 1.0
            x, y, w, h = a["bbox"]
            bbox[si] = torch.tensor([(x + w/2)/ow, (y + h/2)/oh, w/ow, h/oh], dtype=torch.float32)
            kp = a.get("keypoints", [])
            for j in range(min(4, len(kp)//3)):
                kx, ky, v = kp[3*j], kp[3*j+1], kp[3*j+2]
                kpts[si, j, 0] = kx/ow; kpts[si, j, 1] = ky/oh
                kpts[si, j, 2] = 1.0 if v > 0 else 0.0

        sample = {
            "image": img, "depth": depth, "depth_mask": depth_mask,
            "teacher": teacher, "has_teacher": has_teacher,
            "bbox": bbox, "obj_mask": obj_mask, "keypoints": kpts,
            "light": torch.tensor(LIGHT_MAP.get(r["light"], 0), dtype=torch.long),
            "height_cm": torch.tensor([float(r["height_cm"])], dtype=torch.float32),
            "frame_id": fid,
        }
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample


def collate_fn(batch):
    out = {}
    for k in ["image", "depth", "depth_mask", "teacher", "has_teacher", 
                "bbox", "obj_mask", "keypoints", "light", "height_cm"]:
        out[k] = torch.stack([b[k] for b in batch], dim=0)
        
    out["frame_id"] = [b["frame_id"] for b in batch]
    
    return out