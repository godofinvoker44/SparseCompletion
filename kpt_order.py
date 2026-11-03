"""
kpt_order.py

Kanonisasi urutan 4 keypoint zebra cross -> [TL, TR, BL, BR].

Kenapa tidak pakai kuadran dari titik tengah bbox?
    Zebra cross terlihat sebagai trapesium miring karena perspektif. Dua titik
    sering jatuh di kuadran yang sama -> label dobel / ada sudut yang hilang.
    Di dataset v2 ini terjadi pada ~80% objek.

Kenapa tidak cukup sort-y (2 teratas = atas, lalu sort-x)?
    Kalau trapesium miring cukup tajam, 2 titik dengan y terkecil bukan satu
    sisi -> poligon "terpelintir" (sisi saling silang). ~3.5% objek.

Metode di sini (berbasis tepi):
    1. urutkan 4 titik searah jarum jam berdasarkan sudut terhadap centroid
       -> poligon pasti tidak terpelintir.
    2. sisi "atas" = sisi (pasangan titik bertetangga) dengan rata-rata y terkecil.
    3. ujung kiri sisi atas = TL, lalu searah jarum jam: TR, BR, BL.
    Hasil selalu berupa permutasi (tiap sudut tepat satu titik).

Kalau titik visible < 4, urutan asli dipertahankan (tidak bisa ditentukan).
"""

import math

CANON_NAMES = ["TL", "TR", "BL", "BR"]


def canonical_perm(pts):
    """pts: list 4 (x, y). Return list idx [TL, TR, BL, BR] ke dalam pts."""
    cx = sum(p[0] for p in pts) / 4.0
    cy = sum(p[1] for p in pts) / 4.0
    # atan2 naik di koordinat gambar (y ke bawah) = searah jarum jam di layar
    cyc = sorted(range(4), key=lambda j: math.atan2(pts[j][1] - cy, pts[j][0] - cx))
    i = min(range(4), key=lambda i: pts[cyc[i]][1] + pts[cyc[(i + 1) % 4]][1])
    tl, tr, br, bl = (cyc[(i + k) % 4] for k in range(4))
    return [tl, tr, bl, br]


def canonicalize(kpts):
    """kpts: list 4 [x, y, v]. Return list baru urutan [TL, TR, BL, BR].
    Jika ada titik tidak visible, kembalikan urutan asli."""
    if len(kpts) != 4 or any(k[2] <= 0 for k in kpts):
        return [list(k) for k in kpts]
    perm = canonical_perm([(k[0], k[1]) for k in kpts])
    return [list(kpts[j]) for j in perm]


def canonicalize_tensor(xy):
    """xy: torch tensor [..., 4, 2] (mis. prediksi). Return tensor yang sama
    bentuknya, tiap grup 4 titik diurutkan ulang ke [TL, TR, BL, BR]."""
    import torch
    flat = xy.reshape(-1, 4, 2)
    out = torch.empty_like(flat)
    pts_all = flat.detach().cpu().tolist()
    for n, pts in enumerate(pts_all):
        out[n] = flat[n, canonical_perm(pts)]
    return out.reshape(xy.shape)
