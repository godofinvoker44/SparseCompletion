"""
convert_tflite.py

ONNX (dari export_onnx.py) -> TFLite / LiteRT via onnx2tf, lalu verifikasi
TFLite vs ONNX Runtime pada satu gambar asli dari split test.

Butuh env terpisah: tensorflow 2.19, onnx2tf, onnx, onnxruntime
(dipisah dari env training karena dependensinya berat & sering bentrok).

Catatan layout: onnx2tf mengubah input jadi NHWC -> image [1,512,256,3]
float32 RGB [0,1]. Output depth jadi [1,512,256,1].
"""

import os
import csv
import glob
import argparse
import shutil
import subprocess
import sys
import numpy as np
from PIL import Image

IMG_SIZE = (512, 256)  # H, W  (sama dengan dataset.py)


def load_test_images(n=1, csv_path="dataset_v2.csv"):
    """n gambar asli [n,3,H,W] [0,1]; fallback ke random jika csv tak ada."""
    H, W = IMG_SIZE
    xs, paths = [], []
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            for r in csv.DictReader(f):
                if r["split"] == "test" and os.path.exists(r["img_path"]):
                    img = Image.open(r["img_path"]).convert("RGB").resize((W, H), Image.BILINEAR)
                    xs.append(np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0)
                    paths.append(r["img_path"])
                    if len(xs) == n:
                        return np.stack(xs), paths
    return np.random.rand(n, 3, H, W).astype(np.float32), ["random"] * n


def to_nchw(a):
    return a.transpose(0, 3, 1, 2) if a.ndim == 4 and a.shape[-1] == 1 else a


def verify(onnx_path, tflite_path, x_nchw):
    import onnxruntime as ort
    import tensorflow as tf

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    names = [o.name for o in sess.get_outputs()]
    ref = dict(zip(names, sess.run(None, {"image": x_nchw})))

    # tanpa flatc, tflite tak punya signature & nama output jadi Identity, Identity_1, ...
    # (urutannya mengikuti urutan output ONNX)
    interp = tf.lite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()
    interp.set_tensor(interp.get_input_details()[0]["index"], x_nchw.transpose(0, 2, 3, 1))
    interp.invoke()
    outs = {d["name"]: interp.get_tensor(d["index"]) for d in interp.get_output_details()}
    got = {n: outs["Identity" if i == 0 else f"Identity_{i}"] for i, n in enumerate(names)}

    print(f"  {os.path.basename(tflite_path)} ({os.path.getsize(tflite_path)/1e6:.1f} MB)")
    for n in names:
        g = to_nchw(got[n])
        print(f"    {n:11s} {str(got[n].shape):18s} max|diff| = {np.abs(ref[n] - g.reshape(ref[n].shape)).max():.2e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="export/train_v2_efficientnet_b0_softdelta_w0.50.onnx")
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    onnx_path = os.path.abspath(args.onnx)
    out_dir = os.path.abspath(args.out_dir or os.path.splitext(args.onnx)[0] + "_tflite")

    # onnx2tf memanggil onnxsim via CLI -> folder Scripts env harus ada di PATH
    env = os.environ.copy()
    scripts = os.path.join(os.path.dirname(sys.executable), "Scripts")
    env["PATH"] = scripts + os.pathsep + env["PATH"]

    # onnx2tf mengunduh sampel gambar (20 x NHWC) dari GitHub untuk cek akurasi;
    # file lokal di cwd dipakai bila ada -> isi dengan frame test asli.
    work = os.path.join(out_dir, "_work")
    os.makedirs(work, exist_ok=True)
    calib, _ = load_test_images(20)
    np.save(os.path.join(work, "calibration_image_sample_data_20x128x128x3_float32.npy"),
            calib.transpose(0, 2, 3, 1))

    subprocess.run([shutil.which("onnx2tf", path=env["PATH"]), "-i", onnx_path, "-o", out_dir],
                    check=True, cwd=work, env=env)
    shutil.rmtree(work, ignore_errors=True)

    x, src = load_test_images(1)
    print(f"verifikasi dengan input: {src[0]}")
    for tfl in sorted(glob.glob(os.path.join(out_dir, "*.tflite"))):
        verify(onnx_path, tfl, x)
