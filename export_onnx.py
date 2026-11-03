"""
export_onnx.py

Export checkpoint MultiTaskNet -> ONNX (untuk deployment Android).
ONNX ini bisa dipakai langsung dengan ONNX Runtime Mobile, atau dikonversi
ke TFLite (LiteRT) via onnx2tf (lihat convert_tflite.py).

Input  : image [1,3,512,256] float32 RGB, nilai [0,1] (TANPA normalisasi ImageNet,
            sama seperti dataset.py)
Output : depth [1,1,512,256] meter, bbox [1,M,4] (cx,cy,w,h 0..1),
            objectness [1,M] logit, keypoints [1,M,4,2] (x,y 0..1),
            light [1,3] logit (0=No-Light,1=Green,2=Red), log_height [1,1]
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn

from model import MultiTaskNet
from dataset import IMG_SIZE

OUTPUT_NAMES = ["depth", "bbox", "objectness", "keypoints", "light", "log_height"]


class ExportWrapper(nn.Module):
    """dict output -> tuple, supaya urutan output ONNX tetap."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x, use_geometry=False)
        return tuple(out[k if k != "keypoints" else "keypoints_reg"] for k in OUTPUT_NAMES)


def encoder_from_ckpt(fname):
    from evaluate import parse_ckpt_name
    return parse_ckpt_name(os.path.basename(fname))["encoder"]


def export(ckpt, out_path, encoder=None, max_objects=4, opset=17):
    encoder = encoder or encoder_from_ckpt(ckpt)
    model = MultiTaskNet(encoder_name=encoder, pretrained=False, max_objects=max_objects)
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    wrapper = ExportWrapper(model).eval()

    H, W = IMG_SIZE
    dummy = torch.rand(1, 3, H, W)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.onnx.export(wrapper, dummy, out_path, opset_version=opset,
                        input_names=["image"], output_names=OUTPUT_NAMES,
                        do_constant_folding=True, dynamo=False)

    # verifikasi PyTorch vs ONNX Runtime
    import onnx
    import onnxruntime as ort
    onnx.checker.check_model(onnx.load(out_path))
    sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    with torch.no_grad():
        ref = [t.numpy() for t in wrapper(dummy)]
    got = sess.run(None, {"image": dummy.numpy()})
    print(f"[{encoder}] {ckpt} -> {out_path} ({os.path.getsize(out_path)/1e6:.1f} MB)")
    for name, r, g in zip(OUTPUT_NAMES, ref, got):
        print(f"  {name:11s} {str(g.shape):18s} max|diff| = {np.abs(r - g).max():.2e}")
    np.save(os.path.splitext(out_path)[0] + "_sample_input.npy", dummy.numpy())
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/train_v2_efficientnet_b0_softdelta_w0.50_last.pt")
    ap.add_argument("--out", default=None)
    ap.add_argument("--encoder", default=None, help="default: ditebak dari nama checkpoint")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    out = args.out or os.path.join(
        "export", os.path.basename(args.ckpt).replace("_last.pt", "") + ".onnx")
    export(args.ckpt, out, encoder=args.encoder, opset=args.opset)
