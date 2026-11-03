"""
evaluate_tflite.py

Evaluasi model .tflite (hasil convert_tflite.py) di split test, dengan metrik
yang SAMA seperti evaluate.py (batch_size=1), supaya bisa dibandingkan langsung
dengan baris PyTorch di eval_results.csv.

Butuh: torch (untuk dataset & metrik) + ai-edge-litert (interpreter tflite).
"""

import time
import argparse
import numpy as np
import torch
from ai_edge_litert.interpreter import Interpreter

from dataset import CrossingDataset
from evaluate import compute_depth_metrics, compute_depth_metrics_zoned, ZONES
from box_ops import cxcywh_to_xyxy, box_iou

# urutan output tflite (Identity, Identity_1, ...) = urutan output ONNX
OUTPUT_NAMES = ["depth", "bbox", "objectness", "keypoints", "light", "log_height"]


class TFLiteModel:
    def __init__(self, path, num_threads=4):
        self.interp = Interpreter(model_path=path, num_threads=num_threads)
        self.interp.allocate_tensors()
        self.in_idx = self.interp.get_input_details()[0]["index"]
        by_name = {d["name"]: d["index"] for d in self.interp.get_output_details()}
        self.out_idx = [by_name["Identity" if i == 0 else f"Identity_{i}"]
                        for i in range(len(OUTPUT_NAMES))]

    def __call__(self, img_nchw):
        """img [1,3,H,W] [0,1] -> dict tensor dengan layout sama seperti PyTorch."""
        self.interp.set_tensor(self.in_idx, img_nchw.permute(0, 2, 3, 1).numpy())
        self.interp.invoke()
        out = {n: torch.from_numpy(self.interp.get_tensor(i).copy())
               for n, i in zip(OUTPUT_NAMES, self.out_idx)}
        out["depth"] = out["depth"].permute(0, 3, 1, 2)    # NHWC -> NCHW
        return out


@torch.no_grad()
def evaluate_tflite(tflite_path, csv_path="dataset_v2.csv",
                    coco_json="new_ds/train/_annotations_coco.json", num_threads=4):
    ds = CrossingDataset(csv_path, "test", coco_json, use_teacher=False, max_objects=4)
    model = TFLiteModel(tflite_path, num_threads=num_threads)

    depth_sum = {}; n_depth = 0
    zone_sum = {f"{int(a)}-{int(b)}m": {} for (a, b) in ZONES}
    zone_n = {k: 0 for k in zone_sum}
    iou_sum = iou_n = light_correct = kpt_dist_sum = kpt_n = 0.0
    times = []

    for i in range(len(ds)):
        s = ds[i]
        img = s["image"].unsqueeze(0)
        t0 = time.perf_counter()
        out = model(img)
        times.append(time.perf_counter() - t0)

        gt, mask = s["depth"].unsqueeze(0), s["depth_mask"].unsqueeze(0) > 0.5
        m = compute_depth_metrics(out["depth"], gt, mask)
        if m is not None:
            for k, v in m.items():
                depth_sum[k] = depth_sum.get(k, 0.0) + v
            n_depth += 1
        for zlabel, zm in compute_depth_metrics_zoned(out["depth"], gt, mask).items():
            if zm is not None:
                for k, v in zm.items():
                    zone_sum[zlabel][k] = zone_sum[zlabel].get(k, 0.0) + v
                zone_n[zlabel] += 1

        sm = s["obj_mask"].unsqueeze(0)
        iou = box_iou(cxcywh_to_xyxy(out["bbox"]), cxcywh_to_xyxy(s["bbox"].unsqueeze(0)))
        iou_sum += (iou * sm).sum().item(); iou_n += sm.sum().item()

        light_correct += int(out["light"].argmax(1).item() == s["light"].item())

        kp = s["keypoints"].unsqueeze(0)
        valid = (kp[..., 2] > 0.5) & (sm > 0.5).unsqueeze(-1)
        dist = torch.sqrt(((out["keypoints"] - kp[..., :2]) ** 2).sum(-1) + 1e-12)
        kpt_dist_sum += (dist * valid.float()).sum().item(); kpt_n += valid.sum().item()

    res = {k: depth_sum[k] / n_depth for k in depth_sum}
    for zlabel in zone_sum:
        for k in ["RMSE", "delta1", "AbsRel"]:
            res[f"{zlabel}_{k}"] = zone_sum[zlabel][k] / max(zone_n[zlabel], 1)
    res["mean_iou"] = iou_sum / max(iou_n, 1)
    res["light_acc"] = light_correct / len(ds)
    res["kpt_dist"] = kpt_dist_sum / max(kpt_n, 1)
    res["latency_ms_cpu"] = 1000 * float(np.median(times[5:]))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("tflite", nargs="+")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    for path in args.tflite:
        r = evaluate_tflite(path, num_threads=args.threads)
        print(f"\n== {path}")
        for k, v in r.items():
            print(f"  {k:16s}: {v:.4f}")
