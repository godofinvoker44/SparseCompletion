# v3 experiment grid: evaluation report (test split)

Encoder `efficientnet_b0`, depth loss SoftDelta, 50 epochs, single seed (42). Test split: 972 frames from 51 unseen sessions. Depth is always evaluated against the GT sensor depth (valid pixels, 0.5 to 20 m) unless stated otherwise.

Runs: `gt_*` = depth loss on GT sensor depth (valid pixels only); `pseudo_*` = depth loss on Depth-Anything-V2 pseudo labels (all pixels, no GT); `*_raw` = keypoints in annotation order; `*_canonical` = keypoints reordered to TL, TR, BL, BR.

## 1. Depth vs GT sensor (valid pixels)

| Model | δ1 ↑ | δ2 ↑ | δ3 ↑ | AbsRel ↓ | SqRel ↓ | RMSE (m) ↓ | RMSE log ↓ |
|---|---|---|---|---|---|---|---|
| gt_raw | 0.770 | 0.886 | 0.920 | 0.215 | 1.026 | 2.793 | 0.544 |
| gt_canonical | 0.770 | 0.889 | 0.923 | 0.211 | 0.966 | 2.777 | 0.539 |
| pseudo_raw | 0.704 | 0.895 | 0.944 | 0.260 | 1.057 | 2.263 | 0.291 |
| pseudo_canonical | 0.706 | 0.893 | 0.942 | 0.264 | 1.092 | 2.266 | 0.292 |
| *Teacher (pseudo label) ‡* | 0.766 | 0.925 | 0.965 | 0.202 | 0.711 | 2.017 | 0.250 |

‡ The pseudo label is calibrated per frame with that frame's own GT, so its numbers are an optimistic reference. The pseudo-trained students land close to it: they reproduce the teacher, including its errors.

## 2. Depth per distance zone (GT-valid pixels)

| Model | 0-5m δ1 | 0-5m AbsRel | 0-5m RMSE | 5-10m δ1 | 5-10m AbsRel | 5-10m RMSE | 10-20m δ1 | 10-20m AbsRel | 10-20m RMSE |
|---|---|---|---|---|---|---|---|---|---|
| gt_raw | 0.846 | 0.183 | 0.823 | 0.769 | 0.192 | 1.943 | 0.625 | 0.259 | 5.406 |
| gt_canonical | 0.843 | 0.182 | 0.829 | 0.775 | 0.188 | 1.930 | 0.632 | 0.255 | 5.349 |
| pseudo_raw | 0.711 | 0.322 | 1.564 | 0.750 | 0.190 | 1.783 | 0.611 | 0.189 | 3.521 |
| pseudo_canonical | 0.706 | 0.327 | 1.576 | 0.763 | 0.188 | 1.789 | 0.619 | 0.185 | 3.464 |
| *Teacher ‡* | 0.764 | 0.257 | 1.364 | 0.807 | 0.168 | 1.646 | 0.627 | 0.187 | 3.494 |

## 3. Depth in GT sensor holes (GT-invalid pixels)

The standard metrics above ignore these pixels. Reference = pseudo label (rough), so δ1/AbsRel favour the pseudo runs; the last column does not depend on the reference.

| Model | δ1 vs pseudo ↑ | AbsRel vs pseudo ↓ | Predicted < 2 m (implausible) ↓ |
|---|---|---|---|
| gt_raw | 0.341 | 0.531 | 33.5% |
| gt_canonical | 0.332 | 0.539 | 33.8% |
| pseudo_raw | 0.910 | 0.130 | 0.0% |
| pseudo_canonical | 0.912 | 0.129 | 0.0% |

## 4. Detection and keypoints

Keypoint error in pixels on the 256×512 image. *Canonical* error: prediction and GT are both reordered to TL/TR/BL/BR first, so raw and canonical runs are compared fairly. Spread = mean distance between the 4 predicted points ÷ same for GT (≈1 means no collapse).

| Model | bbox IoU ↑ | kpt err (px) ↓ | TL | TR | BL | BR | PCK@10px ↑ | Spread |
|---|---|---|---|---|---|---|---|---|
| gt_raw | 0.483 | 41.8 | 40.5 | 40.2 | 43.0 | 43.5 | 0.187 | 0.97 |
| gt_canonical | 0.485 | 40.8 | 38.8 | 39.1 | 43.0 | 42.2 | 0.207 | 1.06 |
| pseudo_raw | 0.564 | 39.3 | 36.4 | 36.5 | 43.6 | 40.8 | 0.263 | 1.02 |
| pseudo_canonical | 0.596 | 35.2 | 32.9 | 36.0 | 38.4 | 33.6 | 0.320 | 1.03 |

## 5. Traffic light classification

Support (test): No-Light = 868, Green = 39, Red = 65. Recall = per-class accuracy. With only 39 Green frames, one frame ≈ 2.6 points.

| Model | No-Light P | No-Light R | No-Light F1 | Green P | Green R | Green F1 | Red P | Red R | Red F1 | Accuracy | Balanced acc | Macro-F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gt_raw | 0.962 | 0.995 | 0.978 | 0.875 | 0.718 | 0.789 | 0.929 | 0.600 | 0.729 | 0.958 | 0.771 | 0.832 |
| gt_canonical | 0.963 | 1.000 | 0.981 | 0.929 | 0.667 | 0.776 | 0.953 | 0.631 | 0.759 | 0.962 | 0.766 | 0.839 |
| pseudo_raw | 0.968 | 0.999 | 0.983 | 0.939 | 0.795 | 0.861 | 1.000 | 0.662 | 0.796 | 0.968 | 0.818 | 0.880 |
| pseudo_canonical | 0.964 | 0.999 | 0.981 | 0.968 | 0.769 | 0.857 | 0.952 | 0.615 | 0.748 | 0.964 | 0.794 | 0.862 |

Confusion matrices (rows = GT, columns = prediction, order No-Light / Green / Red):

- `gt_raw`: [[864, 1, 3], [11, 28, 0], [23, 3, 39]]
- `gt_canonical`: [[868, 0, 0], [11, 26, 2], [22, 2, 41]]
- `pseudo_raw`: [[867, 1, 0], [8, 31, 0], [21, 1, 43]]
- `pseudo_canonical`: [[867, 1, 0], [7, 30, 2], [25, 0, 40]]

## 6. Figures

- `fig_v3_en_depth.png`: image, GT sensor (grey = invalid), pseudo label, 4 predictions; δ1 vs GT in the corner of each prediction.
- `fig_v3_en_error.png`: per-pixel AbsRel vs GT on valid pixels.
- `fig_v3_en_light.png`: per-class F1 and row-normalised confusion matrices.
- `fig_v3_en_detection.png`: GT (white dashed) vs predicted bbox and keypoints per model.
