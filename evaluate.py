import torch
import numpy as np
from torch.utils.data import DataLoader

from dataset import CrossingDataset, collate_fn
from model import MultiTaskNet
from box_ops import cxcywh_to_xyxy, box_iou


@torch.no_grad()
def compute_depth_metrics(pred, gt, mask, min_depth=0.5, max_depth=20.0):
    # ambil piksel valid, lalu saring ke rentang depth yang masuk akal
    p = pred[mask]
    g = gt[mask]
    valid = (g >= min_depth) & (g <= max_depth)      # buang GT ekstrem (mis. <0.5 m)
    p = p[valid].clamp(min=min_depth, max=max_depth)
    g = g[valid]
    if p.numel() == 0:
        return None

    diff = p - g
    rmse = torch.sqrt((diff ** 2).mean())
    rmse_log = torch.sqrt(((torch.log(p) - torch.log(g)) ** 2).mean())
    absrel = (diff.abs() / g).mean()
    sqrel = ((diff ** 2) / g).mean()
    ratio = torch.max(p / g, g / p)
    d1 = (ratio < 1.25).float().mean()
    d2 = (ratio < 1.25 ** 2).float().mean()
    d3 = (ratio < 1.25 ** 3).float().mean()
    return {"RMSE": rmse.item(), "RMSE_log": rmse_log.item(),
            "AbsRel": absrel.item(), "SqRel": sqrel.item(),
            "delta1": d1.item(), "delta2": d2.item(), "delta3": d3.item()}


@torch.no_grad()
def evaluate(cfg, verbose=True):
    device = cfg["device"]
    ds = CrossingDataset(cfg["csv_path"], "test", cfg["coco_json"],
                         use_teacher=False, max_objects=cfg["max_objects"])
    if verbose:
        print(f"test: {len(ds)} sampel")
        
    dl = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=False,
                    num_workers=cfg["num_workers"], collate_fn=collate_fn)

    model = MultiTaskNet(encoder_name=cfg["encoder_name"], pretrained=False,
                         max_objects=cfg["max_objects"]).to(device)
    # state = torch.load(cfg["ckpt_path"], map_location=device)
    state = torch.load(cfg["ckpt_path"], map_location=device, weights_only=False)
    model.load_state_dict(state, strict=False)
    model.eval()

    depth_sum = {}; n_depth = 0
    iou_sum = 0.0; iou_n = 0
    light_correct = 0; light_total = 0
    kpt_dist_sum = 0.0; kpt_n = 0        

    for batch in dl:
        for k in batch:
            if torch.is_tensor(batch[k]):
                batch[k] = batch[k].to(device)
        out = model(batch["image"])                # tanpa height (deployment-style)

        mask = batch["depth_mask"] > 0.5
        m = compute_depth_metrics(out["depth"], batch["depth"], mask)
        if m is not None:
            for k, v in m.items():
                depth_sum[k] = depth_sum.get(k, 0.0) + v
            n_depth += 1

        p = cxcywh_to_xyxy(out["bbox"]); t = cxcywh_to_xyxy(batch["bbox"])
        iou = box_iou(p, t)
        sm = batch["obj_mask"]
        iou_sum += (iou * sm).sum().item(); iou_n += sm.sum().item()

        pred_light = out["light"].argmax(dim=1)
        light_correct += (pred_light == batch["light"]).sum().item()
        light_total += batch["light"].numel()

        kpt_gt = batch["keypoints"][..., :2]
        vis = batch["keypoints"][..., 2] > 0.5             # [B,M,K]
        slot = batch["obj_mask"] > 0.5                     # [B,M]
        valid_kpt = vis & slot.unsqueeze(-1)               # [B,M,K]
        dist = torch.sqrt(((out["keypoints_reg"] - batch["keypoints"][..., :2]) ** 2).sum(-1) + 1e-12)
        kpt_dist_sum += (dist * valid_kpt.float()).sum().item()
        kpt_n += valid_kpt.sum().item()

    if verbose:
        print("\n" + "=" * 50)
        print("DEPTH (meter, di piksel valid GT sensor)")
        print("=" * 50)

        for k in ["RMSE", "RMSE_log", "AbsRel", "SqRel", "delta1", "delta2", "delta3"]:
            print(f"  {k:10s}: {depth_sum[k]/n_depth:.4f}")

        print("\nDETEKSI")

        print(f"  mean IoU  : {iou_sum/max(iou_n,1):.4f}")
        print("\nLIGHT")
        print(f"  accuracy  : {light_correct/max(light_total,1):.4f}")

        print("\nKEYPOINT")
        print(f"  mean distance  : {kpt_dist_sum/max(kpt_n,1):.4f}")

    else: 
        out = {}
        for k in ["RMSE", "RMSE_log", "AbsRel", "SqRel", "delta1", "delta2", "delta3"]:
            out[k] = depth_sum[k]/n_depth

        out["mean_iou"] = iou_sum/max(iou_n,1)
        out["light_acc"] = light_correct/max(light_total,1)
        out["kpt_dist"] = kpt_dist_sum/max(kpt_n,1)

        return out


if __name__ == "__main__":
    decoder_name, loss = "efficientnet_b0", "softdelta"
    use_teacher = "_w0.10" # "_no_teacher"
    model_name = f"train_{decoder_name}_{loss}{use_teacher}_last"

    print("model_name:", model_name)
    cfg = {
        "csv_path":   "dataset.csv",
        "coco_json":  "new_ds/train/_annotations_coco.json",
        "ckpt_path":  f"checkpoints/{model_name}.pt",       # sesuaikan ke checkpoint-mu
        "encoder_name": decoder_name,         # samakan dgn saat training
        "max_objects": 4,
        "batch_size": 16, 
        "num_workers": 4,
        "device": "cuda"
    }
    evaluate(cfg)
    

    # -----------------------------------------------
    # multi
    # -----------------------------------------------
    # decoder_names = [
    #     'efficientnet_b0', 
    #     'efficientnet_b3', 
    #     'efficientnet_b5',
    #     "mobilenetv4_conv_small",
    #     "mobilenetv3_small_100",  # Mobile standard
    #     "mobilenetv3_large_100",  # Mobile ultra-ringan
    #     "regnety_002",            # Mobile NPU optimized
    #     "regnetx_002",            # Fast mobile (no SE)
    #     "resnet10t",              # ResNet ultra-ringan
    #     "resnet18",               # General edge baseline
    # ]       
    # use_teacher = "_no_teacher"
    # loss = 'softdelta'

    # collected_results = {}

    # for decoder_name in decoder_names:
    #     model_name = f"train_{decoder_name}_{loss}{use_teacher}_last"

    #     cfg = {
    #         "csv_path":   "dataset.csv",
    #         "coco_json":  "new_ds/train/_annotations_coco.json",
    #         "ckpt_path":  f"checkpoints/{model_name}.pt",       # sesuaikan ke checkpoint-mu
    #         "encoder_name": decoder_name,         # samakan dgn saat training
    #         "max_objects": 4,
    #         "batch_size": 16, 
    #         "num_workers": 4,
    #         "device": "cuda"
    #     }
    #     pred = evaluate(cfg, verbose=False)
        
    #     collected_results[decoder_name] = pred

    # import pandas as pd 
    # df = pd.DataFrame.from_dict(collected_results, orient='index')
    # print("\n\nSummary of results (as DataFrame):")
    # print(df)