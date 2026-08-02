"""
model.py

Model multi-task ringan:
    encoder (timm, fallback torchvision) -> feature multi-skala
    UNet decoder -> depth map [1,H,W]
    global head  -> bbox [4], objectness [1], keypoints [4x3], light [3 logits]

Keypoint punya DUA jalur (bisa dipilih saat forward):
    - "regress": langsung regresi (x,y) dari fitur  (default)
    - "geometry": hitung (x,y) memakai height kamera + depth (Pythagoras-style)
        -> diaktifkan bertahap saat training (lihat train.py). Di sini disediakan
        fungsi bantu; penggabungannya dikontrol lewat argumen forward.

Ganti encoder cukup dengan mengganti string `encoder_name`.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------
# encoder builder: timm utama, torchvision fallback
# ------------------------------------------------------------
def build_encoder(encoder_name, pretrained=True, in_chans=3):
    """Return (encoder_module, list_of_feature_channels).
    encoder mengeluarkan list feature multi-skala (5 level, stride 2..32)."""
    try:
        import timm
        enc = timm.create_model(encoder_name, features_only=True,
                                pretrained=pretrained, in_chans=in_chans)
        chs = enc.feature_info.channels()
        return enc, chs, "timm"
    
    except ImportError:
        pass
    
    except Exception as e:
        raise RuntimeError(
            f"timm gagal membuat '{encoder_name}': {e}\n"
            f"Pastikan nama encoder valid untuk timm, atau install timm: pip install timm"
        )
        
    # ---- fallback torchvision (terbatas) ----
    try:
        import torchvision
        from torchvision.models.feature_extraction import create_feature_extractor
    except ImportError:
        raise ImportError("timm or torchvision is required, pip install timm")
        
    raise NotImplementedError(
        "Fallback torchvision belum dipetakan untuk encoder ini. "
        "Paling mudah: install timm (pip install timm) lalu pakai nama timm, "
        "mis. 'efficientnet_b0', 'mobilenetv3_small_100', 'mobilenetv2_100'."
    )


# ------------------------------------------------------------
# UNet decoder block
# ------------------------------------------------------------
class UpBlock(nn.Module):
    
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.reduce = nn.Conv2d(in_ch, out_ch, 1)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups=4, num_channels=out_ch), 
            nn.LeakyReLU(inplace=True),
            
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups=4, num_channels=out_ch), 
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.reduce(x)
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        
        return self.conv(x)


# ------------------------------------------------------------
# main model
# ------------------------------------------------------------
class MultiTaskNet(nn.Module):
    def __init__(self, encoder_name="efficientnet_b0", pretrained=True,
                    n_light=3, n_keypoints=4, max_objects=4, decoder_ch=(256, 128, 64, 32), 
                    min_depth=1e-3, max_depth=20.0
                    ):
        super().__init__()
        self.encoder, enc_chs, self.enc_src = build_encoder(encoder_name, pretrained)
        self.enc_chs = enc_chs
        self.n_keypoints = n_keypoints
        self.max_objects = max_objects
        self.min_depth = min_depth
        self.max_depth = max_depth

        c5, c4, c3, c2, c1 = enc_chs[-1], enc_chs[-2], enc_chs[-3], enc_chs[-4], enc_chs[-5]
        d1, d2, d3, d4 = decoder_ch
        
        self.up1 = UpBlock(c5, c4, d1)
        self.up2 = UpBlock(d1, c3, d2)
        self.up3 = UpBlock(d2, c2, d3)
        self.up4 = UpBlock(d3, c1, d4)
        self.depth_head = nn.Conv2d(d4, 1, 1)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head_feat = nn.Sequential(
                            nn.Linear(c5, 256), 
                            nn.LeakyReLU(inplace=True)
                        )
        M, K = max_objects, n_keypoints
        self.bbox_head   = nn.Linear(256, M * 4)        # cx,cy,w,h per slot
        self.obj_head    = nn.Linear(256, M)            # objectness per slot
        self.kpt_head    = nn.Linear(256, M * K * 2)    # x,y per keypoint per slot
        self.light_head  = nn.Linear(256, n_light)      # per gambar
        self.height_head = nn.Linear(256, 1)            # log(height_cm)

    def forward(self, x, height_cm=None, use_geometry=False):
        H, W = x.shape[-2:]
        feats = self.encoder(x)                # list 5 feature multi-skala
        f1, f2, f3, f4, f5 = feats[-5], feats[-4], feats[-3], feats[-2], feats[-1]

        # ----- depth decoder -----
        d = self.up1(f5, f4)
        d = self.up2(d, f3)
        d = self.up3(d, f2)
        d = self.up4(d, f1)
        depth = self.depth_head(d)
        depth = F.interpolate(depth, size=(H, W), mode="bilinear", align_corners=False)
        depth = F.softplus(depth)              # depth > 0 (meter)
        depth = depth.clamp(min=self.min_depth, max=self.max_depth)  # batasan realistis

        # ----- global head -----
        g = self.gap(f5).flatten(1)
        g = self.head_feat(g)
        B = x.shape[0]
        M, K = self.max_objects, self.n_keypoints

        bbox        = torch.sigmoid(self.bbox_head(g)).view(B, M, 4)    # [B,M,4]
        obj         = self.obj_head(g).view(B, M)                       # [B,M] logit
        kpt_reg     = torch.sigmoid(self.kpt_head(g)).view(B, M, K, 2)  # [B,M,K,2]
        light       = self.light_head(g)                                # [B,n_light]
        log_height  = self.height_head(g)                               # [B,1]

        out = {
            "depth": depth, 
            "bbox": bbox, 
            "objectness": obj,
            "keypoints_reg": kpt_reg, 
            "light": light,
            "log_height": log_height,
        }

        # for pred_key in out.keys(): 
        #     print(f"{pred_key} : {out[pred_key].min()} {out[pred_key].max()} {out[pred_key].mean()}")

        # ----- jalur geometry (opsional) -----
        # saat deployment height tak diketahui -> pakai height prediksi model
        if use_geometry:
            h_use = height_cm if height_cm is not None else torch.exp(log_height)
            out["keypoints_geo"] = self._geometry_keypoints(kpt_reg, depth, h_use)

        return out

    @staticmethod
    def _geometry_keypoints(kpt_reg, depth, height_cm):
        """
        Estimasi jarak horizontal tiap keypoint via Pythagoras:
            horizontal = sqrt(max(depth^2 - height^2, 0))
        kpt_reg : [B,M,K,2] (x,y normalized)
        depth   : [B,1,H,W] meter
        height  : [B,1] cm
        Return  : [B,M,K,3] -> (x, y, horizontal_dist_meter)
        """
        B, M, K, _ = kpt_reg.shape
        h_m = (height_cm.view(B, 1, 1) / 100.0)               # [B,1,1] meter

        grid = kpt_reg.clone() * 2.0 - 1.0                    # 0..1 -> -1..1
        grid = grid.view(B, M * K, 1, 2)                      # [B,M*K,1,2]
        d_at = F.grid_sample(depth, grid, mode="bilinear", align_corners=False)             # [B,1,M*K,1]
        d_at = d_at.view(B, M, K)                             # [B,M,K] depth di kpt
        horiz = torch.sqrt(torch.clamp(d_at**2 - h_m**2, min=0.0))  # [B,M,K]
        return torch.cat([kpt_reg, horiz.unsqueeze(-1)], dim=-1)    # [B,M,K,3]
    
    
if __name__ == '__main__': 
    model = MultiTaskNet(
        encoder_name="efficientnet_b4", 
        pretrained=True,
        n_light=3, 
        n_keypoints=4, 
        max_objects=4, 
        decoder_ch=(256, 128, 64, 32)
    )
    
    sample = torch.randn(1, 3, 256, 128)
    pred = model(sample)
    
    for pred_key in pred.keys(): 
        print(f"{pred_key} : {pred[pred_key].shape}")
    