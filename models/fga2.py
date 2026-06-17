# models/fga2.py
# Module (5): Frequency-Gated Area-Attention (FGA2).
# Thay A2C2f tai neck top-down (idx 11, 14). Chu ky: __init__(c1, c2, n, area, lambda_init).

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules import Conv

from .common import HaarDWT


class FG_AAttn(nn.Module):
    """Area-Attention (AAttn) cua YOLOv12 - GIU NGUYEN cau truc & convention,
    CHI sua logit attention:
        attn = softmax(Q K^T / sqrt(d) + lam * b_key).
    - b_key: bias theo vi tri KEY, lay tu nang luong cao tan DWT (chuan hoa) cua input block.
    - lam: scalar HOC DUOC, khoi tao 0 => bias = 0 => thu ve area-attention goc.
    Khac AAttn goc DUY NHAT o so hang `lam * b_key` cong vao logit truoc softmax.
    """

    def __init__(self, dim, num_heads, area=1, lambda_init=0.0):
        super().__init__()
        self.area = area
        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        all_head_dim = head_dim * self.num_heads
        self.qkv = Conv(dim, all_head_dim * 3, 1, act=False)
        self.proj = Conv(all_head_dim, dim, 1, act=False)
        self.pe = Conv(all_head_dim, dim, 7, 1, 3, g=dim, act=False)   # PE = DWConv 7x7 tren V (giong AAttn)
        self.lam = nn.Parameter(torch.full((1,), float(lambda_init)))

    def forward(self, x, eb):
        # x: (B,C,H,W); eb: (B,1,H,W) nang luong cao tan da chuan hoa per-image
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x).flatten(2).transpose(1, 2)        # (B, N, 3C)
        ebk = eb.flatten(2).transpose(1, 2)                  # (B, N, 1)
        if self.area > 1:                                    # chia 'area' vung token (giong AAttn goc)
            qkv = qkv.reshape(B * self.area, N // self.area, C * 3)
            ebk = ebk.reshape(B * self.area, N // self.area, 1)
            B, N, _ = qkv.shape

        q, k, v = (
            qkv.view(B, N, self.num_heads, self.head_dim * 3)
            .permute(0, 2, 3, 1)
            .split([self.head_dim, self.head_dim, self.head_dim], dim=2)
        )                                                    # q,k,v: (B, heads, head_dim, N)
        attn = (q.transpose(-2, -1) @ k) * (self.head_dim ** -0.5)   # (B, heads, N, N): Q K^T / sqrt(d)
        bkey = ebk.transpose(1, 2).unsqueeze(1)              # (B, 1, 1, N): bias theo cot key
        attn = attn + self.lam * bkey                         # + lam * b_key  (<< thay doi DUY NHAT)
        attn = attn.softmax(dim=-1)
        x = v @ attn.transpose(-2, -1)                        # (B, heads, head_dim, N)
        x = x.permute(0, 3, 1, 2)                             # (B, N, heads, head_dim)
        v = v.permute(0, 3, 1, 2)

        if self.area > 1:
            x = x.reshape(B // self.area, N * self.area, C)
            v = v.reshape(B // self.area, N * self.area, C)
            B, N, _ = x.shape

        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        v = v.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        x = x + self.pe(v)
        return self.proj(x)


class FG_ABlock(nn.Module):
    """ABlock cua YOLOv12 (Attention + MLP, residual) - GIU NGUYEN, attention dung FG_AAttn."""

    def __init__(self, dim, num_heads, mlp_ratio=2.0, area=1, lambda_init=0.0):
        super().__init__()
        self.attn = FG_AAttn(dim, num_heads=num_heads, area=area, lambda_init=lambda_init)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(Conv(dim, mlp_hidden_dim, 1), Conv(mlp_hidden_dim, dim, 1, act=False))

    def forward(self, x, eb):
        x = x + self.attn(x, eb)
        return x + self.mlp(x)


class _FG_ABlockPair(nn.Module):
    """Tuong duong nn.Sequential(ABlock, ABlock) cua A2C2f (a2=True) nhung truyen them eb."""

    def __init__(self, dim, num_heads, mlp_ratio, area, lambda_init):
        super().__init__()
        self.b1 = FG_ABlock(dim, num_heads, mlp_ratio, area, lambda_init)
        self.b2 = FG_ABlock(dim, num_heads, mlp_ratio, area, lambda_init)

    def forward(self, x, eb):
        return self.b2(self.b1(x, eb), eb)


class FGA2_A2C2f(nn.Module):
    """A2C2f (R-ELAN, a2=True) cua YOLOv12 - GIU NGUYEN kien truc & cac lop
    (cv1 -> hidden c_=c2*e -> n nhom (2xABlock) -> concat -> cv2, gamma residual tuy chon),
    CHI thay attention noi bo bang area-attention co cong tan so:
        attn = softmax(Q K^T / sqrt(d) + lam * b_key).
    b_key tinh 1 lan tu nang luong cao tan DWT (Haar) cua input block, thread xuong moi AAttn.

    Args (ghi tuong minh trong YAML da patch):
        c1, c2: kenh vao/ra (vd idx11 yolov12n: 384 -> 128)
        n: so nhom R-ELAN (build_model truyen depth-scaled repeats; neck n=1)
        area: so vung area-attention (l=4 theo de xuat)
        lambda_init: khoi tao lam (0.0 => area-attention thuan)
        e: ty le hidden (0.5 giong A2C2f); mlp_ratio: 2.0 giong A2C2f;
        residual: bat gamma residual (mac dinh tat o neck vi c1 != c2)
    """

    def __init__(self, c1, c2, n=2, area=4, lambda_init=0.0, e=0.5, mlp_ratio=2.0, residual=False):
        super().__init__()
        c_ = int(c2 * e)                          # hidden channels (giong A2C2f)
        num_heads = max(c_ // 32, 1)              # giong A2C2f: ABlock(c_, c_ // 32, ...)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv((1 + n) * c_, c2, 1)       # concat (1 + n) nhanh -> c2 (R-ELAN)
        self.gamma = nn.Parameter(0.01 * torch.ones(c2)) if residual else None
        self.dwt = HaarDWT()
        self.m = nn.ModuleList(
            _FG_ABlockPair(c_, num_heads, mlp_ratio, area, lambda_init) for _ in range(max(n, 1))
        )

    def _freq_energy(self, x):
        """E = mean_c(|LH| + |HL| + |HH|), chuan hoa z-score per-image, ve (H,W)."""
        _, lh, hl, hh = self.dwt(x)
        e = (lh.abs() + hl.abs() + hh.abs()).mean(1, keepdim=True)   # (B,1,H/2,W/2)
        e = F.interpolate(e, size=x.shape[-2:], mode="nearest")
        mu = e.mean(dim=(2, 3), keepdim=True)
        sd = e.std(dim=(2, 3), keepdim=True)
        return (e - mu) / (sd + 1e-6)

    def forward(self, x):
        eb = self._freq_energy(x)                 # b_key tu input (truoc cv1), (B,1,H,W)
        y = [self.cv1(x)]
        for m in self.m:
            y.append(m(y[-1], eb))                # moi nhom R-ELAN deu nhan eb
        out = self.cv2(torch.cat(y, 1))
        if self.gamma is not None:
            return x + self.gamma.view(1, -1, 1, 1) * out
        return out