# benchmark/f3m/modules_f3m.py
# Module F3M - Frequency-domain Feature Fusion Module (Wang et al., JMSE 2026, 14, 20).
# Paradigm "Separate - Project - Fuse" (Eq 1-4) + optional Spatial Attention (Eq 5-6, CBAM).
# Chu ky: F3M(c1, c2, r=0.5, gate=True); F3MWithSA(c1, c2, r=0.5, gate=True) them Stage 2.
# F3M giu nguyen kenh & do phan giai (c2 == c1). KHONG co loss phu - toi uu qua detection loss.

import torch
import torch.nn as nn
import torch.nn.functional as F


def _proj_channels(c, r):
    """C' = max(8, floor(r*C)) (F3M Eq 2)."""
    return max(8, int(c * r))


class SpatialAttention(nn.Module):
    """CBAM spatial attention (F3M Eq 5-6).
    Pool theo TRUC KENH (dim=1) -> 2 map (B,1,H,W) -> concat -> conv 7x7 -> sigmoid -> nhan lai."""

    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel SA phai le de pad 'same'"
        self.conv = nn.Conv2d(2, 1, kernel_size, stride=1, padding=kernel_size // 2, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)        # (B,1,H,W)
        mx, _ = torch.max(x, dim=1, keepdim=True)       # (B,1,H,W)
        m = self.act(self.conv(torch.cat([avg, mx], dim=1)))
        return x * m


class F3M(nn.Module):
    """Frequency-domain Feature Fusion Module - Stage 1 (Separate-Project-Fuse).

    Eq 1 (Separate): Xlf = AvgPool3x3(X) (co dinh, KHONG param); Xhf = X - Xlf.
    Eq 2 (Project) : X~lf = Plf(Xlf), X~hf = Phf(Xhf); Plf/Phf = 2 conv 1x1 RIENG, C -> C'.
    Eq 3 (Fuse)    : Ymid = Conv1x1(X~lf + X~hf), C' -> C. (Upsample neu ds>1 - khong dung o YOLO11n.)
    Eq 4 (Gate)    : G = sigmoid(Conv1x1([X, Ymid])); Y = X + G*Ymid. gate=False -> Y = X + Ymid.

    Chu ky (c1, c2, r, gate): c2 PHAI == c1 (khoi residual giu kenh). build_f3m ghi c2=c1
    tuong minh vao args de tai dung patch parse_model (c2 = args[1]).
    """

    def __init__(self, c1, c2=None, r=0.5, gate=True):
        super().__init__()
        if c2 is None:
            c2 = c1
        assert c2 == c1, f"F3M giu nguyen kenh: c2({c2}) phai == c1({c1})"
        self.c1 = c1
        self.cp = _proj_channels(c1, r)
        self.r = r
        self.use_gate = bool(gate)

        # Separate: low-pass = avg pool 3x3 stride1 pad1 (giu do phan giai, KHONG param)
        self.lowpass = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        # Project: 2 conv 1x1 RIENG BIET C -> C'
        self.proj_lf = nn.Conv2d(c1, self.cp, 1, bias=False)
        self.proj_hf = nn.Conv2d(c1, self.cp, 1, bias=False)
        # Fuse: conv 1x1 C' -> C
        self.fuse = nn.Conv2d(self.cp, c1, 1, bias=False)
        # Gate: concat [X, Ymid] (2C) -> C (per-channel-per-pixel)
        if self.use_gate:
            self.gate = nn.Conv2d(2 * c1, c1, 1, bias=True)
            self.gate_act = nn.Sigmoid()

    def forward(self, x):
        x_lf = self.lowpass(x)              # Eq 1
        x_hf = x - x_lf
        t_lf = self.proj_lf(x_lf)           # Eq 2
        t_hf = self.proj_hf(x_hf)
        y_mid = self.fuse(t_lf + t_hf)      # Eq 3
        # nhanh ds>1: o YOLO11n ca 2 vi tri deu stride-1 nen KHONG kich hoat
        if y_mid.shape[-2:] != x.shape[-2:]:
            y_mid = F.interpolate(y_mid, size=x.shape[-2:], mode="nearest")
        if self.use_gate:                   # Eq 4
            g = self.gate_act(self.gate(torch.cat([x, y_mid], dim=1)))
            return x + g * y_mid
        return x + y_mid


class F3MWithSA(F3M):
    """F3M + Stage 2 Spatial Attention (Eq 5-6). Dung o stem (r=0.33, gate=True)."""

    def __init__(self, c1, c2=None, r=0.5, gate=True):
        super().__init__(c1, c2, r, gate)
        self.sa = SpatialAttention(kernel_size=7)

    def forward(self, x):
        y = super().forward(x)              # Stage 1
        return self.sa(y)                   # Stage 2


if __name__ == "__main__":
    # smoke test: 2 cau hinh tich hop YOLO11n (stem 16ch + deep 256ch), giu shape, khong NaN.
    torch.manual_seed(0)
    # stem: F3MWithSA(16, r0.33, gate=True) + SA
    m1 = F3MWithSA(16, 16, r=0.33, gate=True).eval()
    x1 = torch.randn(2, 16, 160, 160)
    y1 = m1(x1)
    assert y1.shape == x1.shape, y1.shape
    # deep: F3M(256, r0.125, gate=False)
    m2 = F3M(256, 256, r=0.125, gate=False).eval()
    x2 = torch.randn(2, 256, 20, 20)
    y2 = m2(x2)
    assert y2.shape == x2.shape, y2.shape
    assert torch.isfinite(y1).all() and torch.isfinite(y2).all(), "NaN/Inf"
    p1 = sum(p.numel() for p in m1.parameters())
    p2 = sum(p.numel() for p in m2.parameters())
    print(f"OK F3MWithSA(16,r.33,gateT) cp={m1.cp} params={p1}  |  "
          f"F3M(256,r.125,gateF) cp={m2.cp} params={p2}  |  tong={(p1+p2)/1e6:.4f}M")