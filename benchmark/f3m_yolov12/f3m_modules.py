# ultralytics/nn/modules/f3m_modules.py
"""
F3M: Frequency-domain Feature Fusion Module
============================================
Paper : "F3M: A Frequency-Domain Feature Fusion Module for Robust
         Underwater Object Detection", J. Mar. Sci. Eng. 2026, 14, 20.
Paradigm : Separate -> Project -> Fuse  (+ optional Spatial Attention)

>>> Logic theo dung cong thuc Eq.(1)-(6). <<<

DIEM QUAN TRONG VE TICH HOP (da sua loi 'in_channels must be divisible by groups'):
  - F3M / F3MWithSA suy ra SO KENH TU DONG o forward dau tien (LAZY build).
  - Nho do KHONG can chen in_channels qua parser, tuc KHONG can sua parse_model
    trong ultralytics/nn/tasks.py.
  - YAML chi ghi hyperparam, vi du:
        - [-1, 1, F3MWithSA, [0.33, True, 7]]   # r, gate, sa_kernel
        - [-1, 1, F3M,       [0.125, False]]    # r, gate
    Parser di vao nhanh mac dinh 'else: c2 = ch[f]' (giu nguyen so kenh - dung voi
    thiet ke residual), va goi F3M(0.125, False) / F3MWithSA(0.33, True, 7).

Ghi chu trung thanh voi bai bao:
  - Eq.(2)/Eq.(3): projection va fuse la POINTWISE CONV TUYEN TINH (khong BN/ReLU).
    Khong dat ReLU sau projection vi se triet tieu phan AM cua high-frequency
    (X_hf = X - X_lf co the < 0).
  - Module RESIDUAL BAO TOAN CHIEU: out_channels == in_channels, H x W khong doi.
"""

import torch
import torch.nn as nn

__all__ = ("F3M", "F3MWithSA")


# =====================================================================
# Stage 1-A : Frequency Decomposition  (SEPARATE)   --- Eq.(1)
# =====================================================================
class FrequencySeparator(nn.Module):
    """
    Eq.(1):  X_lf = AvgPool_{3x3}(X) ,  X_hf = X - X_lf
    - 3x3 average pooling lam low-pass filter CO DINH (0 params).
    - stride=1, padding=1  -> giu nguyen H x W. AvgPool2d von depthwise (theo kenh).
    """

    def __init__(self):
        super().__init__()
        self.avg_pool = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)

    def forward(self, x):
        x_lf = self.avg_pool(x)
        x_hf = x - x_lf
        return x_lf, x_hf


# =====================================================================
# Stage 1-B : Adaptive Feature Projection  (PROJECT)   --- Eq.(2)
# =====================================================================
class FrequencyProjector(nn.Module):
    """
    Eq.(2):  X~_lf = P_lf(X_lf) ,  X~_hf = P_hf(X_hf)
    P_lf, P_hf : 1x1 pointwise conv  C -> C' = max(8, floor(r*C)).  TUYEN TINH.
    """

    def __init__(self, in_channels: int, reduction_ratio: float = 0.25):
        super().__init__()
        assert 0.0 < reduction_ratio <= 1.0, "r phai thuoc (0, 1]"
        mid_channels = max(8, int(in_channels * reduction_ratio))
        self.mid_channels = mid_channels
        self.proj_lf = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=True)
        self.proj_hf = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=True)

    def forward(self, x_lf, x_hf):
        return self.proj_lf(x_lf), self.proj_hf(x_hf)


# =====================================================================
# Stage 1-C : Feature Fusion + Gated Residual  (FUSE)   --- Eq.(3)(4)
# =====================================================================
class FrequencyFuser(nn.Module):
    """
    Eq.(3):  Y_mid = Conv_{1x1}(X~_lf + X~_hf)            # SUM roi 1x1 conv C'->C
    Eq.(4):  G = sigma(Conv_{1x1}([X, Y_mid])) ,  Y = X + G (.) Y_mid
             use_gate=False -> Y = X + Y_mid (residual chuan).
    """

    def __init__(self, in_channels: int, mid_channels: int, use_gate: bool = True):
        super().__init__()
        self.use_gate = use_gate
        self.fuse_conv = nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=True)
        if use_gate:
            self.gate_conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=True)
            self.sigmoid = nn.Sigmoid()

    def forward(self, x, x_lf_proj, x_hf_proj):
        y_mid = self.fuse_conv(x_lf_proj + x_hf_proj)                 # Eq.(3)
        if self.use_gate:
            g = self.sigmoid(self.gate_conv(torch.cat([x, y_mid], dim=1)))
            return x + g * y_mid                                     # Eq.(4) gated
        return x + y_mid                                             # residual chuan


# =====================================================================
# Stage 2 : Spatial Attention Module  (CBAM-style)   --- Eq.(5)(6)
# =====================================================================
class SpatialAttentionModule(nn.Module):
    """
    Eq.(6):  M_spatial = sigma(Conv_{7x7}([AvgPool(Y), MaxPool(Y)]))   # pool theo kenh
    Eq.(5):  Y~ = Y (.) M_spatial
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size phai le"
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size,
                              padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, y):
        avg_out = torch.mean(y, dim=1, keepdim=True)
        max_out, _ = torch.max(y, dim=1, keepdim=True)
        m = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return y * m


# =====================================================================
# (1) F3M  ->  dat tai NECK (pre-SPPF)   [LAZY channel inference]
# =====================================================================
class F3M(nn.Module):
    """
    F3M = Separate -> Project -> Fuse (gated residual). Bao toan chieu.
    Cau hinh "lite" theo bai bao: reduction_ratio=0.125, use_gate=False.

    Chu ky goi tu YAML:  F3M(reduction_ratio, use_gate)
    So kenh in_channels duoc suy ra TU DONG o forward dau tien (lazy build),
    nen KHONG can sua parse_model.
    """

    def __init__(self, reduction_ratio: float = 0.125, use_gate: bool = False):
        super().__init__()
        self.reduction_ratio = float(reduction_ratio)
        self.use_gate = bool(use_gate)
        self.separator = FrequencySeparator()   # khong tham so
        # Xay dung lazy khi biet so kenh:
        self.projector = None
        self.fuser = None
        self._built = False

    def _build(self, in_channels: int):
        self.projector = FrequencyProjector(in_channels, self.reduction_ratio)
        self.fuser = FrequencyFuser(in_channels, self.projector.mid_channels, self.use_gate)
        self._built = True

    def forward(self, x):
        if not self._built:
            self._build(x.shape[1])
            self.to(x.device)                   # dam bao conv moi tao nam cung device voi x
        x_lf, x_hf = self.separator(x)           # Eq.(1)
        x_lf_p, x_hf_p = self.projector(x_lf, x_hf)   # Eq.(2)
        return self.fuser(x, x_lf_p, x_hf_p)     # Eq.(3)(4)


# =====================================================================
# (2) F3MWithSA  ->  dat tai STEM (early)   [LAZY channel inference]
# =====================================================================
class F3MWithSA(nn.Module):
    """
    F3MWithSA = F3M + CBAM Spatial Attention. Bao toan chieu.
    Cau hinh theo bai bao: reduction_ratio=0.33, use_gate=True, sa_kernel_size=7.
    Thu tu (theo bai bao): F3M truoc -> Spatial Attention sau.

    Chu ky goi tu YAML:  F3MWithSA(reduction_ratio, use_gate, sa_kernel_size)
    """

    def __init__(self, reduction_ratio: float = 0.33, use_gate: bool = True,
                 sa_kernel_size: int = 7):
        super().__init__()
        self.f3m = F3M(reduction_ratio, use_gate)
        self.spatial_attention = SpatialAttentionModule(kernel_size=sa_kernel_size)

    def forward(self, x):
        y = self.f3m(x)                          # Eq.(1)-(4)
        return self.spatial_attention(y)         # Eq.(5)(6)


# =====================================================================
# Self-test
# =====================================================================
if __name__ == "__main__":
    def count(m):
        return sum(p.numel() for p in m.parameters())

    x = torch.randn(2, 256, 80, 80)
    neck = F3M(reduction_ratio=0.125, use_gate=False)        # NECK lite
    stem = F3MWithSA(reduction_ratio=0.33, use_gate=True)    # STEM full
    neck.eval(); stem.eval()
    with torch.no_grad():
        y_neck, y_stem = neck(x), stem(x)   # lazy build kich hoat o day

    print("Input            :", tuple(x.shape))
    print("F3M  (Neck)      :", tuple(y_neck.shape), "| params =", count(neck))
    print("F3MWithSA (Stem) :", tuple(y_stem.shape), "| params =", count(stem))
    assert y_neck.shape == x.shape and y_stem.shape == x.shape

    sep = FrequencySeparator()
    xlf, xhf = sep(x)
    assert torch.allclose(xlf + xhf, x, atol=1e-5)
    print("OK - lazy build, dimension-preserving, Eq.(1) X_lf + X_hf == X verified")