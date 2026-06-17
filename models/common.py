# models/common.py
# Khoi dung chung cho SFDF (2) va FGA2 (5).
# Shared building blocks: fixed-filter Haar DWT/IDWT, pinwheel conv, SE attention.

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules import Conv  # CBS = Conv2d + BN + SiLU


class HaarDWT(nn.Module):
    """Haar DWT 1 cap. (B,C,H,W) -> (LL, LH, HL, HH), moi tensor (B,C,H/2,W/2).

    Bo loc Haar truc chuan (he so +-0.5) cai dat bang conv stride-2 voi
    trong so co dinh (register_buffer => khong hoc, khong xuat hien trong optimizer).
    """

    def __init__(self):
        super().__init__()
        ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        lh = torch.tensor([[0.5, 0.5], [-0.5, -0.5]])   # bien doc (vertical detail)
        hl = torch.tensor([[0.5, -0.5], [0.5, -0.5]])   # bien ngang (horizontal detail)
        hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]])   # cheo (diagonal)
        # (4, 1, 2, 2)
        self.register_buffer("filt", torch.stack([ll, lh, hl, hh]).unsqueeze(1))

    def forward(self, x):
        b, c, h, w = x.shape
        if (h % 2) or (w % 2):  # pad chan de chia 2 (hiem khi xay ra voi imgsz 640)
            x = F.pad(x, (0, w % 2, 0, h % 2), mode="replicate")
        f = self.filt.repeat(c, 1, 1, 1)                  # (4c,1,2,2)
        y = F.conv2d(x, f, stride=2, groups=c)            # (b,4c,h/2,w/2)
        y = y.view(b, c, 4, y.shape[-2], y.shape[-1])
        return y[:, :, 0], y[:, :, 1], y[:, :, 2], y[:, :, 3]


class HaarIDWT(nn.Module):
    """Nghich dao chinh xac cua HaarDWT (bo loc truc chuan => conv_transpose)."""

    def __init__(self):
        super().__init__()
        ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        lh = torch.tensor([[0.5, 0.5], [-0.5, -0.5]])
        hl = torch.tensor([[0.5, -0.5], [0.5, -0.5]])
        hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]])
        self.register_buffer("filt", torch.stack([ll, lh, hl, hh]).unsqueeze(1))

    def forward(self, ll, lh, hl, hh):
        b, c, h, w = ll.shape
        y = torch.stack([ll, lh, hl, hh], dim=2).view(b, 4 * c, h, w)
        f = self.filt.repeat(c, 1, 1, 1)                  # (4c,1,2,2)
        return F.conv_transpose2d(y, f, stride=2, groups=c)


class PinwheelConv(nn.Module):
    """Pinwheel-shaped conv (PConv paper, Eq 1-3): 4 nhanh kernel bat doi xung
    (2 ngang 1xk + 2 doc kx1, trong so rieng biet W1..W4) + fuse 2x2.

    Padding bat doi xung 2-PHIA dung theo paper (left,right,top,bottom):
      cv_l (1xk): (k,0,1,0)   cv_r (1xk): (0,k,0,1)
      cv_t (kx1): (0,1,k,0)   cv_b (kx1): (1,0,0,k)
    Moi nhanh -> (h+1, w+1); fuse 2x2 valid -> (h, w) => GIU NGUYEN H,W input.
    Dung tren tung subband DWT trong SFDF de bat ket cau dinh huong (canh/tua san ho).
    """

    def __init__(self, c1, c2, k=3):
        super().__init__()
        c_ = max(c2 // 4, 8)
        # 4 nhanh trong so rieng biet (W1..W4 trong Eq 1): 2 ngang (1xk) + 2 doc (kx1)
        self.cv_l = nn.Conv2d(c1, c_, (1, k), padding=0, bias=False)
        self.cv_r = nn.Conv2d(c1, c_, (1, k), padding=0, bias=False)
        self.cv_t = nn.Conv2d(c1, c_, (k, 1), padding=0, bias=False)
        self.cv_b = nn.Conv2d(c1, c_, (k, 1), padding=0, bias=False)
        # padding = (left, right, top, bottom) - bat doi xung dung theo PConv paper (Eq 1)
        self.pad_l = (k, 0, 1, 0)
        self.pad_r = (0, k, 0, 1)
        self.pad_t = (0, 1, k, 0)
        self.pad_b = (1, 0, 0, k)
        self.fuse = Conv(4 * c_, c2, 2, 1, p=0)   # PConv fuse 2x2 (Eq 3): valid (h+1,w+1)->(h,w)

    def forward(self, x):
        y = torch.cat(
            [
                self.cv_l(F.pad(x, self.pad_l)),
                self.cv_r(F.pad(x, self.pad_r)),
                self.cv_t(F.pad(x, self.pad_t)),
                self.cv_b(F.pad(x, self.pad_b)),
            ],
            dim=1,
        )                                          # 4 nhanh deu (h+1, w+1)
        return self.fuse(y)                        # 2x2 valid -> (h, w): giu nguyen H,W input


class ChannelAttention(nn.Module):
    """Channel attention nhanh spatial cua SFDF - FAITHFUL Eq 6:
        X's = sigma(GAP(SiLU(BN(DWConv(Xs1))))).
    Thu tu: DWConv 3x3 (depthwise, giu C) -> BN -> SiLU -> GAP (gom khong gian)
    -> sigmoid => vector trong so kenh (B,C,1,1). KHONG dung bottleneck SE.

    gate=True  : tra ve dac trung da gate (x * w).
    gate=False : tra ve TRONG SO chu y tran w (B,C,1,1) - dung cho cross-interaction
                 SFDF (Eq 11) de tranh tich feature x feature gay NaN.
    (reduction giu lai cho tuong thich chu ky goi; khong con dung sau khi bo bottleneck.)
    """

    def __init__(self, c, reduction=8, gate=True):
        super().__init__()
        self.gate = gate
        self.dw = Conv(c, c, 3, 1, g=c)       # DWConv -> BN -> SiLU (Eq 6)
        self.pool = nn.AdaptiveAvgPool2d(1)    # GAP

    def forward(self, x):
        w = torch.sigmoid(self.pool(self.dw(x)))   # (B,C,1,1) - trong so kenh tran
        return x * w if self.gate else w


class DSA(nn.Module):
    """Dynamic Spatial Attention (SF-YOLO Fig 3, nhanh tan so cua SFDF).

    Sinh KERNEL DONG tu mo ta kenh (GAP -> Conv -> ReLU -> Conv) roi tich chap
    len ban do trung binh kenh (Mean across channels) de tao spatial attention
    rieng cho tung anh trong batch (per-sample dynamic conv qua groups=B).
    """

    def __init__(self, c, k=3, gate=True):
        super().__init__()
        self.k = k
        self.gate = gate
        c_ = max(c // 4, 8)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.kgen = nn.Sequential(
            nn.Conv2d(c, c_, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_, k * k, 1),
        )

    def forward(self, x):
        b, c, h, w = x.shape
        kern = self.kgen(self.gap(x))             # (b, k*k, 1, 1) - kernel dong
        kern = kern.reshape(b, 1, self.k, self.k)
        m = x.mean(dim=1, keepdim=True)           # (b, 1, h, w) - mean across channels
        m = m.reshape(1, b, h, w)                 # gom batch vao kenh
        a = F.conv2d(m, kern, padding=self.k // 2, groups=b)  # depthwise per-sample
        a = torch.sigmoid(a.reshape(b, 1, h, w))  # (b,1,h,w) - ban do khong gian tran
        return x * a if self.gate else a