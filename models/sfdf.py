# models/sfdf.py
# Module (2): Spatial-Frequency Dual-domain Fusion - FAITHFUL port SF-YOLO (Fig 3).
# Thay 4 khoi dac trung backbone (idx 2/4/6/8). Chu ky: SFDF(c1, c2, reduction, k, swap).

import torch
import torch.nn as nn

from ultralytics.nn.modules import Conv

from .common import ChannelAttention, DSA, HaarDWT, HaarIDWT, PinwheelConv


class SFDF(nn.Module):
    """Spatial-Frequency Dual-domain Fusion (SF-YOLO Fig 3).

    Args:
        c1: kenh vao (da scale tuong minh boi build_model)
        c2: kenh ra
        reduction: he so giam channel-attention (nhanh spatial)
        k: kernel pinwheel conv (nhanh tan so)
        swap: True => spatial nhan 3C/4, frequency nhan C/4
              (paper dao ti le o 2 khoi sau de uu tien dac trung khong gian)

    Forward (Fig 3):
        Split  : Xs (C/4), Xf (3C/4)                  # swap => doi cho
        Spatial: Xs -> CBS3 -> CBS3 -> CBS1 = Xs1 -> ChannelAttention = X's (B,C,1,1)
        Freq   : Xf -> DWT -> PConv -> IWT -> CBS1 = Xf1 -> DSA = X'f (B,1,H,W)
        Cross  : out = X's * Xf1 + X'f * Xs1           # trong so tran GATE dac trung cheo roi CONG
    """

    def __init__(self, c1, c2, reduction=8, k=3, swap=False):
        super().__init__()
        if swap:
            cs, cf = c1 - c1 // 4, c1 // 4      # spatial = 3C/4
        else:
            cs, cf = c1 // 4, c1 - c1 // 4      # spatial = C/4 (mac dinh paper)
        self.cs, self.cf = cs, cf

        # ---- nhanh spatial: CBS(3x3) x2 -> CBS(1x1) -> channel attention ----
        self.s1 = Conv(cs, c2, 3, 1)
        self.s2 = Conv(c2, c2, 3, 1)
        self.s3 = Conv(c2, c2, 1, 1)
        self.s_att = ChannelAttention(c2, reduction, gate=False)   # tra ve trong so kenh tran (B,C,1,1)

        # ---- nhanh frequency: DWT -> 1 PConv (tren 4 subband ghep) -> IWT -> CBS -> DSA ----
        self.dwt = HaarDWT()
        self.pconv = PinwheelConv(4 * cf, 4 * cf, k)
        self.idwt = HaarIDWT()
        self.f_cbs = Conv(cf, c2, 1, 1)
        self.dsa = DSA(c2, gate=False)                             # tra ve ban do khong gian tran (B,1,H,W)

        self.out = Conv(c2, c2, 1, 1)

    def forward(self, x):
        xs, xf = torch.split(x, [self.cs, self.cf], dim=1)

        # nhanh spatial
        xs1 = self.s3(self.s2(self.s1(xs)))            # Xs1 (dac trung HxWxC)
        x_s = self.s_att(xs1)                          # X's = trong so kenh tran (B,C,1,1)

        # nhanh frequency: 1 PConv chung tren 4 subband ghep theo kenh
        ll, lh, hl, hh = self.dwt(xf)
        y = self.pconv(torch.cat([ll, lh, hl, hh], dim=1))
        ll, lh, hl, hh = torch.split(y, self.cf, dim=1)
        rec = self.idwt(ll, lh, hl, hh)
        rec = rec[:, :, : xf.shape[-2], : xf.shape[-1]]  # crop ve size goc: DWT pad le -> IDWT du 1px (rect-val)
        xf1 = self.f_cbs(rec)                          # Xf1 (dac trung HxWxC)
        x_f = self.dsa(xf1)                            # X'f = ban do khong gian tran (B,1,H,W)

        # cross-interaction (Eq 11): trong so tran GATE dac trung CHEO roi CONG.
        #   out = w_c (.) Xf1 + w_s (.) Xs1  -> bi chan boi sigmoid, KHONG con tich Xs1*Xf1 (nguon NaN cu).
        out = x_s * xf1 + x_f * xs1
        return self.out(out)