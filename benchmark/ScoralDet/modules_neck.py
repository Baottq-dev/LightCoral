# benchmark/scoraldet/modules_neck.py
# Module (2): GSConv + VoV-GSCSP (slim-neck, Li et al. 2022) - SCoralDet Fig 3.
# Vendored de doc lap phien ban Ultralytics. Chu ky:
#   GSConv(c1, c2, k=1, s=1)         -> doi kenh (c2 = args[1]); ho tro stride 2 (downsample neck)
#   VoVGSCSP(c1, c2, n=1)            -> CSP block nhe (c2 = args[1])

import torch
import torch.nn as nn

from ultralytics.nn.modules import Conv


class GSConv(nn.Module):
    """GSConv (Fig 3 trai): SC(C2/2) -> [SC_out, DSC(SC_out)] concat -> channel shuffle.

    Ket hop standard conv (tron kenh) + depthwise-separable conv (nhe). Channel shuffle
    tron 2 nua de tang da dang dac trung. Ho tro stride > 1 (thay Conv downsample neck).
    """

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)        # standard conv (co the stride 2)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)        # depthwise conv 5x5

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = torch.cat((x1, self.cv2(x1)), dim=1)              # (b, c2, h, w)
        # channel shuffle: tron 2 nua kenh
        b, c, h, w = x2.shape
        x2 = x2.view(b, 2, c // 2, h, w)
        x2 = x2.transpose(1, 2).contiguous()
        return x2.view(b, c, h, w)


class _GSBottleneck(nn.Module):
    """GS bottleneck = GSConv(1x1) -> GSConv(3x3, no act) + shortcut."""

    def __init__(self, c1, c2, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.conv = nn.Sequential(
            GSConv(c1, c_, 1, 1),
            GSConv(c_, c2, 3, 1, act=False),
        )
        self.shortcut = Conv(c1, c2, 1, 1, act=False) if c1 != c2 else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class VoVGSCSP(nn.Module):
    """VoV-GSCSP (Fig 3 phai): nhanh SC || (SC -> n x GS-bottleneck) -> Concat -> SC.

    Mo rong GSConv kieu CSP: giam FLOPs giu nang luc hoc. Chu ky (c1, c2, n=1).
    """

    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)                                   # nhanh GS
        self.cv2 = Conv(c1, c_, 1, 1)                                   # nhanh SC (shortcut CSP)
        self.gsb = nn.Sequential(*(_GSBottleneck(c_, c_, e=1.0) for _ in range(max(1, n))))
        self.cv3 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        return self.cv3(torch.cat((self.gsb(self.cv1(x)), self.cv2(x)), dim=1))


if __name__ == "__main__":
    # smoke test: shape + downsample.
    x = torch.randn(2, 64, 80, 80)
    assert GSConv(64, 128, 1, 1)(x).shape == (2, 128, 80, 80)
    assert GSConv(64, 128, 3, 2)(x).shape == (2, 128, 40, 40), "GSConv downsample loi"
    assert VoVGSCSP(64, 128, n=2)(x).shape == (2, 128, 80, 80)
    print("OK GSConv + VoVGSCSP")