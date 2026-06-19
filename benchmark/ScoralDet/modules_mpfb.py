# benchmark/scoraldet/modules_mpfb.py
# Module (1): MPFB - Multi-Path Fusion Block (SCoralDet Fig 2, Eq 1).
# 5 nhanh song song -> cong element-wise -> SiLU. Reparameterize ve 1 conv NxN khi
# inference (Eq 8-9, = Diverse Branch Block cua Ding et al. 2021).
# Chu ky: MPFB(c1, c2, k=3). C2f_MPFB(c1, c2, n, shortcut, g, e) - thay 2 conv trong Bottleneck.

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules import C2f


def _fuse_conv_bn(conv_w, bn):
    """Fold BatchNorm vao conv weight/bias. Tra ve (w, b) tuong duong (khong BN).
    conv_w: (cout, cin, kh, kw). bn: nn.BatchNorm2d (da train, dung running stats)."""
    gamma = bn.weight
    beta = bn.bias
    mean = bn.running_mean
    var = bn.running_var
    eps = bn.eps
    std = (var + eps).sqrt()
    t = (gamma / std).reshape(-1, 1, 1, 1)
    w = conv_w * t
    b = beta - mean * gamma / std
    return w, b


def _pad_to_kxk(w, k):
    """Pad kernel bat ky (1x1, Nx1, 1xN, axb) ve kich thuoc k x k (chen 0 quanh tam)."""
    kh, kw = w.shape[-2:]
    ph, pw = (k - kh) // 2, (k - kw) // 2
    return F.pad(w, [pw, k - kw - pw, ph, k - kh - ph])


class _ConvBN(nn.Module):
    """Conv (khong bias) + BN. p mac dinh = 'same' cho kernel bat ky (ho tro asymmetric)."""

    def __init__(self, c1, c2, k, s=1, p=None, g=1):
        super().__init__()
        if isinstance(k, int):
            k = (k, k)
        if p is None:
            p = (k[0] // 2, k[1] // 2)
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)

    def forward(self, x):
        return self.bn(self.conv(x))


class MPFB(nn.Module):
    """Multi-Path Fusion Block (SCoralDet Fig 2).

    5 nhanh song song (Fig 2), moi nhanh -> BN rieng, cong element-wise -> SiLU:
        1) NxN          -> BN
        2) 1x1 -> BN    -> NxN -> BN
        3) Nx1          -> BN
        4) 1xN          -> BN
        5) 1x1 -> BN    -> AvgPool(NxN, s1) -> BN
    stride luon = 1 (MPFB nam trong Bottleneck, khong downsample).
    Sau switch_to_deploy(): chay 1 conv NxN duy nhat (reparam Eq 8-9).
    """

    def __init__(self, c1, c2, k=3):
        super().__init__()
        assert k % 2 == 1, "MPFB can kernel le de pad 'same'"
        self.c1, self.c2, self.k = c1, c2, k
        self.deploy = False

        # nhanh 1: NxN
        self.b_kxk = _ConvBN(c1, c2, k)
        # nhanh 2: 1x1 -> NxN
        self.b_1x1_kxk_a = _ConvBN(c1, c2, 1)
        self.b_1x1_kxk_b = _ConvBN(c2, c2, k)
        # nhanh 3: Nx1
        self.b_kx1 = _ConvBN(c1, c2, (k, 1))
        # nhanh 4: 1xN
        self.b_1xk = _ConvBN(c1, c2, (1, k))
        # nhanh 5: 1x1 -> AvgPool
        self.b_1x1_avg_conv = _ConvBN(c1, c2, 1)
        self.b_1x1_avg_pool = nn.AvgPool2d(k, stride=1, padding=k // 2)
        self.b_1x1_avg_bn = nn.BatchNorm2d(c2)

        self.act = nn.SiLU()
        self.reparam = None  # nn.Conv2d sau switch_to_deploy()

    def forward(self, x):
        if self.deploy and self.reparam is not None:
            return self.act(self.reparam(x))
        y = self.b_kxk(x)
        y = y + self.b_1x1_kxk_b(self.b_1x1_kxk_a(x))
        y = y + self.b_kx1(x)
        y = y + self.b_1xk(x)
        y = y + self.b_1x1_avg_bn(self.b_1x1_avg_pool(self.b_1x1_avg_conv(x)))
        return self.act(y)

    # ---- reparameterization (Eq 8-9): gop 5 nhanh ve 1 conv NxN ----
    def _equivalent_kernel_bias(self):
        k = self.k
        # nhanh 1: NxN
        w1, b1 = _fuse_conv_bn(self.b_kxk.conv.weight, self.b_kxk.bn)
        # nhanh 2: 1x1 -> NxN (fuse tuan tu 2 conv)
        wa, ba = _fuse_conv_bn(self.b_1x1_kxk_a.conv.weight, self.b_1x1_kxk_a.bn)
        wb, bb = _fuse_conv_bn(self.b_1x1_kxk_b.conv.weight, self.b_1x1_kxk_b.bn)
        # conv(1x1) roi conv(NxN): w = wb (*) wa theo kenh; b = wb @ ba (sum spatial) + bb
        w2 = torch.einsum("o m h w, m i -> o i h w", wb, wa.squeeze(-1).squeeze(-1))
        b2 = (wb.sum(dim=(2, 3)) * ba.reshape(1, -1)).sum(dim=1) + bb
        # nhanh 3: Nx1 -> pad ve NxN
        w3, b3 = _fuse_conv_bn(self.b_kx1.conv.weight, self.b_kx1.bn)
        w3 = _pad_to_kxk(w3, k)
        # nhanh 4: 1xN -> pad ve NxN
        w4, b4 = _fuse_conv_bn(self.b_1xk.conv.weight, self.b_1xk.bn)
        w4 = _pad_to_kxk(w4, k)
        # nhanh 5: 1x1 -> Avg(NxN). Fold BN(1x1) truoc, roi nhan trung binh 1/(k*k),
        # cuoi cung fold BN sau avg.
        wc, bc = _fuse_conv_bn(self.b_1x1_avg_conv.conv.weight, self.b_1x1_avg_conv.bn)
        # avg = depthwise conv NxN he so 1/(k*k); ket hop voi 1x1: w[o,i] = wc[o,i]/(k*k) tren toan NxN
        w5 = wc.repeat(1, 1, k, k) / (k * k)        # (cout,cin,1,1) -> (cout,cin,k,k)
        b5 = bc                                       # avg cua hang so = hang so
        # fold BN sau cua nhanh 5
        bn = self.b_1x1_avg_bn
        std = (bn.running_var + bn.eps).sqrt()
        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        w5 = w5 * t
        b5 = bn.bias + (b5 - bn.running_mean) * bn.weight / std
        w = w1 + w2 + w3 + w4 + w5
        b = b1 + b2 + b3 + b4 + b5
        return w, b

    @torch.no_grad()
    def switch_to_deploy(self):
        if self.deploy:
            return
        w, b = self._equivalent_kernel_bias()
        self.reparam = nn.Conv2d(self.c1, self.c2, self.k, 1, self.k // 2, bias=True)
        self.reparam.weight.copy_(w)
        self.reparam.bias.copy_(b)
        # xoa cac nhanh train de tiet kiem bo nho
        for name in ("b_kxk", "b_1x1_kxk_a", "b_1x1_kxk_b", "b_kx1", "b_1xk",
                     "b_1x1_avg_conv", "b_1x1_avg_pool", "b_1x1_avg_bn"):
            if hasattr(self, name):
                self.__delattr__(name)
        self.deploy = True


class _MPFBBottleneck(nn.Module):
    """Bottleneck cua C2f nhung 2 conv -> 2 MPFB (Fig 2). shortcut: cong residual neu c1==c2."""

    def __init__(self, c1, c2, shortcut=True, k=3, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = MPFB(c1, c_, k)
        self.cv2 = MPFB(c_, c2, k)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class C2f_MPFB(C2f):
    """C2f cua YOLOv10/v8 nhung Bottleneck dung MPFB (SCoralDet backbone).

    Chu ky giong C2f: (c1, c2, n=1, shortcut=False, g=1, e=0.5). cv1/cv2 (1x1 split/merge)
    giu nguyen Conv goc; chi thay 2 conv 3x3 ben trong tung Bottleneck bang MPFB.
    """

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            _MPFBBottleneck(self.c, self.c, shortcut, k=3, e=1.0) for _ in range(n)
        )

    def switch_to_deploy(self):
        for bottleneck in self.m:
            bottleneck.cv1.switch_to_deploy()
            bottleneck.cv2.switch_to_deploy()


if __name__ == "__main__":
    # smoke test: forward shape + train (==) deploy sau reparameterize (sai so < 1e-4).
    torch.manual_seed(0)
    m = MPFB(32, 64, k=3).eval()
    x = torch.randn(2, 32, 40, 40)
    y_train = m(x)
    assert y_train.shape == (2, 64, 40, 40), y_train.shape
    m.switch_to_deploy()
    y_deploy = m(x)
    err = (y_train - y_deploy).abs().max().item()
    assert err < 1e-4, f"reparam sai lech qua lon: {err}"
    # C2f_MPFB
    blk = C2f_MPFB(64, 128, n=2, shortcut=True).eval()
    yb = blk(torch.randn(2, 64, 80, 80))
    assert yb.shape == (2, 128, 80, 80), yb.shape
    print(f"OK MPFB reparam err={err:.2e}  C2f_MPFB out={tuple(yb.shape)}")