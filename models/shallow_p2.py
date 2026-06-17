# models/shallow_p2.py
# Module (1): Shallow-feature retention - FAITHFUL port SF-YOLO (2-head).
#   - AMCF (Fig 4): thay Conv@7, stride 1 => bo downsample cuoi (640->320->160->80->40).
#   - neck (cfg/base_yolov12n_shallow.yaml): top-down vuon toi stride 4, fuse layer 2 & 4
#     => 2 head P4/P5 o 80x80 + 40x40 (node P3 td idx 14 ton tai, KHONG gan head).

from pathlib import Path

import torch
import torch.nn as nn

from ultralytics.nn.modules import Conv

CFG_DIR = Path(__file__).resolve().parents[1] / "cfg"
BASE_YAML = CFG_DIR / "base_yolov12n.yaml"
SHALLOW_YAML = CFG_DIR / "base_yolov12n_shallow.yaml"

# Diem cam (PHAI khop ca 2 YAML) - co assert trong validate_yaml().
SFDF_REPLACE_IDXS = (2, 4, 6, 8)   # 4 khoi dac trung backbone (module 2)
AMCF_REPLACE_IDX = 7               # Conv downsample P4->P5 (module 1)
FGA2_REPLACE_IDXS = (11, 14)       # 2 A2C2f top-down trong neck (module 5)


# ============================ AMCF (Fig 4) ============================
class _PinwheelPConv(nn.Module):
    """PConv pinwheel cua AMCF: 4 nhanh CBS bat doi xung (1x3 / 3x1) -> CBS(3x3, stride).

    Cac nhanh bat doi xung bat ket cau dinh huong. stride truyen tu build_model qua specs;
    faithful SF-YOLO dung stride=1 => AMCF KHONG downsample.
    """

    def __init__(self, c1, c2, stride=2):
        super().__init__()
        c_ = max(c2 // 4, 8)
        self.b1 = Conv(c1, c_, (1, 3), 1)
        self.b2 = Conv(c1, c_, (3, 1), 1)
        self.b3 = Conv(c1, c_, (1, 3), 1)
        self.b4 = Conv(c1, c_, (3, 1), 1)
        self.fuse = Conv(4 * c_, c2, 3, stride)   # stride tu specs: 1 = giu nguyen size (faithful), 2 = downsample

    def forward(self, x):
        y = torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)
        return self.fuse(y)


class _StarBlock(nn.Module):
    """Star_Block (Fig 4): DWConv7 -> Conv * ReLU6(Conv) -> Conv -> DWConv7 + residual."""

    def __init__(self, c, mlp_ratio=3):
        super().__init__()
        ch = c * mlp_ratio
        self.dw1 = Conv(c, c, 7, 1, g=c)
        self.f1 = nn.Conv2d(c, ch, 1)
        self.f2 = nn.Conv2d(c, ch, 1)
        self.act = nn.ReLU6(inplace=True)
        self.g = nn.Conv2d(ch, c, 1)
        self.dw2 = Conv(c, c, 7, 1, g=c)

    def forward(self, x):
        identity = x
        x = self.dw1(x)
        x = self.f1(x) * self.act(self.f2(x))   # phep "star" (nhan element-wise)
        x = self.g(x)
        x = self.dw2(x)
        return identity + x


class AMCF(nn.Module):
    """Adaptive Multi-scale Cross Fusion (SF-YOLO Fig 4).

    Thay Conv@7 trong backbone: PConv pinwheel(3x3) -> DWConv5x5 -> Conv(1x1) -> Star_Block.
    Chu ky: AMCF(c1, c2, stride). faithful SF-YOLO dung stride=1 (KHONG downsample);
    c2 = kenh ra (giu nguyen so kenh cua Conv goc bi thay).
    """

    def __init__(self, c1, c2, stride=2):
        super().__init__()
        self.pconv = _PinwheelPConv(c1, c2, stride)
        self.dw = Conv(c2, c2, 5, 1, g=c2)   # depthwise 5x5 (Eq 12 paper: mid-range deps)
        self.cv = Conv(c2, c2, 1, 1)         # pointwise 1x1 => depthwise-SEPARABLE 5x5
        self.star = _StarBlock(c2)

    def forward(self, x):
        x = self.pconv(x)             # F_out (stride tu specs; faithful stride=1 => giu size)
        x = self.cv(self.dw(x))       # DWConv5x5 -> Conv(1x1) = depthwise-sep (F_i)
        return self.star(x)           # Star_Block = F_o


# ============================ YAML helpers ============================
def select_base_yaml(modules) -> Path:
    """Module 1 ON => dung YAML co shallow-fusion. AMCF van duoc build_model va vao idx 7."""
    return SHALLOW_YAML if 1 in set(int(m) for m in modules) else BASE_YAML


def validate_yaml(d: dict, with_shallow: bool) -> None:
    """Assert cac diem cam khong bi xe dich sau khi load YAML."""
    layers = d["backbone"] + d["head"]
    for i in SFDF_REPLACE_IDXS:
        assert layers[i][2] in ("C3k2", "A2C2f"), (
            f"Layer {i} phai la C3k2/A2C2f (diem cam SFDF), thay {layers[i][2]}"
        )
    for i in FGA2_REPLACE_IDXS:
        assert layers[i][2] == "A2C2f", (
            f"Layer {i} phai la A2C2f (diem cam FGA2), thay {layers[i][2]}"
        )
    if with_shallow:
        assert layers[AMCF_REPLACE_IDX][2] == "Conv", (
            f"Layer {AMCF_REPLACE_IDX} phai la Conv (diem cam AMCF), thay {layers[AMCF_REPLACE_IDX][2]}"
        )
    expected = 2 if with_shallow else 3
    n_det = len(layers[-1][0])
    assert n_det == expected, (
        f"Detect can {expected} nhanh ("
        f"{'P4/P5 - shallow 2-head' if with_shallow else 'P3/P4/P5'}), thay {n_det}"
    )