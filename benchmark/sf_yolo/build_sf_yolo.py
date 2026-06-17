# benchmark/sf_yolo/build_sf_yolo.py
# Build SF-YOLO faithful (base YOLOv11n): va SFDF x4 + AMCF vao YAML shallow.
# TAI DUNG lop SFDF/AMCF va register_custom_modules() cua repo. Khac build_model.py:
#   - base = v11n (C3k2 + SPPF + C2PSA) thay vi v12n,
#   - KHONG co FGA2/PG-DAM (SF-YOLO nguyen ban khong dung 2 module rieng cua nhom).

import math
import sys
from pathlib import Path

# chay truc tiep: dam bao root repo (sc-yolo12/) trong sys.path
_ROOT_INIT = Path(__file__).resolve().parents[2]
if str(_ROOT_INIT) not in sys.path:
    sys.path.insert(0, str(_ROOT_INIT))

import yaml

from models.registry import register_custom_modules

ROOT = Path(__file__).resolve().parents[2]              # .../sc-yolo12/
HERE = Path(__file__).resolve().parent
BASE_YAML = HERE / "base_yolo11n_shallow.yaml"

# Diem cam (PHAI khop base_yolo11n_shallow.yaml)
SFDF_REPLACE_IDXS = (2, 4, 6, 8)
SFDF_SWAP_IDXS = (6, 8)            # 2 khoi sau dao ti le split (spatial = 3C/4)
AMCF_REPLACE_IDX = 7

# parse_model tu scale args[0]; them C2PSA so voi build_model.py (base v12n khong co)
BUILTIN_SCALED = {"Conv", "C3k2", "A2C2f", "C2f", "SPPF", "C2PSA"}


def _make_divisible(x, divisor=8):
    return math.ceil(x / divisor) * divisor


def _scaled(c, width, max_ch):
    return _make_divisible(min(c, max_ch) * width, 8)


def _compute_channels(layers, width, max_ch):
    """Mo phong parse_model lan truyen kenh tren YAML DA LOAD (truoc khi va).
    Tra ve out[i] = kenh ra cua layer i. Moi layer la built-in/Concat/Upsample/Detect."""
    out = []
    for i, (f, n, m, args) in enumerate(layers):
        f0 = f[0] if isinstance(f, list) else f
        if m in BUILTIN_SCALED:
            c2 = _scaled(args[0], width, max_ch)
        elif m == "Concat":
            c2 = sum(out[x] for x in f)         # index am cung dung
        elif m == "nn.Upsample":
            c2 = out[f0]
        elif m == "Detect":
            c2 = out[f0]
        else:
            raise ValueError(f"Chua ho tro tinh kenh cho module {m}")
        out.append(c2)
    return out


def _validate(layers):
    for i in SFDF_REPLACE_IDXS:
        assert layers[i][2] in ("C3k2", "A2C2f"), \
            f"Layer {i} phai la C3k2/A2C2f (diem cam SFDF), thay {layers[i][2]}"
    assert layers[AMCF_REPLACE_IDX][2] == "Conv", \
        f"Layer {AMCF_REPLACE_IDX} phai la Conv (diem cam AMCF), thay {layers[AMCF_REPLACE_IDX][2]}"
    n_det = len(layers[-1][0])
    assert n_det == 2, f"SF-YOLO faithful can 2 head (P4/P5), thay {n_det}"


def build_sf_yolo_yaml(nc=6, reduction=8, pconv_kernel=3, amcf_stride=1,
                       save_dir=ROOT / "runs" / "cfg"):
    """Va SFDF x4 + AMCF vao base_yolo11n_shallow.yaml; tra ve duong dan YAML da sinh."""
    d = yaml.safe_load(BASE_YAML.read_text(encoding="utf-8"))
    d["nc"] = nc

    depth, width, max_ch = d["scales"]["n"]
    nb = len(d["backbone"])
    layers = [list(x) for x in d["backbone"] + d["head"]]
    _validate(layers)
    ch = _compute_channels(layers, width, max_ch)

    # ---- SFDF: thay 4 khoi dac trung backbone (idx 2,4,6,8) ----
    for i in SFDF_REPLACE_IDXS:
        c1, c2 = ch[i - 1], ch[i]
        layers[i] = [-1, 1, "SFDF", [c1, c2, reduction, pconv_kernel, i in SFDF_SWAP_IDXS]]

    # ---- AMCF: thay downsample cuoi (idx 7), stride 1 => KHONG downsample ----
    i = AMCF_REPLACE_IDX
    layers[i] = [-1, 1, "AMCF", [ch[i - 1], ch[i], amcf_stride]]

    d["backbone"], d["head"] = layers[:nb], layers[nb:]

    # ten chua 'yolo11n' de ultralytics nhan scale 'n'
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    out = save_dir / "sf_yolo11n.yaml"
    out.write_text(yaml.safe_dump(d, sort_keys=False, default_flow_style=None), encoding="utf-8")
    return out


def build_sf_yolo_model(nc=6, verbose=True, **kw):
    """register module + sinh YAML + tra ve YOLO chua train (tien cho smoke test)."""
    from ultralytics import YOLO
    from models.registry import is_channel_patch_applied
    register_custom_modules()
    assert is_channel_patch_applied(), \
        "parse_model chua duoc va kenh -> SFDF/AMCF co the dung sai c1 (pin dung ban ultralytics)"
    yaml_path = build_sf_yolo_yaml(nc=nc, **kw)
    return YOLO(str(yaml_path), verbose=verbose)


def _iter_tensors(o):
    import torch
    if isinstance(o, torch.Tensor):
        yield o
    elif isinstance(o, dict):
        for v in o.values():
            yield from _iter_tensors(v)
    elif isinstance(o, (list, tuple)):
        for v in o:
            yield from _iter_tensors(v)


if __name__ == "__main__":
    # smoke test: build + forward 1 tensor 640, kiem NaN + 2 head + stride [8,16].
    import torch

    m = build_sf_yolo_model(verbose=False)
    det = m.model.model[-1]
    m.model.eval()
    with torch.no_grad():
        y = m.model(torch.randn(1, 3, 640, 640))
    tensors = list(_iter_tensors(y))
    assert tensors, "khong tim thay tensor output"
    for t in tensors:
        assert torch.isfinite(t).all(), "NaN/Inf trong output SF-YOLO"
    assert det.nl == 2, f"SF-YOLO can 2 head, thay {det.nl}"
    assert [int(s) for s in det.stride] == [8, 16], \
        f"stride {[int(s) for s in det.stride]} != [8, 16]"
    n_par = sum(p.numel() for p in m.model.parameters())
    print(f"OK SF-YOLO faithful  nl={det.nl}  stride={[int(s) for s in det.stride]}  params={n_par/1e6:.2f}M")