# benchmark/f3m/build_f3m.py
# Build F3M-YOLO11n: va F3MWithSA(idx1) + F3M(idx10) vao YAML voi kenh scale tuong minh.
# TAI DUNG models/registry.py: them module F3M vao CUSTOM_MODULES roi register_custom_modules()
# (dang ky lop + va parse_model doc kenh ra tu args[1]). Co smoke-test.

import math
import sys
from pathlib import Path

# chay truc tiep: dam bao root repo (sc-yolo12/) trong sys.path
_ROOT_INIT = Path(__file__).resolve().parents[2]
if str(_ROOT_INIT) not in sys.path:
    sys.path.insert(0, str(_ROOT_INIT))

import yaml

from models import registry as reg

# ho tro chay ca '-m benchmark.f3m.build_f3m' lan 'python benchmark/f3m/build_f3m.py'
try:
    from benchmark.f3m.modules_f3m import F3M, F3MWithSA
except ImportError:
    from modules_f3m import F3M, F3MWithSA

ROOT = Path(__file__).resolve().parents[2]              # .../sc-yolo12/
HERE = Path(__file__).resolve().parent
BASE_YAML = HERE / "base_f3m_yolo11n.yaml"

# Diem cam F3M (PHAI khop base_f3m_yolo11n.yaml): idx -> (r, gate, with_sa)
F3M_NODES = {
    1: dict(r=0.33, gate=True, sa=True),     # stem  (F3MWithSA, 16 kenh)
    10: dict(r=0.125, gate=False, sa=False),  # deep  (F3M, 256 kenh, pre-SPPF)
}

# module built-in duoc parse_model tu scale tu args[0]
BUILTIN_SCALED = {"Conv", "C3k2", "SPPF", "C2PSA"}

# module F3M can dang ky (parse_model doc kenh ra tu args[1])
F3M_MODULES = {"F3M": F3M, "F3MWithSA": F3MWithSA}


def register_f3m_modules():
    """Them module F3M vao registry.CUSTOM_MODULES roi register.
    Phai goi TRUOC khi patch parse_model: patch dong bang frozenset(CUSTOM_MODULES) luc chay."""
    reg.CUSTOM_MODULES.update(F3M_MODULES)
    reg.register_custom_modules()


def _make_divisible(x, divisor=8):
    return math.ceil(x / divisor) * divisor


def _scaled(c, width, max_ch):
    return _make_divisible(min(c, max_ch) * width, 8)


def _compute(layers, depth, width, max_ch):
    """Mo phong parse_model lan truyen kenh tren YAML DA LOAD (truoc khi va).
    F3M/F3MWithSA giu nguyen kenh (passthrough = kenh dau vao)."""
    out, reps = [], []
    for i, (f, n, m, args) in enumerate(layers):
        f0 = f[0] if isinstance(f, list) else f
        n_scaled = max(round(n * depth), 1) if n > 1 else n
        if m in BUILTIN_SCALED:
            c2 = _scaled(args[0], width, max_ch)
        elif m in ("F3M", "F3MWithSA"):
            c2 = out[f0]                       # passthrough: giu nguyen kenh
        elif m == "Concat":
            c2 = sum(out[x] for x in f)
        elif m == "nn.Upsample":
            c2 = out[f0]
        elif m in ("Detect", "v10Detect"):
            c2 = out[f0]
        else:
            raise ValueError(f"Chua ho tro tinh kenh cho module {m}")
        out.append(c2)
        reps.append(n_scaled)
    return out, reps


def _validate(layers):
    assert layers[1][2] == "F3MWithSA", f"Layer 1 phai la F3MWithSA (stem), thay {layers[1][2]}"
    assert layers[10][2] == "F3M", f"Layer 10 phai la F3M (pre-SPPF), thay {layers[10][2]}"
    assert layers[11][2] == "SPPF", f"Layer 11 phai la SPPF, thay {layers[11][2]}"
    assert layers[-1][2] in ("Detect", "v10Detect"), "Layer cuoi phai la Detect/v10Detect"
    n_det = len(layers[-1][0])
    assert n_det == 3, f"F3M-YOLO11n can 3 head (P3/P4/P5), thay {n_det}"


def build_f3m_yaml(nc=6, save_dir=ROOT / "runs" / "cfg"):
    """Va 2 node F3M vao base_f3m_yolo11n.yaml; tra ve duong dan YAML da sinh."""
    d = yaml.safe_load(BASE_YAML.read_text(encoding="utf-8"))
    d["nc"] = nc

    depth, width, max_ch = d["scales"]["n"]
    nb = len(d["backbone"])
    layers = [list(x) for x in d["backbone"] + d["head"]]
    _validate(layers)
    ch, _ = _compute(layers, depth, width, max_ch)

    # F3M from = -1, giu kenh => c1 = c2 = ch[i-1]. Ghi tuong minh [c1, c1, r, gate].
    for i, cfg in F3M_NODES.items():
        c1 = ch[i - 1]
        layers[i] = [layers[i][0], 1, layers[i][2], [c1, c1, cfg["r"], cfg["gate"]]]

    d["backbone"], d["head"] = layers[:nb], layers[nb:]

    # ten chua 'yolo11n' de ultralytics nhan scale 'n'
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    out = save_dir / "f3m_yolo11n.yaml"
    out.write_text(yaml.safe_dump(d, sort_keys=False, default_flow_style=None), encoding="utf-8")
    return out


def build_f3m_model(nc=6, verbose=True, **kw):
    """register module + sinh YAML + tra ve YOLO chua train."""
    from ultralytics import YOLO
    from models.registry import is_channel_patch_applied
    register_f3m_modules()
    assert is_channel_patch_applied(), \
        "parse_model chua duoc va kenh -> F3M co the dung sai c1 (pin dung ban ultralytics)"
    yaml_path = build_f3m_yaml(nc=nc, **kw)
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
    # smoke test: build + forward 1 tensor 640, kiem NaN + 3 head + stride [8,16,32]; in params.
    import torch

    m = build_f3m_model(verbose=False)
    det = m.model.model[-1]
    m.model.eval()
    with torch.no_grad():
        y = m.model(torch.randn(1, 3, 640, 640))
    tensors = list(_iter_tensors(y))
    assert tensors, "khong tim thay tensor output"
    for t in tensors:
        assert torch.isfinite(t).all(), "NaN/Inf trong output F3M"
    assert det.nl == 3, f"F3M-YOLO11n can 3 head, thay {det.nl}"
    assert [int(s) for s in det.stride] == [8, 16, 32], \
        f"stride {[int(s) for s in det.stride]} != [8, 16, 32]"
    n_par = sum(p.numel() for p in m.model.parameters())
    print(f"OK F3M-YOLO11n  nl={det.nl}  stride={[int(s) for s in det.stride]}  "
          f"params={n_par/1e6:.3f}M (ky vong ~2.61M)")