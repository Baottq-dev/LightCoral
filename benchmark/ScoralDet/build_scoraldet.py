# benchmark/scoraldet/build_scoraldet.py
# Build SCoralDet (base YOLOv10n): va C2f_MPFB + VoVGSCSP + GSConv vao YAML voi kenh scale tuong minh.
# TAI DUNG models/registry.py: them module SCoralDet vao CUSTOM_MODULES roi register_custom_modules()
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

# ho tro chay ca '-m benchmark.scoraldet.build_scoraldet' lan 'python benchmark/scoraldet/build_scoraldet.py'
try:
    from benchmark.scoraldet.modules_mpfb import C2f_MPFB
    from benchmark.scoraldet.modules_neck import GSConv, VoVGSCSP
except ImportError:
    from modules_mpfb import C2f_MPFB
    from modules_neck import GSConv, VoVGSCSP

ROOT = Path(__file__).resolve().parents[2]              # .../sc-yolo12/
HERE = Path(__file__).resolve().parent
BASE_YAML = HERE / "base_scoraldet.yaml"

# Diem cam (PHAI khop base_scoraldet.yaml)
MPFB_REPLACE_IDXS = (2, 4, 6)            # C2f -> C2f_MPFB (backbone)
NECK_VOVGSCSP_IDXS = (13, 16, 19, 22)    # C2f/C2fCIB -> VoVGSCSP (neck)
NECK_GSCONV_IDXS = (17, 20)              # Conv/SCDown (stride 2) -> GSConv

# module built-in duoc parse_model tu scale tu args[0]
BUILTIN_SCALED = {"Conv", "C2f", "SCDown", "SPPF", "PSA", "C2fCIB"}

# module SCoralDet can dang ky (parse_model doc kenh ra tu args[1])
SCORALDET_MODULES = {"C2f_MPFB": C2f_MPFB, "GSConv": GSConv, "VoVGSCSP": VoVGSCSP}


def register_scoraldet_modules():
    """Them module SCoralDet vao registry.CUSTOM_MODULES roi register.
    Phai goi TRUOC khi patch parse_model: patch dong bang frozenset(CUSTOM_MODULES) luc chay."""
    reg.CUSTOM_MODULES.update(SCORALDET_MODULES)
    reg.register_custom_modules()


def _make_divisible(x, divisor=8):
    return math.ceil(x / divisor) * divisor


def _scaled(c, width, max_ch):
    return _make_divisible(min(c, max_ch) * width, 8)


def _compute(layers, depth, width, max_ch):
    """Mo phong parse_model lan truyen kenh + scale repeats tren YAML DA LOAD (truoc khi va).
    Tra ve (out[i]=kenh ra layer i, reps[i]=so repeat da scale theo depth)."""
    out, reps = [], []
    for i, (f, n, m, args) in enumerate(layers):
        f0 = f[0] if isinstance(f, list) else f
        n_scaled = max(round(n * depth), 1) if n > 1 else n
        if m in BUILTIN_SCALED:
            c2 = _scaled(args[0], width, max_ch)
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
    for i in MPFB_REPLACE_IDXS:
        assert layers[i][2] == "C2f", f"Layer {i} phai la C2f (diem cam MPFB), thay {layers[i][2]}"
    for i in NECK_VOVGSCSP_IDXS:
        assert layers[i][2] in ("C2f", "C2fCIB"), \
            f"Layer {i} phai la C2f/C2fCIB (diem cam VoVGSCSP), thay {layers[i][2]}"
    for i in NECK_GSCONV_IDXS:
        assert layers[i][2] in ("Conv", "SCDown"), \
            f"Layer {i} phai la Conv/SCDown (diem cam GSConv), thay {layers[i][2]}"
    assert layers[-1][2] in ("Detect", "v10Detect"), "Layer cuoi phai la Detect/v10Detect"
    n_det = len(layers[-1][0])
    assert n_det == 3, f"SCoralDet can 3 head (P3/P4/P5), thay {n_det}"


def build_scoraldet_yaml(nc=6, mpfb_kernel=3, save_dir=ROOT / "runs" / "cfg"):
    """Va C2f_MPFB + VoVGSCSP + GSConv vao base_scoraldet.yaml; tra ve duong dan YAML da sinh."""
    d = yaml.safe_load(BASE_YAML.read_text(encoding="utf-8"))
    d["nc"] = nc

    depth, width, max_ch = d["scales"]["n"]
    nb = len(d["backbone"])
    layers = [list(x) for x in d["backbone"] + d["head"]]
    _validate(layers)
    ch, reps = _compute(layers, depth, width, max_ch)

    # ---- C2f_MPFB: thay 3 khoi C2f backbone (idx 2,4,6) ----
    for i in MPFB_REPLACE_IDXS:
        c1, c2 = ch[i - 1], ch[i]
        shortcut = bool(layers[i][3][1]) if len(layers[i][3]) > 1 else True
        layers[i] = [layers[i][0], 1, "C2f_MPFB", [c1, c2, reps[i], shortcut]]

    # ---- VoVGSCSP: thay C2f/C2fCIB neck (idx 13,16,19,22). from = -1 => c1 = ch[i-1] ----
    for i in NECK_VOVGSCSP_IDXS:
        c1, c2 = ch[i - 1], ch[i]
        layers[i] = [layers[i][0], 1, "VoVGSCSP", [c1, c2, reps[i]]]

    # ---- GSConv: thay downsample neck (idx 17,20), giu stride 2 ----
    for i in NECK_GSCONV_IDXS:
        c1, c2 = ch[i - 1], ch[i]
        layers[i] = [layers[i][0], 1, "GSConv", [c1, c2, 3, 2]]

    d["backbone"], d["head"] = layers[:nb], layers[nb:]

    # ten chua 'yolov10n' de ultralytics nhan scale 'n'
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    out = save_dir / "scoraldet_yolov10n.yaml"
    out.write_text(yaml.safe_dump(d, sort_keys=False, default_flow_style=None), encoding="utf-8")
    return out


def build_scoraldet_model(nc=6, verbose=True, **kw):
    """register module + sinh YAML + tra ve YOLO chua train (tien cho smoke test / train wrapper)."""
    from ultralytics import YOLO
    from models.registry import is_channel_patch_applied
    register_scoraldet_modules()
    assert is_channel_patch_applied(), \
        "parse_model chua duoc va kenh -> C2f_MPFB/VoVGSCSP/GSConv co the dung sai c1 (pin dung ban ultralytics)"
    yaml_path = build_scoraldet_yaml(nc=nc, **kw)
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

    m = build_scoraldet_model(verbose=False)
    det = m.model.model[-1]
    m.model.eval()
    with torch.no_grad():
        y = m.model(torch.randn(1, 3, 640, 640))
    tensors = list(_iter_tensors(y))
    assert tensors, "khong tim thay tensor output"
    for t in tensors:
        assert torch.isfinite(t).all(), "NaN/Inf trong output SCoralDet"
    assert det.nl == 3, f"SCoralDet can 3 head, thay {det.nl}"
    assert [int(s) for s in det.stride] == [8, 16, 32], \
        f"stride {[int(s) for s in det.stride]} != [8, 16, 32]"
    n_par = sum(p.numel() for p in m.model.parameters())
    print(f"OK SCoralDet  nl={det.nl}  stride={[int(s) for s in det.stride]}  params(train)={n_par/1e6:.2f}M")
    # tuy chon: reparameterize de do params/FLOPs inference (gan voi paper 2.4M/5.9G)
    for mod in m.model.modules():
        if hasattr(mod, "switch_to_deploy") and mod.__class__.__name__ == "C2f_MPFB":
            mod.switch_to_deploy()
    n_par_dep = sum(p.numel() for p in m.model.parameters())
    print(f"   params(deploy, sau reparam MPFB)={n_par_dep/1e6:.2f}M")