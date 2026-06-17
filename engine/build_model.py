# engine/build_model.py
# Sinh YAML dong theo to hop module va tao model Ultralytics.
# Quy uoc: custom module trong YAML co repeats n=1 va args chua KENH DA SCALE
# tuong minh (parse_model khong scale args cua module la - xem models/registry.py).

from pathlib import Path

import yaml

from models.registry import register_custom_modules
from models.shallow_p2 import select_base_yaml, validate_yaml

ROOT = Path(__file__).resolve().parents[1]

BUILTIN_SCALED = {"Conv", "C3k2", "A2C2f", "SPPF", "C2f"}   # parse_model tu scale args[0]
CUSTOM = {"SFDF", "AMCF", "PGDAM_FiLM", "FGA2_A2C2f"}        # args tuong minh, khong scale


def _make_divisible(x, divisor=8):
    import math
    return math.ceil(x / divisor) * divisor


def _scaled(c, width, max_ch):
    return _make_divisible(min(c, max_ch) * width, 8)


def _compute_channels(layers, width, max_ch):
    """Mo phong cach parse_model lan truyen kenh de biet c1/c2 tung layer.
    Tra ve list ch: ch[i] = kenh ra cua layer i.
    Chay tren YAML DA LOAD (truoc khi va) => moi layer la built-in/Concat/Upsample/Detect.
    """
    ch = [3]
    out = []
    for i, (f, n, m, args) in enumerate(layers):
        f0 = f[0] if isinstance(f, list) else f
        c1 = out[f0] if (isinstance(f0, int) and f0 >= 0) else (out[f0] if out else ch[0])
        if i == 0:
            c1 = ch[0]
        if m in BUILTIN_SCALED:
            c2 = _scaled(args[0], width, max_ch)
        elif m in CUSTOM:
            c2 = args[1]                      # da tuong minh
        elif m == "Concat":
            c2 = sum(out[x] for x in f)       # indexing am cung dung
        elif m == "nn.Upsample":
            c2 = c1
        elif m == "Detect":
            c2 = c1
        else:
            raise ValueError(f"Chua ho tro tinh kenh cho module {m}")
        out.append(c2)
    return out


def _depth_n(n, depth):
    return max(round(n * depth), 1) if n > 1 else n


def _reindex_after_insert(layers, pos):
    """Sau khi chen 1 layer tai vi tri pos: moi index tuyet doi >= pos thi +1.
    So am (tuong doi) giu nguyen."""
    def fix(x):
        return x + 1 if (isinstance(x, int) and x >= pos) else x
    for layer in layers:
        f = layer[0]
        layer[0] = [fix(x) for x in f] if isinstance(f, list) else fix(f)


def build_yaml_for_modules(modules, specs, nc=6, save_dir=ROOT / "runs" / "cfg"):
    """modules: iterable cac so 1..5 (rong = baseline B0). Tra ve duong dan YAML."""
    mods = set(int(m) for m in modules)
    assert mods <= {1, 2, 3, 4, 5}, f"Module khong hop le: {mods}"

    base = select_base_yaml(mods)
    d = yaml.safe_load(Path(base).read_text(encoding="utf-8"))
    validate_yaml(d, with_shallow=(1 in mods))
    d["nc"] = nc

    depth, width, max_ch = d["scales"]["n"]
    nb = len(d["backbone"])
    layers = [list(x) for x in d["backbone"] + d["head"]]
    ch = _compute_channels(layers, width, max_ch)

    sp = specs["modules"]

    # ---- (2) SFDF: thay 4 khoi dac trung backbone (idx 2,4,6,8) ----
    if 2 in mods:
        swap_idxs = set(sp["sfdf"].get("swap_layer_idxs", []))
        for i in sp["sfdf"]["replace_layer_idxs"]:
            c1, c2 = ch[i - 1], ch[i]
            layers[i] = [-1, 1, "SFDF",
                         [c1, c2, sp["sfdf"]["reduction"], sp["sfdf"]["pconv_kernel"], i in swap_idxs]]

    # ---- (1) AMCF: thay downsample cuoi backbone (idx 7) khi dung shallow variant ----
    if 1 in mods:
        i = sp["shallow"]["amcf_replace_idx"]
        assert layers[i][2] == "Conv", f"Layer {i} phai la Conv (diem cam AMCF)"
        c1, c2 = ch[i - 1], ch[i]
        layers[i] = [-1, 1, "AMCF", [c1, c2, sp["shallow"]["amcf_stride"]]]

    # ---- (5) FGA2: thay A2C2f tai idx 11, 14 ----
    if 5 in mods:
        for i in sp["fga2"]["replace_layer_idxs"]:
            f, n, m, args = layers[i]
            assert m == "A2C2f", f"Layer {i} phai la A2C2f"
            f0 = f[0] if isinstance(f, list) else f
            c1 = ch[f0] if f0 >= 0 else ch[i - 1]
            c2 = ch[i]
            layers[i] = [f, 1, "FGA2_A2C2f",
                         [c1, c2, _depth_n(n, depth), sp["fga2"]["area"], sp["fga2"]["lambda_init"]]]

    # ---- (4) PG-DAM: chen sau stem (idx 0), reindex (LAM CUOI cung) ----
    if 4 in mods:
        pos = sp["pg_dam"]["insert_after_idx"] + 1
        c_stem = ch[sp["pg_dam"]["insert_after_idx"]]
        _reindex_after_insert(layers, pos)
        layers.insert(pos, [-1, 1, "PGDAM_FiLM",
                            [c_stem, c_stem, sp["pg_dam"]["z_dim"], sp["pg_dam"]["hidden"]]])
        nb += 1  # backbone dai them 1 layer

    d["backbone"], d["head"] = layers[:nb], layers[nb:]

    # luu YAML (ten chua 'yolov12n' de ultralytics nhan dien scale 'n')
    tag = "".join(str(m) for m in sorted(mods)) or "0"
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    out = save_dir / f"yolov12n-m{tag}.yaml"
    out.write_text(yaml.safe_dump(d, sort_keys=False, default_flow_style=None), encoding="utf-8")
    return out


def build_model(modules, specs, nc=6, verbose=True):
    """Tien ich nhanh cho smoke test: tra ve doi tuong YOLO chua train."""
    from ultralytics import YOLO
    register_custom_modules()
    yaml_path = build_yaml_for_modules(modules, specs, nc=nc)
    return YOLO(str(yaml_path), verbose=verbose)


def _iter_tensors(o):
    """Duyet de quy moi cau truc output (tensor/list/tuple/dict) -> sinh ra cac Tensor.
    Detect end2end (YOLOv10/v12) tra ve dict {'one2many':..., 'one2one':...} nen can gom de quy."""
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
    # smoke test: build du 16 to hop kien truc, forward 1 tensor, kiem NaN + so head + stride.
    import itertools, torch
    from models.registry import is_channel_patch_applied
    specs = yaml.safe_load((ROOT / "cfg" / "module_specs.yaml").read_text(encoding="utf-8"))
    register_custom_modules()
    assert is_channel_patch_applied(), "parse_model chua duoc va kenh -> SFDF/FGA2/AMCF co the sai c1"
    arch_mods = [1, 2, 4, 5]  # module 3 khong doi kien truc
    for r in range(len(arch_mods) + 1):
        for combo in itertools.combinations(arch_mods, r):
            m = build_model(combo, specs, verbose=False)
            det = m.model.model[-1]                       # lop Detect
            m.model.eval()                                # eval => output on dinh, tranh dict loss
            with torch.no_grad():
                y = m.model(torch.randn(1, 3, 640, 640))  # co the la tensor/list/tuple/dict
            tensors = list(_iter_tensors(y))              # gom moi tensor (Detect end2end tra dict)
            assert tensors, f"modules={combo}: khong tim thay tensor output"
            for t in tensors:
                assert torch.isfinite(t).all(), f"NaN/Inf trong output modules={combo}"
            if 1 in combo:                                # shallow => 2 head P4/P5 (stride 8,16)
                assert det.nl == 2, f"modules={combo}: can 2 head, thay {det.nl}"
                assert [int(s) for s in det.stride] == [8, 16], \
                    f"modules={combo}: stride {[int(s) for s in det.stride]} != [8, 16]"
            print(f"OK modules={combo or ('baseline',)}  nl={det.nl}  stride={[int(s) for s in det.stride]}")