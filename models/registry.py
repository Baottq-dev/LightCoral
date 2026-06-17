# models/registry.py
# Dang ky custom module vao ultralytics.nn.tasks de parse_model nhin thay.

import inspect
import re
import textwrap

import ultralytics.nn.tasks as tasks_mod
import ultralytics.nn.modules as nn_modules
from ultralytics.utils import LOGGER

from .sfdf import SFDF
from .shallow_p2 import AMCF
from .pg_dam import PGDAM_FiLM
from .fga2 import FGA2_A2C2f

CUSTOM_MODULES = {
    "SFDF": SFDF,                # module 2
    "AMCF": AMCF,                # module 1 (va vao idx 7 boi build_model)
    "PGDAM_FiLM": PGDAM_FiLM,    # module 4
    "FGA2_A2C2f": FGA2_A2C2f,    # module 5
}


_PARSE_PATCHED_FLAG = "_sc_parse_model_patched"


def register_custom_modules():
    """Inject custom class vao namespace ma parse_model tra cuu.

    parse_model (ultralytics/nn/tasks.py) phan giai ten module bang
    globals() cua module tasks => setattr vao tasks_mod la du.
    Setattr them vao nn_modules de ho tro deserialize checkpoint (torch.load
    can tim thay class theo duong dan import da pickle).
    """
    for name, cls in CUSTOM_MODULES.items():
        setattr(tasks_mod, name, cls)
        setattr(nn_modules, name, cls)

    # Day so kenh dau ra dung cho custom module vao parse_model.
    _patch_parse_model_channels()


def _patch_parse_model_channels():
    """Va parse_model de no biet kenh dau ra cua custom module.

    Van de: parse_model khong biet custom module doi so kenh => roi vao nhanh
    `else: c2 = ch[f]`, tuc lay NHAM kenh dau VAO lam kenh dau RA. Voi module
    doi kenh (SFDF 64->128, FGA2_A2C2f 384->128) thi layer ke tiep dung sai c1
    => RuntimeError "expected X channels, but got Y".

    Cach sua it phu thuoc phien ban nhat: doc source cua parse_model, chen mot
    nhanh `elif m in _SC_CUSTOM_MODULES: c2 = args[1]` ngay truoc `else`.
    build_model.py luon ghi args tuong minh [c1, c2, ...] nen args[1] = kenh ra.
    Neu khong va duoc (source khac mong doi) thi bo qua an toan.
    """
    if getattr(tasks_mod, _PARSE_PATCHED_FLAG, False):
        return
    try:
        src = textwrap.dedent(inspect.getsource(tasks_mod.parse_model))
    except (OSError, TypeError):
        return

    # tim nhanh `else:` <newline> `c2 = ch[f]` (cung muc thut dau)
    pat = re.compile(r"(?P<ind>[ \t]+)else:\s*\n(?P=ind)[ \t]+c2 = ch\[f\]")
    mo = pat.search(src)
    if not mo:
        LOGGER.warning(
            "SC-YOLO12: KHONG va duoc parse_model (cau truc nguon khac mong doi). "
            "Custom module doi kenh (SFDF/FGA2) co the dung sai c1 -> RuntimeError. "
            "Hay PIN dung phien ban ultralytics da kiem thu va chay "
            "'python -m engine.build_model' de smoke test."
        )
        return  # cau truc parse_model khac -> khong va, tranh lam hong

    ind = mo.group("ind")
    body = ind + "    "
    injected = (
        f"{ind}elif m in _SC_CUSTOM_MODULES:\n"
        f"{body}c2 = args[1] if len(args) > 1 else ch[f]\n"
        f"{mo.group(0)}"
    )
    src = src[: mo.start()] + injected + src[mo.end():]
    src = src.replace("def parse_model(", "def _sc_parse_model(", 1)

    # exec trong ban sao namespace cua tasks (du cac ten Conv, make_divisible, ...)
    ns = dict(tasks_mod.__dict__)
    ns["_SC_CUSTOM_MODULES"] = frozenset(CUSTOM_MODULES.values())
    try:
        exec(compile(src, "<sc_parse_model>", "exec"), ns)
        new_fn = ns["_sc_parse_model"]
    except Exception as e:
        LOGGER.warning(f"SC-YOLO12: va parse_model that bai ({e}); giu nguyen ban goc.")
        return

    tasks_mod.parse_model = new_fn
    setattr(tasks_mod, _PARSE_PATCHED_FLAG, True)


def is_registered():
    return all(hasattr(tasks_mod, n) for n in CUSTOM_MODULES)


def is_channel_patch_applied():
    """True neu da va thanh cong parse_model (kenh ra custom module = args[1]).

    Nen `assert is_channel_patch_applied()` sau register_custom_modules() khi dung
    SFDF/FGA2: neu patch that bai, model van build duoc o vai to hop nhung SAI kenh."""
    return bool(getattr(tasks_mod, _PARSE_PATCHED_FLAG, False))