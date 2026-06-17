# f3m_register.py
"""
DANG KY module F3M / F3MWithSA voi Ultralytics khi CHAY (runtime).
=================================================================
Import file nay TRUOC khi goi YOLO(cfg). No se nap class F3M, F3MWithSA tu
f3m_modules.py va tiem vao namespace ma parser cua Ultralytics tra cuu:
    - ultralytics.nn.tasks    (noi parse_model dung globals()[name])
    - ultralytics.nn.modules

KHONG can sua ultralytics/nn/modules/__init__.py.
KHONG can sua parse_model trong ultralytics/nn/tasks.py.

Vi sao khong can sua parse_model nua:
    F3M/F3MWithSA suy ra SO KENH TU DONG o forward dau tien (lazy build),
    nen khong can parser chen in_channels vao args. Parser di vao nhanh mac
    dinh 'else: c2 = ch[f]' (giu nguyen so kenh, dung voi thiet ke residual)
    va goi F3M(0.125, False) / F3MWithSA(0.33, True, 7) tu YAML.

Cach dung:
    import f3m_register
    from ultralytics import YOLO
    model = YOLO("yolo12n-f3m.yaml")
"""

import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from f3m_modules import F3M, F3MWithSA
except ImportError:
    from ultralytics.nn.modules.f3m_modules import F3M, F3MWithSA


def _register():
    for mod_name in ("ultralytics.nn.tasks", "ultralytics.nn.modules"):
        try:
            tgt = importlib.import_module(mod_name)
        except Exception as e:
            print(f"[f3m_register] WARN: khong import duoc {mod_name}: {e}")
            continue
        setattr(tgt, "F3M", F3M)
        setattr(tgt, "F3MWithSA", F3MWithSA)
    print("[f3m_register] Da dang ky F3M, F3MWithSA (lazy build, khong can sua parse_model).")


_register()