# run_in_yolo.py
"""
Chay tung module rieng le THONG QUA Ultralytics (de kiem tra tich hop YAML).
Yeu cau: da dang ky F3M/F3MWithSA (import f3m_register hoac copy vao package).

Cach dung:
    python run_in_yolo.py stem    # chi build model co F3MWithSA o stem
    python run_in_yolo.py neck    # chi build model co F3M o neck
    python run_in_yolo.py full    # build model co ca 2 (yolo12n-f3m.yaml)
"""

import sys
import torch
import f3m_register            # dang ky F3M / F3MWithSA
from ultralytics import YOLO

CFG = {
    "stem": "yolo12n-f3m-stem-only.yaml",
    "neck": "yolo12n-f3m-neck-only.yaml",
    "full": "yolo12n-f3m.yaml",
}


def main(which):
    cfg = CFG[which]
    print(f"Building model from: {cfg}")
    model = YOLO(cfg)

    # In tom tat kien truc
    model.info(verbose=True)

    # Forward thu mot anh gia
    dummy = torch.randn(1, 3, 640, 640)
    model.model.eval()
    with torch.no_grad():
        out = model.model(dummy)
    print("Forward OK.")


if __name__ == "__main__":
    which = sys.argv[1].lower() if len(sys.argv) > 1 else "full"
    assert which in CFG, f"Chon mot trong: {list(CFG)}"
    main(which)