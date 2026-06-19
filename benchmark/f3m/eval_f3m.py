# benchmark/f3m/eval_f3m.py
# Danh gia F3M-YOLO11n tren tap TEST (split="test") - khop dinh nghia metric voi paper Table 1.
# Chay tu ROOT repo:
#   python -m benchmark.f3m.eval_f3m --data data/scoraldet_fold0.yaml \
#       --weights runs/benchmark/F3M_s0/weights/best.pt --split test

import argparse
import sys
from pathlib import Path

# chay truc tiep: dam bao root repo (sc-yolo12/) trong sys.path
_ROOT_INIT = Path(__file__).resolve().parents[2]
if str(_ROOT_INIT) not in sys.path:
    sys.path.insert(0, str(_ROOT_INIT))

from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_num_params, get_flops

# ho tro chay ca '-m benchmark.f3m.eval_f3m' lan 'python benchmark/f3m/eval_f3m.py'
try:
    from benchmark.f3m.build_f3m import register_f3m_modules
except ImportError:
    from build_f3m import register_f3m_modules

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser("F3M evaluator (benchmark, test split)")
    ap.add_argument("--data", required=True, help="data YAML (PHAI co khai bao 'test:')")
    ap.add_argument("--weights", required=True, help="checkpoint best.pt da train")
    ap.add_argument("--split", default="test", choices=["test", "val", "train"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--project", default=str(ROOT / "benchmark" / "runs"))
    ap.add_argument("--name", default="F3M_eval")
    args = ap.parse_args()

    register_f3m_modules()              # PHAI goi truoc khi load checkpoint custom (deserialize class)
    model = YOLO(args.weights)

    metrics = model.val(
        data=args.data, split=args.split, imgsz=args.imgsz, batch=args.batch,
        device=args.device, project=args.project, name=args.name,
        exist_ok=True, verbose=True,
    )
    b = metrics.box
    n_par = get_num_params(model.model)
    gflops = get_flops(model.model, imgsz=args.imgsz)
    print(f"\n===== F3M-YOLO11n  (split={args.split}) =====")
    print(f"P        = {b.mp:.4f}")
    print(f"R        = {b.mr:.4f}")
    print(f"mAP50    = {b.map50:.4f}   (paper 0.797)")
    print(f"mAP50-95 = {b.map:.4f}   (paper 0.539)")
    print(f"Params   = {n_par/1e6:.3f}M   (paper 2.61M)")
    print(f"GFLOPs   = {gflops:.2f}   (paper 6.5)")
    # mAP50-95 theo tung lop (doi chieu Table 2 neu can)
    if getattr(b, "maps", None) is not None:
        print("per-class mAP50-95:", [round(float(x), 4) for x in b.maps])


if __name__ == "__main__":
    main()