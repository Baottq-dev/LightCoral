# benchmark/scoraldet/eval_scoraldet.py
# Danh gia SCoralDet tren tap TEST (split="test") - khop dinh nghia metric voi paper Table 1/3.
# Chay tu ROOT repo:
#   python -m benchmark.scoraldet.eval_scoraldet --data data/scoraldet_fold0.yaml \
#       --weights runs/benchmark/SCoralDet_s0/weights/best.pt --split test --reparam

import argparse
import sys
from pathlib import Path

# chay truc tiep: dam bao root repo (sc-yolo12/) trong sys.path
_ROOT_INIT = Path(__file__).resolve().parents[2]
if str(_ROOT_INIT) not in sys.path:
    sys.path.insert(0, str(_ROOT_INIT))

from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_num_params, get_flops

# ho tro chay ca '-m benchmark.scoraldet.eval_scoraldet' lan 'python benchmark/scoraldet/eval_scoraldet.py'
try:
    from benchmark.scoraldet.build_scoraldet import register_scoraldet_modules
except ImportError:
    from build_scoraldet import register_scoraldet_modules

ROOT = Path(__file__).resolve().parents[2]


def _reparam(model):
    """Reparameterize moi C2f_MPFB ve 1 conv NxN (do params/FLOPs che do deploy)."""
    n = 0
    for mod in model.model.modules():
        if mod.__class__.__name__ == "C2f_MPFB" and hasattr(mod, "switch_to_deploy"):
            mod.switch_to_deploy()
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser("SCoralDet evaluator (benchmark, test split)")
    ap.add_argument("--data", required=True, help="data YAML (PHAI co khai bao 'test:')")
    ap.add_argument("--weights", required=True, help="checkpoint best.pt da train")
    ap.add_argument("--split", default="test", choices=["test", "val", "train"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--reparam", action="store_true",
                    help="switch_to_deploy MPFB de do Params/GFLOPs deploy (paper 2.4M/5.9G)")
    ap.add_argument("--project", default=str(ROOT / "benchmark" / "runs"))
    ap.add_argument("--name", default="SCoralDet_eval")
    args = ap.parse_args()

    register_scoraldet_modules()        # PHAI goi truoc khi load checkpoint custom
    model = YOLO(args.weights)

    metrics = model.val(
        data=args.data, split=args.split, imgsz=args.imgsz, batch=args.batch,
        device=args.device, project=args.project, name=args.name,
        exist_ok=True, verbose=True,
    )
    b = metrics.box
    n_par = get_num_params(model.model)
    gflops = get_flops(model.model, imgsz=args.imgsz)
    print(f"\n===== SCoralDet  (split={args.split}) =====")
    print(f"P        = {b.mp:.4f}")
    print(f"R        = {b.mr:.4f}")
    print(f"mAP50    = {b.map50:.4f}   (paper 0.819)")
    print(f"mAP50-95 = {b.map:.4f}   (paper 0.532)")
    print(f"Params(train)   = {n_par/1e6:.3f}M")
    print(f"GFLOPs(train)   = {gflops:.2f}")
    if args.reparam:
        k = _reparam(model)
        n_dep = get_num_params(model.model)
        g_dep = get_flops(model.model, imgsz=args.imgsz)
        print(f"-- sau reparam {k} khoi C2f_MPFB --")
        print(f"Params(deploy)  = {n_dep/1e6:.3f}M   (paper ~2.4M)")
        print(f"GFLOPs(deploy)  = {g_dep:.2f}   (paper ~5.9)")


if __name__ == "__main__":
    main()