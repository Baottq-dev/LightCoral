# benchmark/sf_yolo/eval_sf_yolo.py
# Danh gia SF-YOLO faithful tren tap TEST (split="test").
# Chay tu ROOT repo:
#   python -m benchmark.sf_yolo.eval_sf_yolo --data data/scoraldet_fold0.yaml \
#       --weights runs/benchmark/SF_YOLO_s0/weights/best.pt --split test

import argparse
import sys
from pathlib import Path

# chay truc tiep: dam bao root repo (sc-yolo12/) trong sys.path
_ROOT_INIT = Path(__file__).resolve().parents[2]
if str(_ROOT_INIT) not in sys.path:
    sys.path.insert(0, str(_ROOT_INIT))

from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_num_params, get_flops

from models.registry import register_custom_modules

ROOT = Path(__file__).resolve().parents[2]


def _save_metrics(metrics, model, split, save_dir):
    """Luu metrics_<split>.csv + metrics_<split>.json - khop dinh dang test.py."""
    import csv, json
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    box = metrics.box
    names = model.names if isinstance(model.names, dict) else {i: n for i, n in enumerate(model.names)}
    header = ["class_id", "class_name", "precision", "recall", "mAP50", "mAP50-95"]
    rows = []
    for i, ci in enumerate(box.ap_class_index):
        p, r, ap50, ap = box.class_result(i)
        rows.append([int(ci), names.get(int(ci), str(ci)),
                     f"{p:.6f}", f"{r:.6f}", f"{ap50:.6f}", f"{ap:.6f}"])
    all_row = ["ALL", "all", f"{box.mp:.6f}", f"{box.mr:.6f}",
               f"{box.map50:.6f}", f"{box.map:.6f}"]
    csv_path = save_dir / f"metrics_{split}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([header] + rows + [all_row])
    summary = {
        "split": split,
        "map50_95": float(box.map), "map50": float(box.map50),
        "map75": float(box.map75) if hasattr(box, "map75") else None,
        "precision": float(box.mp), "recall": float(box.mr),
        "per_class_map50_95": {
            names.get(int(ci), str(ci)): float(box.maps[int(ci)])
            for ci in box.ap_class_index
        },
    }
    json_path = save_dir / f"metrics_{split}.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Metrics saved -> {csv_path}  |  {json_path}")


def main():
    ap = argparse.ArgumentParser("SF-YOLO evaluator (benchmark, test split)")
    ap.add_argument("--data", required=True, help="data YAML (PHAI co khai bao 'test:')")
    ap.add_argument("--weights", required=True, help="checkpoint best.pt da train")
    ap.add_argument("--split", default="test", choices=["test", "val", "train"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--project", default=str(ROOT / "benchmark" / "runs"))
    ap.add_argument("--name", default="SF_YOLO_eval")
    args = ap.parse_args()

    register_custom_modules()           # PHAI goi truoc khi load checkpoint custom (SFDF/AMCF)
    model = YOLO(args.weights)

    save_dir = Path(args.project) / args.name
    save_dir.mkdir(parents=True, exist_ok=True)

    metrics = model.val(
        data=args.data, split=args.split, imgsz=args.imgsz, batch=args.batch,
        device=args.device, project=str(save_dir.parent), name=save_dir.name,
        exist_ok=True, verbose=True, plots=True, save_json=False,
    )
    _save_metrics(metrics, model, args.split, save_dir)
    b = metrics.box
    n_par = get_num_params(model.model)
    gflops = get_flops(model.model, imgsz=args.imgsz)
    det = model.model.model[-1]
    print(f"\n===== SF-YOLO faithful  (split={args.split}) =====")
    print(f"heads    = {det.nl}  (SF-YOLO = 2 head P4/P5)")
    print(f"P        = {b.mp:.4f}")
    print(f"R        = {b.mr:.4f}")
    print(f"mAP50    = {b.map50:.4f}")
    print(f"mAP50-95 = {b.map:.4f}")
    print(f"Params   = {n_par/1e6:.3f}M")
    print(f"GFLOPs   = {gflops:.2f}")
    if getattr(b, "maps", None) is not None:
        print("per-class mAP50-95:", [round(float(x), 4) for x in b.maps])


if __name__ == "__main__":
    main()