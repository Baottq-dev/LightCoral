"""
Script 4: Evaluate trained model trên SCoralDet test set

Usage:
    python scripts/4_evaluate.py --weights runs/coral_benchmark/yolov8s_imgsz640_ep100/weights/best.pt
    python scripts/4_evaluate.py --weights runs/.../best.pt --split test --imgsz 640
"""

import argparse
import json
from pathlib import Path
from ultralytics import YOLO


# ─── CONFIG ────────────────────────────────────────────────────────────────────
# Dùng path relative đến script để portable trên mọi máy
_SCRIPT_DIR = Path(__file__).resolve().parent          # f:.../scripts/
_REPO_ROOT  = _SCRIPT_DIR.parent                       # f:.../LightCoral-YOLO/
YAML_PATH   = _REPO_ROOT / "configs" / "coral_soft.yaml"
# ───────────────────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate YOLO model on SCoralDet")
    parser.add_argument("--weights", type=str, required=True, help="Path to best.pt")
    parser.add_argument("--split",   type=str, default="test", choices=["val", "test"])
    parser.add_argument("--imgsz",   type=int, default=640)
    parser.add_argument("--batch",   type=int, default=8)
    parser.add_argument("--device",  type=str, default="0")
    parser.add_argument("--conf",    type=float, default=0.25)
    parser.add_argument("--iou",     type=float, default=0.5)
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Directory to save eval JSON. Default: same folder as best.pt")
    return parser.parse_args()


def main():
    args = parse_args()
    weights = Path(args.weights)

    print(f"📊 Evaluating: {weights.name}")
    print(f"   Split : {args.split}")
    print(f"   ImgSz : {args.imgsz}")

    model = YOLO(str(weights))

    metrics = model.val(
        data    = str(YAML_PATH),
        split   = args.split,
        imgsz   = args.imgsz,
        batch   = args.batch,
        device  = args.device,
        conf    = args.conf,
        iou     = args.iou,
        plots   = True,
        save_json = True,
    )

    # Extract key metrics
    names = ["Euphflfiaancora", "Favosites", "Platygyra", "Sarcophyton", "Sinularia", "WavingHand"]
    print("\n" + "="*60)
    print(f"{'Metric':<25} {'Value':>10}")
    print("="*60)
    print(f"{'mAP@0.5':<25} {metrics.box.map50:>10.4f}")
    print(f"{'mAP@0.5:0.95':<25} {metrics.box.map:>10.4f}")
    print(f"{'Precision (mean)':<25} {metrics.box.mp:>10.4f}")
    print(f"{'Recall (mean)':<25} {metrics.box.mr:>10.4f}")
    print("-"*60)
    print("\nPer-class AP@0.5:")
    for i, (name, ap) in enumerate(zip(names, metrics.box.ap50)):
        print(f"  {name:<22} {ap:>8.4f}")
    print("="*60)

    # Save results to JSON
    run_name = weights.parent.parent.name

    # out_dir: --out_dir arg > thư mục của best.pt > PROJECT_DIR
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = weights.parent.parent  # runs/coral_benchmark/<run_name>/

    out_dir.mkdir(parents=True, exist_ok=True)  # tạo nếu chưa có
    out_json = out_dir / f"eval_{run_name}_{args.split}.json"

    result_dict = {
        "model": str(weights),
        "split": args.split,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "per_class_AP50": {name: float(ap) for name, ap in zip(names, metrics.box.ap50)},
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=2)
    print(f"\n✅ Results saved: {out_json}")


if __name__ == "__main__":
    main()
