# test.py  (dat o ROOT repo sc-yolo12/, ngang hang train.py)
# Danh gia mot run da train tren tap test (hoac val/train).
# Chay tu thu muc goc:
#   python test.py --weights runs/scyolo12/E8_s0/weights/best.pt --data cfg/coral_soft_yolo.yaml
#   python test.py --weights runs/scyolo12/B0_s0/weights/best.pt --data cfg/coral_soft_yolo.yaml --csv runs/scyolo12/B0_s0_test.csv

import argparse
import csv
import json
from pathlib import Path

from ultralytics import YOLO

from engine.build_model import ROOT
from models.registry import register_custom_modules
from utils.seed import set_seed


def main():
    ap = argparse.ArgumentParser("SC-YOLO12 evaluator (test/val)")
    ap.add_argument("--weights", required=True, help="duong dan .pt (vd runs/scyolo12/E8_s0/weights/best.pt)")
    ap.add_argument("--data", required=True, help="data YAML (vd cfg/coral_soft_yolo.yaml)")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"], help="tap danh gia (mac dinh: test)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None, help="vd 0 | cpu (mac dinh: tu chon)")
    ap.add_argument("--conf", type=float, default=0.001, help="conf threshold khi DO mAP (chuan Ultralytics, KHONG phai inference)")
    ap.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--project", default=str(ROOT / "runs" / "scyolo12"))
    ap.add_argument("--name", default=None, help="ten thu muc ket qua; mac dinh suy ra tu --weights (vd E8_s0_test)")
    ap.add_argument("--csv", default=None, help="duong dan CSV xuat mAP per-class (tuy chon)")
    args = ap.parse_args()

    set_seed(args.seed)
    register_custom_modules()       # PHAI goi truoc khi load model co custom module (SFDF/PG-DAM/FGA2)

    # ten run mac dinh: <ten_run_train>_<split>, vd runs/scyolo12/E8_s0 -> E8_s0_test
    run_name = args.name or f"{Path(args.weights).parents[1].name}_{args.split}"

    model = YOLO(args.weights)
    val_kwargs = dict(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,
        iou=args.iou,
        project=args.project,
        name=run_name,
        plots=True,
        verbose=True,
    )
    if args.device is not None:
        val_kwargs["device"] = args.device

    metrics = model.val(**val_kwargs)
    box = metrics.box

    # ---------- tong hop ----------
    print(f"\n================ KET QUA TREN TAP '{args.split}' ================")
    print(f"mAP50-95 : {box.map:.4f}")
    print(f"mAP50    : {box.map50:.4f}")
    print(f"mAP75    : {box.map75:.4f}")
    print(f"Precision: {box.mp:.4f}    Recall: {box.mr:.4f}")

    # ---------- per-class ----------
    names = model.names if isinstance(model.names, dict) else {i: n for i, n in enumerate(model.names)}
    rows = []
    print("\nPer-class (P / R / mAP50 / mAP50-95):")
    for i, ci in enumerate(box.ap_class_index):
        p, r, ap50, ap = box.class_result(i)
        cname = names.get(int(ci), str(ci))
        print(f"  {int(ci):>2}  {cname:<22} P={p:.3f}  R={r:.3f}  mAP50={ap50:.3f}  mAP50-95={ap:.3f}")
        rows.append([int(ci), cname, f"{p:.6f}", f"{r:.6f}", f"{ap50:.6f}", f"{ap:.6f}"])

    # ---------- LUON luu ket qua vao thu muc run ----------
    # lay dung save_dir ma Ultralytics da tao (co the bi increment neu trung ten)
    save_dir = Path(
        getattr(metrics, "save_dir", None)
        or getattr(getattr(model, "validator", None), "save_dir", None)
        or (Path(args.project) / run_name)
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    header = ["class_id", "class_name", "precision", "recall", "mAP50", "mAP50-95"]
    all_row = ["ALL", "all", f"{box.mp:.6f}", f"{box.mr:.6f}", f"{box.map50:.6f}", f"{box.map:.6f}"]

    # CSV: luon ghi vao thu muc run; neu co --csv thi ghi them ban sao o do
    csv_targets = [save_dir / f"metrics_{args.split}.csv"]
    if args.csv:
        csv_targets.append(Path(args.csv))
    for out in csv_targets:
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
            w.writerow(all_row)
        print(f"Da luu CSV: {out}")

    # JSON tom tat (tien cho script tong hop ablation doc lai)
    summary = {
        "split": args.split,
        "weights": str(args.weights),
        "map50_95": float(box.map),
        "map50": float(box.map50),
        "map75": float(box.map75),
        "precision": float(box.mp),
        "recall": float(box.mr),
        "per_class_map50_95": {names.get(int(ci), str(ci)): float(box.maps[int(ci)]) for ci in box.ap_class_index},
    }
    sjson = save_dir / f"metrics_{args.split}.json"
    sjson.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Da luu JSON: {sjson}")
    print(f"\nTat ca ket qua (plots + CSV + JSON) o: {save_dir}")


if __name__ == "__main__":
    main()