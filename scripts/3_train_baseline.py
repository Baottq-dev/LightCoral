"""
Script 3: Train YOLO baseline on SCoralDet

Supported model families (ultralytics hub — downloaded automatically):
  YOLOv8  : yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
  YOLOv9  : yolov9c, yolov9e
  YOLOv10 : yolov10n, yolov10s, yolov10m, yolov10b, yolov10l, yolov10x
  YOLO11  : yolo11n, yolo11s, yolo11m, yolo11l, yolo11x
  YOLO12  : yolo12n, yolo12s, yolo12m, yolo12l, yolo12x
  YOLO26  : yolo26n, yolo26s, yolo26m, yolo26l, yolo26x
  RT-DETR : rtdetr-r50, rtdetr-r101

Custom / local models (provide full path via --weights):
  YOLO26  : --weights f:/YOLO26-SELF_DISTI/weights/yolo26s.pt
  Any .pt : --weights path/to/your_model.pt

Usage:
  # Quick start (yolov8s, 100 epochs, imgsz=640)
  python scripts/3_train_baseline.py

  # Specify model and epochs
  python scripts/3_train_baseline.py --model yolo12s --epochs 150 --imgsz 1280

  # Custom data yaml and output
  python scripts/3_train_baseline.py --data configs/coral_soft.yaml --project runs/exp1

  # Custom/local weights (e.g. YOLO26)
  python scripts/3_train_baseline.py --weights f:/YOLO26-SELF_DISTI/weights/yolo26s.pt

  # Resume interrupted run
  python scripts/3_train_baseline.py --resume runs/coral_benchmark/yolov8s_ep100/weights/last.pt

  # CPU (no GPU)
  python scripts/3_train_baseline.py --device cpu

  # ──────────────────────────────────────────────────────
  # Custom online augmentation (RAM-based, từ 2_augmentations.py)
  # ──────────────────────────────────────────────────────
  # Bật custom augmentation (tất cả 4 groups, intensity mặc định = medium)
  python scripts/3_train_baseline.py --model yolov8s --custom_aug

  # Chỉ dùng underwater + noise, intensity mạnh
  python scripts/3_train_baseline.py --model yolo11s --custom_aug \\
      --aug_groups underwater noise --aug_intensity strong

  # Preview transforms trước khi train (không cần train)
  python scripts/2_augmentations.py --preview datasets/coral_soft_yolo/images/train/Sinularia_1.JPG
"""

import argparse
import importlib.util
import sys
from pathlib import Path


# ── Model registry ─────────────────────────────────────────────────────────────
# Maps CLI alias → pretrained weights filename (auto-downloaded from ultralytics hub)
MODELS: dict[str, str] = {
    # YOLOv8 family
    "yolov8n": "yolov8n.pt",
    "yolov8s": "yolov8s.pt",
    "yolov8m": "yolov8m.pt",
    "yolov8l": "yolov8l.pt",
    "yolov8x": "yolov8x.pt",
    # YOLOv9 family
    "yolov9c": "yolov9c.pt",
    "yolov9e": "yolov9e.pt",
    # YOLOv10 family
    "yolov10n": "yolov10n.pt",
    "yolov10s": "yolov10s.pt",
    "yolov10m": "yolov10m.pt",
    "yolov10b": "yolov10b.pt",
    "yolov10l": "yolov10l.pt",
    "yolov10x": "yolov10x.pt",
    # YOLO11 family
    "yolo11n": "yolo11n.pt",
    "yolo11s": "yolo11s.pt",
    "yolo11m": "yolo11m.pt",
    "yolo11l": "yolo11l.pt",
    "yolo11x": "yolo11x.pt",
    # YOLO12 family (attention-based, released 2025)
    "yolo12n": "yolo12n.pt",
    "yolo12s": "yolo12s.pt",
    "yolo12m": "yolo12m.pt",
    "yolo12l": "yolo12l.pt",
    "yolo12x": "yolo12x.pt",
    # YOLO26 family (ultralytics hub)
    "yolo26n": "yolo26n.pt",
    "yolo26s": "yolo26s.pt",
    "yolo26m": "yolo26m.pt",
    "yolo26l": "yolo26l.pt",
    "yolo26x": "yolo26x.pt",
    # RT-DETR family (transformer-based, ultralytics hub)
    # Dung .yaml de init backbone ImageNet pretrained (khong can download COCO .pt)
    "rtdetr-r50":  "rtdetr-resnet50.yaml",
    "rtdetr-r101": "rtdetr-resnet101.yaml",
}
# ───────────────────────────────────────────────────────────────────────────────


def _load_aug_module():
    """
    Import 2_augmentations.py động (tên file có số nên không dùng 'import' trực tiếp).
    Đảm bảo module được đăng ký vào sys.modules trước exec_module để tránh lỗi dataclass.
    """
    aug_path = Path(__file__).parent / "2_augmentations.py"
    if not aug_path.exists():
        print(f"[ERROR] Không tìm thấy: {aug_path}")
        sys.exit(1)
    spec = importlib.util.spec_from_file_location("coral_augmentations", str(aug_path))
    mod  = importlib.util.module_from_spec(spec)
    sys.modules["coral_augmentations"] = mod   # đăng ký trước khi exec → fix dataclass
    spec.loader.exec_module(mod)
    return mod


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train YOLO baseline on SCoralDet dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Model ──────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--model", type=str, default="yolov8s", choices=list(MODELS.keys()),
        help="Model alias (ultralytics hub). Ignored if --weights is set.",
    )
    parser.add_argument(
        "--weights", type=str, default=None, metavar="PATH",
        help=("Path to custom local weights (.pt). Use this for models NOT in "
              "ultralytics hub, e.g. YOLO26: "
              "--weights f:/YOLO26-SELF_DISTI/weights/yolo26s.pt"),
    )
    parser.add_argument(
        "--resume", type=str, default=None, metavar="LAST_PT",
        help="Resume training from last.pt (overrides --model and --weights).",
    )

    # ── Data ───────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--data", type=str, default="configs/coral_soft.yaml",
        help="Path to dataset YAML config.",
    )

    # ── Training hyperparameters ───────────────────────────────────────────────
    parser.add_argument("--epochs",       type=int,   default=100)
    parser.add_argument("--imgsz",        type=int,   default=640,
                        help="Input image size (pixels). Try 1280 for high-res images.")
    parser.add_argument("--batch",        type=int,   default=16,
                        help="Batch size. Use -1 for auto-batch.")
    parser.add_argument("--optimizer",    type=str,   default="SGD",
                        choices=["SGD", "Adam", "AdamW", "NAdam", "RAdam", "RMSProp", "auto"])
    parser.add_argument("--lr0",          type=float, default=0.001,
                        help="Initial learning rate.")
    parser.add_argument("--lrf",          type=float, default=0.01,
                        help="Final learning rate factor (lr0 * lrf).")
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--warmup_epochs",type=int,   default=3)

    # ── Augmentation ───────────────────────────────────────────────────────────
    parser.add_argument("--mosaic",  type=float, default=1.0,
                        help="Mosaic augmentation probability.")
    parser.add_argument("--flipud",  type=float, default=0,
                        help="Vertical flip probability (coral orientation-invariant).")
    parser.add_argument("--fliplr",  type=float, default=0.5)
    parser.add_argument("--degrees", type=float, default=0,
                        help="Rotation degrees (coral can appear at any angle).")
    parser.add_argument("--hsv_h",   type=float, default=0.015,
                        help="HSV hue shift (small: underwater has fixed color cast).")
    parser.add_argument("--hsv_s",   type=float, default=0.7)
    parser.add_argument("--hsv_v",   type=float, default=0.4)

    # ── Custom augmentation (2_augmentations.py) ───────────────────────────────────
    parser.add_argument(
        "--custom_aug", action="store_true",
        help=(
            "Dùng CoralTrainer với custom underwater augmentation (từ 2_augmentations.py). "
            "Augmentation chạy in-RAM, không ghi file ra disk."
        ),
    )
    parser.add_argument(
        "--aug_groups", nargs="+",
        default=["underwater", "noise", "occlusion", "gamma"],
        choices=["underwater", "noise", "occlusion", "gamma"],
        metavar="GROUP",
        help="Các nhóm augmentation (chỉ dùng với --custom_aug). "
             "Choices: underwater noise occlusion gamma",
    )
    parser.add_argument(
        "--aug_intensity", type=str, default="medium",
        choices=["light", "medium", "strong"],
        help="Độ mạnh augmentation: nhân hệ số 0.5×/1×/1.5× lên probability (chỉ dùng với --custom_aug).",
    )

    # ── Runtime ────────────────────────────────────────────────────────────────
    parser.add_argument("--device",  type=str, default="0",
                        help="CUDA device id (e.g. '0', '0,1') or 'cpu'.")
    parser.add_argument("--workers", type=int, default=0,
                        help="Dataloader workers. Use 0 on Windows to avoid multiprocessing issues.")

    # ── Output ─────────────────────────────────────────────────────────────────
    parser.add_argument("--project", type=str, default="runs/coral_benchmark",
                        help="Root directory for saving training runs.")
    parser.add_argument("--name",    type=str, default=None,
                        help="Run name (auto-generated from model/imgsz/epochs if not set).")

    return parser.parse_args()


def main():
    args = parse_args()

    # ── Import check ───────────────────────────────────────────────────────────
    try:
        from ultralytics import YOLO, RTDETR
    except ImportError:
        print("[ERROR] ultralytics is not installed.")
        print("        Run: pip install ultralytics")
        sys.exit(1)

    def _load_model(weights_str):
        """Load YOLO or RTDETR depending on model name."""
        if 'rtdetr' in weights_str.lower():
            return RTDETR(weights_str)
        return YOLO(weights_str)

    # ── Resolve model (priority: --resume > --weights > --model) ──────────────
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            print(f"[ERROR] Resume checkpoint not found: {resume_path}")
            sys.exit(1)
        model    = _load_model(str(resume_path))
        run_name = args.name or resume_path.parent.parent.name + "_resumed"
        print(f"Resuming from: {resume_path}")
    elif args.weights:
        weights_path = Path(args.weights)
        if not weights_path.exists():
            print(f"[ERROR] Custom weights not found: {weights_path}")
            sys.exit(1)
        model    = _load_model(str(weights_path))
        run_name = args.name or f"{weights_path.stem}_imgsz{args.imgsz}_ep{args.epochs}"
        print(f"Custom weights: {weights_path}")
    else:
        hub_weights = MODELS[args.model]
        model       = _load_model(hub_weights)
        run_name    = args.name or f"{args.model}_imgsz{args.imgsz}_ep{args.epochs}"

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"[ERROR] Dataset config not found: {data_path}")
        print("        Run: python scripts/1_prepare_dataset.py  first.")
        sys.exit(1)

    # ── Print config ───────────────────────────────────────────────────────────
    print("=" * 56)
    print(f"  Model   : {args.model}  ({MODELS.get(args.model, 'resumed')})")
    print(f"  Data    : {data_path}")
    print(f"  Epochs  : {args.epochs}   ImgSz: {args.imgsz}   Batch: {args.batch}")
    print(f"  Device  : {args.device}   Workers: {args.workers}")
    print(f"  Output  : {args.project}/{run_name}")
    print("=" * 56)

    # ── Train ────────────────────────────────────────────────────────────────
    if args.custom_aug:
        # ── Custom aug: dùng CoralTrainer từ 2_augmentations.py ──────────────────
        aug = _load_aug_module()
        coral_config = aug.CoralAugConfig(
            groups    = args.aug_groups,
            intensity = args.aug_intensity,
        )
        print(coral_config.summary())
        print("=" * 56)

        overrides = dict(
            model         = args.weights or MODELS[args.model],
            data          = str(data_path),
            epochs        = args.epochs,
            imgsz         = args.imgsz,
            batch         = args.batch,
            optimizer     = args.optimizer,
            lr0           = args.lr0,
            lrf           = args.lrf,
            weight_decay  = args.weight_decay,
            warmup_epochs = args.warmup_epochs,
            # Ultralytics built-in augmentation
            mosaic        = args.mosaic,
            flipud        = args.flipud,
            fliplr        = args.fliplr,
            degrees       = args.degrees,
            hsv_h         = args.hsv_h,
            hsv_s         = args.hsv_s,
            hsv_v         = args.hsv_v,
            # Runtime
            device        = args.device,
            workers       = args.workers,
            # Output
            project       = args.project,
            name          = run_name,
            save          = True,
            plots         = True,
            val           = True,
            resume        = bool(args.resume),
        )
        trainer = aug.CoralTrainer(overrides=overrides, coral_config=coral_config)
        results = trainer.train()

    else:
        # ── Standard training: dùng model.train() của Ultralytics ────────────────────
        results = model.train(
            data          = str(data_path),
            epochs        = args.epochs,
            imgsz         = args.imgsz,
            batch         = args.batch,
            optimizer     = args.optimizer,
            lr0           = args.lr0,
            lrf           = args.lrf,
            weight_decay  = args.weight_decay,
            warmup_epochs = args.warmup_epochs,
            # Augmentation
            mosaic        = args.mosaic,
            flipud        = args.flipud,
            fliplr        = args.fliplr,
            degrees       = args.degrees,
            hsv_h         = args.hsv_h,
            hsv_s         = args.hsv_s,
            hsv_v         = args.hsv_v,
            # Runtime
            device        = args.device,
            workers       = args.workers,
            # Output
            project       = args.project,
            name          = run_name,
            save          = True,
            plots         = True,
            val           = True,
            resume        = bool(args.resume),
        )

    # ── Summary (chung cho cả hai mode) ───────────────────────────────────────
    rd       = getattr(results, "results_dict", {}) or {}
    map50    = rd.get("metrics/mAP50(B)")
    map50_95 = rd.get("metrics/mAP50-95(B)")

    print("\n" + "=" * 56)
    mode_label = " (CoralAug pipeline)" if args.custom_aug else ""
    print(f"Training complete!{mode_label}")
    print(f"  Best weights : {args.project}/{run_name}/weights/best.pt")
    if map50    is not None: print(f"  mAP@0.5      : {map50:.4f}")
    if map50_95 is not None: print(f"  mAP@0.5:0.95 : {map50_95:.4f}")
    print("=" * 56)
    print(f"\nNext step:")
    print(f"  python scripts/4_evaluate.py --weights {args.project}/{run_name}/weights/best.pt")


if __name__ == "__main__":
    main()
