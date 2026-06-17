"""
Script 1: Prepare SCoralDet dataset — Convert + Split in one pass.

Convert LabelMe custom JSON annotations to YOLO format (.txt),
then split into train/val/test with stratified sampling per class.

Annotation format (per-image JSON):
  [{"image": "ClassName_N.JPG",
    "annotations": [{"label": "...", "coordinates": {"x": cx, "y": cy, "width": w, "height": h}}]}]
  NOTE: x, y are BBOX CENTER (not top-left). EXIF orientation is handled automatically.

YOLO output format (per image .txt):
  <class_id> <cx_norm> <cy_norm> <w_norm> <h_norm>

Usage examples:
  # Default (70/15/15 split)
  python scripts/1_prepare_dataset.py

  # Custom split ratio  train/val/test
  python scripts/1_prepare_dataset.py --split 0.8/0.1/0.1

  # Custom input/output folders
  python scripts/1_prepare_dataset.py \\
      --ann_dir  data/coral_soft/annotations \\
      --img_root data/coral_soft/image \\
      --out_dir  datasets/coral_soft_yolo

  # All options
  python scripts/1_prepare_dataset.py \\
      --ann_dir  data/coral_soft/annotations \\
      --img_root data/coral_soft/image \\
      --out_dir  datasets/coral_soft_yolo \\
      --split 0.7/0.15/0.15 \\
      --seed 42
"""

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image
from PIL.ExifTags import TAGS
from tqdm import tqdm


# ── Constants ──────────────────────────────────────────────────────────────────
CLASS_NAMES = [
    "Euphflfiaancora",
    "Favosites",
    "Platygyra",
    "Sarcophyton",
    "Sinularia",
    "WavingHand",
]
CLASS_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}
# ───────────────────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert SCoralDet annotations to YOLO format and split dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ── Input / Output ──────────────────────────────────────────────────────────
    parser.add_argument(
        "--ann_dir",
        type=Path,
        default=Path("data/coral_soft/annotations"),
        help="Directory containing per-image JSON annotation files.",
    )
    parser.add_argument(
        "--img_root",
        type=Path,
        default=Path("data/coral_soft/image"),
        help="Root directory containing per-class image subfolders.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("datasets/coral_soft_yolo"),
        help="Output root directory (images/ and labels/ will be created here).",
    )
    # ── Split ratios ────────────────────────────────────────────────────────────
    parser.add_argument(
        "--split",
        type=str,
        default="0.7/0.15/0.15",
        metavar="TRAIN/VAL/TEST",
        help="Split ratios as 'train/val/test', must sum to 1.0. Example: 0.8/0.1/0.1",
    )
    # ── Misc ────────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible splits.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite output directory if it already exists.",
    )

    args = parser.parse_args()

    # Parse and validate --split
    try:
        parts = [float(x) for x in args.split.split("/")]
        if len(parts) != 3:
            raise ValueError
    except ValueError:
        parser.error("--split must be in 'TRAIN/VAL/TEST' format, e.g. 0.7/0.15/0.15")

    args.train, args.val, args.test = parts
    total = args.train + args.val + args.test
    if not (0.99 < total < 1.01):
        parser.error(f"--split values must sum to 1.0 (got {total:.3f}: {args.split})")

    return args


# ── Helpers ────────────────────────────────────────────────────────────────────

def get_image_info(path: Path) -> tuple[int, int, int]:
    """
    Return (effective_width, effective_height, orientation) using Pillow.
    For orientation 6/8 (90-degree rotations), width and height are swapped
    to match the coordinate space used by the annotation tool.
    """
    with Image.open(path) as img:
        raw_w, raw_h = img.size
        exif_data = img._getexif() or {}

    orientation = 1
    for tag_id, val in exif_data.items():
        if TAGS.get(tag_id) == "Orientation":
            orientation = val
            break

    # Swap dims if camera stored the image rotated 90 degrees
    if orientation in (6, 8):
        return raw_h, raw_w, orientation
    return raw_w, raw_h, orientation


def find_image(stem: str, img_root: Path) -> Path | None:
    """Search for an image file (any extension) across all class subfolders."""
    for ext in (".JPG", ".jpg", ".jpeg", ".png", ".PNG"):
        for class_name in CLASS_NAMES:
            candidate = img_root / class_name / (stem + ext)
            if candidate.exists():
                return candidate
    return None


def get_primary_class(stem: str) -> str:
    """Infer the primary class from the image filename stem (e.g. 'Sinularia_12')."""
    for class_name in CLASS_NAMES:
        if stem.startswith(class_name):
            return class_name
    return "unknown"


def convert_record(record: dict, img_path: Path) -> list[str] | None:
    """
    Convert one annotation record to YOLO label lines.
    Returns a list of strings (one per bbox), or None on error.
    """
    eff_w, eff_h, orientation = get_image_info(img_path)
    lines = []

    for ann in record["annotations"]:
        label = ann["label"]
        if label not in CLASS_MAP:
            print(f"  [WARN] Unknown class '{label}' — skipped")
            continue

        class_id = CLASS_MAP[label]
        cx = ann["coordinates"]["x"]
        cy = ann["coordinates"]["y"]
        bw = ann["coordinates"]["width"]
        bh = ann["coordinates"]["height"]

        # Normalize using effective (post-EXIF) dimensions
        cx_n = max(0.0, min(1.0, cx / eff_w))
        cy_n = max(0.0, min(1.0, cy / eff_h))
        bw_n = max(0.0, min(1.0, bw / eff_w))
        bh_n = max(0.0, min(1.0, bh / eff_h))

        lines.append(f"{class_id} {cx_n:.6f} {cy_n:.6f} {bw_n:.6f} {bh_n:.6f}")

    return lines


def write_pair(img_path: Path, label_lines: list[str], split: str, out_dir: Path):
    """Copy image and write label .txt into the appropriate split directory."""
    img_dst = out_dir / "images" / split / img_path.name
    lbl_dst = out_dir / "labels" / split / (img_path.stem + ".txt")

    img_dst.parent.mkdir(parents=True, exist_ok=True)
    lbl_dst.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy2(img_path, img_dst)
    lbl_dst.write_text("\n".join(label_lines), encoding="utf-8")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    random.seed(args.seed)

    # ── Validate output dir ────────────────────────────────────────────────────
    if args.out_dir.exists() and not args.overwrite:
        print(f"[ERROR] Output directory already exists: {args.out_dir}")
        print("        Use --overwrite to replace it.")
        return
    if args.out_dir.exists() and args.overwrite:
        shutil.rmtree(args.out_dir)

    # ── Load and group annotations by primary class ────────────────────────────
    json_files = sorted(args.ann_dir.glob("*.json"))
    print(f"Found {len(json_files)} annotation files in {args.ann_dir}")

    # class_name → list of (img_path, record)
    class_groups: dict[str, list] = defaultdict(list)
    skipped = 0

    for json_path in json_files:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        record = data[0]
        stem = Path(record["image"]).stem

        img_path = find_image(stem, args.img_root)
        if img_path is None:
            print(f"  [WARN] Image not found for {json_path.name} — skipped")
            skipped += 1
            continue

        primary_class = get_primary_class(stem)
        class_groups[primary_class].append((img_path, record))

    print(f"Loaded {sum(len(v) for v in class_groups.values())} records "
          f"({skipped} skipped)\n")

    # ── Stratified split ───────────────────────────────────────────────────────
    splits: dict[str, list] = {"train": [], "val": [], "test": []}

    print(f"Split ratios — train:{args.train:.0%}  val:{args.val:.0%}  test:{args.test:.0%}")
    print("-" * 52)

    for class_name in CLASS_NAMES:
        pairs = class_groups.get(class_name, [])
        random.shuffle(pairs)
        n       = len(pairs)
        n_train = round(n * args.train)
        n_val   = round(n * args.val)
        # test = remainder (avoids rounding drift)
        n_test  = n - n_train - n_val

        splits["train"].extend(pairs[:n_train])
        splits["val"].extend(pairs[n_train:n_train + n_val])
        splits["test"].extend(pairs[n_train + n_val:])

        print(f"  {class_name:<22}  total={n:>3}  "
              f"train={n_train:>3}  val={n_val:>3}  test={n_test:>3}")

    # ── Convert + write ────────────────────────────────────────────────────────
    print()
    stats = {"total": 0, "bbox_total": 0, "errors": 0}

    for split_name, pairs in splits.items():
        print(f"Writing {split_name} ({len(pairs)} images)...")
        for img_path, record in tqdm(pairs, desc=f"  {split_name}", leave=False):
            lines = convert_record(record, img_path)
            if lines is None:
                stats["errors"] += 1
                continue
            write_pair(img_path, lines, split_name, args.out_dir)
            stats["total"] += 1
            stats["bbox_total"] += len(lines)

    # ── Summary ────────────────────────────────────────────────────────────────
    total = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
    print(f"""
{'='*52}
Done.
  Images written : {stats['total']}
  Total bbox     : {stats['bbox_total']}
  Errors         : {stats['errors']}
  Output         : {args.out_dir}

  Split summary  :
    train  {len(splits['train']):>4}  ({len(splits['train'])/total*100:.1f}%)
    val    {len(splits['val']):>4}  ({len(splits['val'])/total*100:.1f}%)
    test   {len(splits['test']):>4}  ({len(splits['test'])/total*100:.1f}%)
{'='*52}
Next step:
  python scripts/3_train_baseline.py --model yolov8s --epochs 100
""")


if __name__ == "__main__":
    main()
