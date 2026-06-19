"""
Script 2: Prepare UTDAC2020 dataset — Convert COCO JSON to YOLO format.

Convert COCO-format annotations (instances_train2017.json, instances_val2017.json)
to YOLO detection format (.txt per image), then organise into train/val/test splits.

Strategy:
  - train2017 → train  (giữ nguyên)
  - val2017   → chia 1:1 thành val + test (shuffle + seed)

COCO bbox: [x_min, y_min, width, height]  (pixel, top-left origin)
YOLO bbox:  <class_id> <cx_norm> <cy_norm> <w_norm> <h_norm>

Usage:
  # Default (output to datasets/utdac2020_yolo)
  python benchmark/scripts/2_prepare_utdac2020.py

  # Custom output
  python benchmark/scripts/2_prepare_utdac2020.py --out_dir datasets/my_utdac

  # Overwrite existing
  python benchmark/scripts/2_prepare_utdac2020.py --overwrite
"""

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm


# ── Constants ──────────────────────────────────────────────────────────────────
# UTDAC2020 4-class mapping (COCO category_id → YOLO class_id)
COCO_CAT_TO_YOLO = {
    1: 0,  # echinus
    2: 1,  # starfish
    3: 2,  # holothurian
    4: 3,  # scallop
}

CLASS_NAMES = ["echinus", "starfish", "holothurian", "scallop"]
# ───────────────────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert UTDAC2020 COCO annotations to YOLO format and split dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("data/UTDAC2020"),
        help="Root directory of UTDAC2020 dataset (contains train2017/, val2017/, annotations/).",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("datasets/utdac2020_yolo"),
        help="Output root directory (images/ and labels/ will be created here).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible val/test split.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite output directory if it already exists.",
    )
    return parser.parse_args()


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_coco(json_path: Path) -> tuple[dict, list, list]:
    """Load COCO JSON and return (id2img, annotations, categories)."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    id2img = {img["id"]: img for img in data["images"]}
    return id2img, data["annotations"], data["categories"]


def build_image_annotations(annotations: list) -> dict[int, list]:
    """Group annotations by image_id."""
    img_anns = defaultdict(list)
    for ann in annotations:
        img_anns[ann["image_id"]].append(ann)
    return img_anns


def coco_bbox_to_yolo(bbox: list, img_w: int, img_h: int) -> tuple[float, float, float, float]:
    """Convert COCO bbox [x_min, y_min, w, h] to YOLO [cx, cy, w, h] normalised."""
    x_min, y_min, bw, bh = bbox
    cx = (x_min + bw / 2.0) / img_w
    cy = (y_min + bh / 2.0) / img_h
    w_n = bw / img_w
    h_n = bh / img_h
    # Clamp to [0, 1]
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    w_n = max(0.0, min(1.0, w_n))
    h_n = max(0.0, min(1.0, h_n))
    return cx, cy, w_n, h_n


def convert_and_write(
    image_ids: list[int],
    id2img: dict,
    img_anns: dict[int, list],
    src_img_dir: Path,
    out_dir: Path,
    split_name: str,
) -> dict:
    """Convert annotations and copy images for a given split. Returns stats."""
    img_dst_dir = out_dir / "images" / split_name
    lbl_dst_dir = out_dir / "labels" / split_name
    img_dst_dir.mkdir(parents=True, exist_ok=True)
    lbl_dst_dir.mkdir(parents=True, exist_ok=True)

    stats = {"images": 0, "bboxes": 0, "skipped_anns": 0, "missing_imgs": 0}

    for img_id in tqdm(image_ids, desc=f"  {split_name}", leave=False):
        img_info = id2img[img_id]
        file_name = img_info["file_name"]
        img_w, img_h = img_info["width"], img_info["height"]

        src_path = src_img_dir / file_name
        if not src_path.exists():
            stats["missing_imgs"] += 1
            continue

        # Convert annotations for this image
        lines = []
        for ann in img_anns.get(img_id, []):
            cat_id = ann["category_id"]
            if cat_id not in COCO_CAT_TO_YOLO:
                stats["skipped_anns"] += 1
                continue
            if ann.get("iscrowd", 0):
                stats["skipped_anns"] += 1
                continue

            yolo_cls = COCO_CAT_TO_YOLO[cat_id]
            cx, cy, w_n, h_n = coco_bbox_to_yolo(ann["bbox"], img_w, img_h)
            lines.append(f"{yolo_cls} {cx:.6f} {cy:.6f} {w_n:.6f} {h_n:.6f}")

        # Copy image
        shutil.copy2(src_path, img_dst_dir / file_name)

        # Write label (empty file if no annotations — YOLO uses this for negative samples)
        lbl_path = lbl_dst_dir / (Path(file_name).stem + ".txt")
        lbl_path.write_text("\n".join(lines), encoding="utf-8")

        stats["images"] += 1
        stats["bboxes"] += len(lines)

    return stats


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    random.seed(args.seed)

    data_dir = args.data_dir
    out_dir = args.out_dir

    # ── Validate paths ─────────────────────────────────────────────────────────
    assert (data_dir / "annotations").exists(), f"Annotations dir not found: {data_dir / 'annotations'}"
    assert (data_dir / "train2017").exists(), f"Train images not found: {data_dir / 'train2017'}"
    assert (data_dir / "val2017").exists(), f"Val images not found: {data_dir / 'val2017'}"

    if out_dir.exists() and not args.overwrite:
        print(f"[ERROR] Output directory already exists: {out_dir}")
        print("        Use --overwrite to replace it.")
        return
    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)

    # ── Load COCO annotations ─────────────────────────────────────────────────
    print("Loading train annotations...")
    train_id2img, train_anns, categories = load_coco(
        data_dir / "annotations" / "instances_train2017.json"
    )
    train_img_anns = build_image_annotations(train_anns)

    print("Loading val annotations...")
    val_id2img, val_anns, _ = load_coco(
        data_dir / "annotations" / "instances_val2017.json"
    )
    val_img_anns = build_image_annotations(val_anns)

    # Print dataset info
    cat_names = {c["id"]: c["name"] for c in categories}
    print(f"\nDataset: UTDAC2020")
    print(f"Categories: {cat_names}")
    print(f"Train: {len(train_id2img)} images, {len(train_anns)} annotations")
    print(f"Val (original): {len(val_id2img)} images, {len(val_anns)} annotations")

    # ── Split val → val + test (1:1) ──────────────────────────────────────────
    val_image_ids = sorted(val_id2img.keys())
    random.shuffle(val_image_ids)
    mid = len(val_image_ids) // 2
    new_val_ids = val_image_ids[:mid]
    new_test_ids = val_image_ids[mid:]

    print(f"\nSplitting original val → val ({len(new_val_ids)}) + test ({len(new_test_ids)})")
    print("-" * 52)

    # ── Convert train ──────────────────────────────────────────────────────────
    train_image_ids = sorted(train_id2img.keys())
    print(f"\nWriting train ({len(train_image_ids)} images)...")
    train_stats = convert_and_write(
        train_image_ids, train_id2img, train_img_anns,
        data_dir / "train2017", out_dir, "train",
    )

    # ── Convert val ────────────────────────────────────────────────────────────
    print(f"\nWriting val ({len(new_val_ids)} images)...")
    val_stats = convert_and_write(
        new_val_ids, val_id2img, val_img_anns,
        data_dir / "val2017", out_dir, "val",
    )

    # ── Convert test ───────────────────────────────────────────────────────────
    print(f"\nWriting test ({len(new_test_ids)} images)...")
    test_stats = convert_and_write(
        new_test_ids, val_id2img, val_img_anns,
        data_dir / "val2017", out_dir, "test",
    )

    # ── Summary ────────────────────────────────────────────────────────────────
    total_imgs = train_stats["images"] + val_stats["images"] + test_stats["images"]
    total_bboxes = train_stats["bboxes"] + val_stats["bboxes"] + test_stats["bboxes"]
    print(f"""
{'='*52}
Done.
  Images written : {total_imgs}
  Total bbox     : {total_bboxes}
  Missing images : {train_stats['missing_imgs'] + val_stats['missing_imgs'] + test_stats['missing_imgs']}
  Skipped anns   : {train_stats['skipped_anns'] + val_stats['skipped_anns'] + test_stats['skipped_anns']}
  Output         : {out_dir}

  Split summary  :
    train  {train_stats['images']:>5}  ({train_stats['bboxes']} bboxes)
    val    {val_stats['images']:>5}  ({val_stats['bboxes']} bboxes)
    test   {test_stats['images']:>5}  ({test_stats['bboxes']} bboxes)
{'='*52}
Next step:
  python train.py --data cfg/utdac2020.yaml --modules "" --epochs 100
""")


if __name__ == "__main__":
    main()
