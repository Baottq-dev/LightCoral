"""
Script phan tich SCoralDet: xac nhan cac thong ke ve dataset.
Kiem tra: format, so anh/bbox, phan bo class, bbox out-of-bounds, EXIF orientation.
"""
import json
from pathlib import Path
from collections import defaultdict, Counter
from PIL import Image
from PIL.ExifTags import TAGS

ANNOTATION_DIR = Path(r"f:/LightCoral-YOLO/data/coral_soft/annotations")
IMAGE_ROOT     = Path(r"f:/LightCoral-YOLO/data/coral_soft/image")

CLASS_NAMES = ["Euphflfiaancora", "Favosites", "Platygyra", "Sarcophyton", "Sinularia", "WavingHand"]


# ── 1. Đọc tất cả annotation ──────────────────────────────────────────────────
all_records = []
for json_path in sorted(ANNOTATION_DIR.glob("*.json")):
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    record = data[0]
    all_records.append(record)

print(f"[1] Số file annotation : {len(all_records)}")
print(f"[1] Format             : LabelMe custom JSON (không phải COCO chuẩn)")


# ── 2. Đếm ảnh + bbox ─────────────────────────────────────────────────────────
total_bbox = 0
class_bbox_count = defaultdict(int)

for rec in all_records:
    for ann in rec["annotations"]:
        total_bbox += 1
        class_bbox_count[ann["label"]] += 1

print(f"\n[2] Số ảnh (records)   : {len(all_records)}")
print(f"[2] Tổng bbox          : {total_bbox}")
print(f"\n[3] Phân bố bbox per class:")
for cls in CLASS_NAMES:
    print(f"    {cls:<22}: {class_bbox_count[cls]:>4}")
print(f"    {'TOTAL':<22}: {sum(class_bbox_count.values()):>4}")


# ── 3. Đọc EXIF orientation + kích thước thực ─────────────────────────────────
def get_jpeg_dims_and_exif(path: Path):
    """Dung Pillow de doc kich thuoc raw va EXIF orientation."""
    with Image.open(path) as img:
        raw_w, raw_h = img.size  # Pillow tra ve (W, H) theo raw storage
        exif_data = img._getexif() or {}
    orientation = 1
    for tag_id, val in exif_data.items():
        if TAGS.get(tag_id) == "Orientation":
            orientation = val
            break
    return raw_w, raw_h, orientation


# Ánh xạ image filename → image path
image_map = {}
for cls in CLASS_NAMES:
    for img_path in (IMAGE_ROOT / cls).glob("*.JPG"):
        image_map[img_path.name] = img_path

print(f"\n[4] Đang đọc EXIF + kích thước {len(all_records)} ảnh...")

orientation_counter = Counter()
oob_raw = 0      # Out-of-bounds khi dùng raw size (bỏ qua EXIF rotation)
oob_exif = 0     # Out-of-bounds sau khi áp dụng EXIF orientation

oob_raw_examples = []
oob_exif_examples = []

for rec in all_records:
    img_name = rec["image"]
    if img_name not in image_map:
        continue

    raw_w, raw_h, orient = get_jpeg_dims_and_exif(image_map[img_name])
    orientation_counter[orient] += 1

    # Nếu orient 6 hoặc 8 → ảnh bị xoay 90°, width/height đổi chỗ
    if orient in (6, 8):
        eff_w, eff_h = raw_h, raw_w
    else:
        eff_w, eff_h = raw_w, raw_h

    for ann in rec["annotations"]:
        cx = ann["coordinates"]["x"]
        cy = ann["coordinates"]["y"]
        bw = ann["coordinates"]["width"]
        bh = ann["coordinates"]["height"]

        # Tính bbox edges
        x1 = cx - bw / 2
        y1 = cy - bh / 2
        x2 = cx + bw / 2
        y2 = cy + bh / 2

        # Check với raw size
        if x1 < 0 or y1 < 0 or x2 > raw_w or y2 > raw_h:
            oob_raw += 1
            if len(oob_raw_examples) < 3:
                oob_raw_examples.append(f"{img_name}: bbox=[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}] vs raw=({raw_w},{raw_h})")

        # Check với effective size (sau EXIF)
        if x1 < 0 or y1 < 0 or x2 > eff_w or y2 > eff_h:
            oob_exif += 1
            if len(oob_exif_examples) < 3:
                oob_exif_examples.append(f"{img_name} [orient={orient}]: bbox=[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}] vs eff=({eff_w},{eff_h})")


print(f"\n[4] EXIF Orientation distribution:")
for orient, count in sorted(orientation_counter.items()):
    tag = ""
    if orient == 1: tag = "(normal)"
    elif orient == 6: tag = "(rotate 90° CW)"
    elif orient == 8: tag = "(rotate 90° CCW)"
    print(f"    Orientation {orient} {tag}: {count} ảnh")

print(f"\n[4] Bbox out-of-bounds (raw size)       : {oob_raw}")
for ex in oob_raw_examples:
    print(f"    → {ex}")

print(f"\n[4] Bbox out-of-bounds (sau EXIF orient): {oob_exif}")
for ex in oob_exif_examples:
    print(f"    → {ex}")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"  Total images     : {len(all_records)}")
print(f"  Total bbox       : {total_bbox}")
print(f"  EXIF orient != 1 : {sum(v for k,v in orientation_counter.items() if k != 1)} ảnh")
print(f"  OOB (raw)        : {oob_raw}")
print(f"  OOB (post-EXIF)  : {oob_exif}")
