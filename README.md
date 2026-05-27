# LightCoral-YOLO — SCoralDet Benchmark

Benchmark pipeline for the **SCoralDet** dataset (soft coral detection) using the Ultralytics YOLO family.

---

## Dataset — SCoralDet

| Property | Value |
|---|---|
| Total images | 646 |
| Total bounding boxes | 2,199 |
| Classes | 6 |
| Annotation format | LabelMe custom JSON (center-based bbox) |
| Image sizes | Mixed (e.g. 6000×4000, 3264×2448, …) |
| EXIF orientation | 606 normal · 19 rotate-90°CW · 21 rotate-90°CCW |

### Class distribution

| ID | Class | Bbox count |
|---|---|---|
| 0 | Euphflfiaancora | 408 |
| 1 | Favosites | 221 |
| 2 | Platygyra | 185 |
| 3 | Sarcophyton | 339 |
| 4 | Sinularia | 761 |
| 5 | WavingHand | 285 |

### Directory layout (raw)

```
data/coral_soft/
├── annotations/        # 646 per-image JSON files
│   ├── Euphflfiaancora_1.json
│   └── ...
└── image/
    ├── Euphflfiaancora/   112 images
    ├── Favosites/         107 images
    ├── Platygyra/         103 images
    ├── Sarcophyton/       111 images
    ├── Sinularia/         110 images
    └── WavingHand/        103 images
```

---

## Installation

### Cài tất cả 1 lần

```bash
pip install -r requirements.txt
```

`requirements.txt` đã bao gồm `--extra-index-url` trỏ vào pytorch.org và pin sẵn `torch==2.7.1+cu128` — pip sẽ tự kéo bản GPU đúng cho **RTX 4060 (CUDA 12.8)**.

Kiểm tra GPU sau khi cài:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expected: True  NVIDIA GeForce RTX 4060 Laptop GPU
```

> **Máy khác / CUDA version khác:** Đổi `cu128` → đúng version trên [pytorch.org](https://pytorch.org/get-started/locally/) rồi sửa trong `requirements.txt`.

---

## Pipeline

### Step 0 — Analyze dataset (optional)

```bash
python scripts/0_analyze_dataset.py
```

Outputs: class distribution, EXIF orientation counts, out-of-bounds bbox stats.

---

### Step 1 — Prepare dataset (convert + split)

Converts LabelMe JSON annotations → YOLO `.txt` labels and splits the data into train/val/test.

```bash
# Default: 70% train / 15% val / 15% test
python scripts/1_prepare_dataset.py

# Custom split
python scripts/1_prepare_dataset.py --split 0.8/0.1/0.1

# All options
python scripts/1_prepare_dataset.py \
    --ann_dir  data/coral_soft/annotations \
    --img_root data/coral_soft/image \
    --out_dir  datasets/coral_soft_yolo \
    --split    0.7/0.15/0.15 \
    --seed     42 \
    --overwrite
```

**Output structure:**

```
datasets/coral_soft_yolo/
├── images/
│   ├── train/   452 images
│   ├── val/      97 images
│   └── test/     97 images
└── labels/
    ├── train/   452 .txt files
    ├── val/      97 .txt files
    └── test/     97 .txt files
```

---

### Step 2 — Train baseline

```bash
# Quick start (yolov8s, 100 epochs, 640px)
python scripts/3_train_baseline.py

# Larger model at higher resolution
python scripts/3_train_baseline.py --model yolov8m --imgsz 1280 --epochs 150

# Resume interrupted run
python scripts/3_train_baseline.py --resume runs/coral_benchmark/yolov8s_ep100/weights/last.pt
```

**Supported models (ultralytics hub — auto-downloaded):**

| Family | Variants |
|---|---|
| YOLOv8 | `yolov8n` `yolov8s` `yolov8m` `yolov8l` `yolov8x` |
| YOLOv9 | `yolov9c` `yolov9e` |
| YOLOv10 | `yolov10n` `yolov10s` `yolov10m` `yolov10b` `yolov10l` `yolov10x` |
| YOLO11 | `yolo11n` `yolo11s` `yolo11m` `yolo11l` `yolo11x` |
| YOLO12 | `yolo12n` `yolo12s` `yolo12m` `yolo12l` `yolo12x` |

All pretrained weights are downloaded automatically on first run.

**Custom / local model (e.g. YOLO26):**

YOLO26 is not in the ultralytics hub — use `--weights` to point to your local `.pt`:

```bash
python scripts/3_train_baseline.py \
    --weights f:/YOLO26-SELF_DISTI/weights/yolo26s.pt \
    --epochs 100 --imgsz 640
```

---

### Step 2b — Multi-model Benchmark (PowerShell)

Chạy nhiều model liên tiếp, tự động train → evaluate từng model.

**Lần đầu tiên** — cho phép chạy script PS1 (chỉ cần làm 1 lần):
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Chạy benchmark:**
```powershell
# Mặc định: tất cả model trong $Models list, 100 epochs, 640px, train+eval
.\scripts\run_benchmark.ps1

# Xem lệnh sẽ chạy mà không thực thi (kiểm tra trước)
.\scripts\run_benchmark.ps1 -DryRun

# Custom params
.\scripts\run_benchmark.ps1 -Epochs 50 -ImgSz 1280 -Batch 8

# Evaluate trên val thay vì test
.\scripts\run_benchmark.ps1 -EvalSplit val

# Chỉ train, bỏ qua evaluate
.\scripts\run_benchmark.ps1 -SkipEval
```

**Chọn model** — mở `scripts/run_benchmark.ps1` và bỏ comment:
```powershell
$Models = @(
    "yolov8n",
    "yolov8s",
    # "yolov8m",          # ← comment = bỏ qua
    "yolo11s",
    "yolo12s",
    # YOLO26 (custom — phải dùng __custom__:path):
    # "__custom__:f:/YOLO26-SELF_DISTI/weights/yolo26s.pt"
)
```

> ⚠️ **YOLO26** không có trên ultralytics hub. Phải dùng `"__custom__:path/to/yolo26s.pt"`, không được ghi thẳng `"yolo26n"`.

**Output sau khi chạy xong:**
```
runs/coral_benchmark/
├── yolov8n_imgsz640_ep100/weights/best.pt
├── yolo11s_imgsz640_ep100/weights/best.pt
└── _logs/
    ├── summary_20260521_192300.txt   ← bảng tổng hợp
    ├── yolov8n_train_*.txt           ← log train từng model
    ├── yolov8n_eval_*.txt            ← log eval từng model
    └── eval_yolov8n_*_test.json      ← metrics JSON
```

Bảng tổng hợp cuối chạy:
```
Model      Train           Eval    mAP50   Duration
------     ------          ----    -----   --------
yolov8n    OK (18.3m)      OK      0.7821  19.1m
yolo11s    OK (21.5m)      OK      0.8043  22.4m
```

---

### Step 3 — Evaluate (single model)

```bash
python scripts/4_evaluate.py \
    --weights runs/coral_benchmark/yolov8s_imgsz640_ep100/weights/best.pt

# Evaluate on val split instead of test
python scripts/4_evaluate.py --weights ... --split val
```

Output: per-class AP@0.5, mAP@0.5, mAP@0.5:0.95, Precision, Recall — saved as JSON.

---

## Full CLI Reference

### `1_prepare_dataset.py`

| Argument | Default | Description |
|---|---|---|
| `--ann_dir` | `data/coral_soft/annotations` | JSON annotation directory |
| `--img_root` | `data/coral_soft/image` | Image root (with class subfolders) |
| `--out_dir` | `datasets/coral_soft_yolo` | Output directory |
| `--split` | `0.7/0.15/0.15` | Train/val/test ratios (must sum to 1) |
| `--seed` | `42` | Random seed |
| `--overwrite` | `False` | Overwrite existing output |

### `3_train_baseline.py`

| Argument | Default | Description |
|---|---|---|
| `--model` | `yolov8s` | Model alias (see table above) |
| `--resume` | `None` | Path to `last.pt` to resume |
| `--data` | `configs/coral_soft.yaml` | Dataset YAML |
| `--epochs` | `100` | Training epochs |
| `--imgsz` | `640` | Input image size |
| `--batch` | `16` | Batch size (`-1` = auto) |
| `--optimizer` | `AdamW` | Optimizer |
| `--lr0` | `0.001` | Initial learning rate |
| `--device` | `0` | CUDA device or `cpu` |
| `--workers` | `0` | Dataloader workers (0 = safe on Windows) |
| `--project` | `runs/coral_benchmark` | Output root |
| `--name` | auto | Run name |

### `4_evaluate.py`

| Argument | Default | Description |
|---|---|---|
| `--weights` | *(required)* | Path to `best.pt` |
| `--split` | `test` | Dataset split to evaluate |
| `--imgsz` | `640` | Inference image size |
| `--conf` | `0.25` | Confidence threshold |
| `--iou` | `0.5` | IoU threshold |

---

## Project Structure

```
LightCoral-YOLO/
├── data/
│   └── coral_soft/               # Raw dataset (annotations + images)
├── datasets/
│   └── coral_soft_yolo/          # Prepared YOLO dataset (generated)
├── configs/
│   └── coral_soft.yaml           # Ultralytics dataset config
├── scripts/
│   ├── 0_analyze_dataset.py      # Dataset statistics & validation
├── ├── 1_prepare_dataset.py      # Convert + split pipeline
│   ├── 3_train_baseline.py       # YOLO training script (single model)
│   ├── 4_evaluate.py             # Evaluation & metrics export
│   └── run_benchmark.ps1         # Multi-model benchmark runner (PowerShell)
├── runs/                         # Training outputs (generated)
├── coco2yolo.py                  # Original author converter (COCO format)
├── requirements.txt
└── README.md
```

---

## Notes

- **EXIF orientation** is handled automatically in `1_prepare_dataset.py`. Images with orientation 6/8 (40 images) have their effective width/height swapped before normalizing bbox coordinates.
- **Imbalanced classes**: Sinularia has 4× more bboxes than Platygyra. Consider class-weighted loss or oversampling if needed.
- **Small dataset**: Only 646 images. Pretrained COCO weights + strong augmentation are important to avoid overfitting.
- **Windows**: `--workers 0` is default to avoid multiprocessing issues with PyTorch DataLoader on Windows.
