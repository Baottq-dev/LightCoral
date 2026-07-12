# F3M-YOLO11n (reimplemented)

Reimplement **F3M** (Frequency-domain Feature Fusion Module, Wang et al., *J. Mar. Sci. Eng.* 2026, 14, 20) trên base **YOLO11n** làm đối thủ benchmark cho SC-YOLO12 trên Soft-Coral. F3M là khối *plug-and-play* giữ nguyên kênh, **không thêm loss phụ** — tối ưu hoàn toàn qua detection loss chuẩn.

## Cách chạy (từ root repo `sc-yolo12/`)

```bash
# 1) Smoke-test build (forward 640, kiem 3 head + stride [8,16,32] + params ~2.61M)
python -m benchmark.f3m.build_f3m

# 2) Smoke-test rieng module (params ~0.026M, giu shape)
python -m benchmark.f3m.modules_f3m
```

### Train — tham số cơ bản

```bash
# Chạy mặc định theo paper (seed=42, imgsz=640, batch=16, optimizer=auto, 100ep)
python -m benchmark.f3m.train_f3m --data data/scoraldet_fold0.yaml

# Chỉ định epochs (epochs là tham số linh hoạt duy nhất theo dataset)
python -m benchmark.f3m.train_f3m --data data/scoraldet_fold0.yaml --epochs 200

# Train từ scratch (giống điều kiện gốc paper — không dùng pretrained)
python -m benchmark.f3m.train_f3m --data data/scoraldet_fold0.yaml --scratch

# Chỉ định GPU và số workers
python -m benchmark.f3m.train_f3m --data data/scoraldet_fold0.yaml --device 0 --workers 4
```

### Train — tất cả CLI args (ghi đè hyperparameter paper khi cần)

```bash
python -m benchmark.f3m.train_f3m \
    --data          data/scoraldet_fold0.yaml \  # [BẮT BUỘC] data YAML chứa split cố định
    --seed          42          \  # hạt giống ngẫu nhiên (paper=42, giữ cố định toàn bộ)
    --epochs        100         \  # số epoch — LINH HOẠT theo dataset (mặc định: td["epochs"])
    --imgsz         640         \  # kích thước ảnh đầu vào (paper=640×640)
    --batch         16          \  # kích thước lô (paper=16)
    --optimizer     auto        \  # bộ tối ưu Adam-based (paper=auto)
    --lr0           0.01        \  # tốc độ học ban đầu (paper=0.01)
    --lrf           0.01        \  # tốc độ học cuối cùng (paper=0.01)
    --weight-decay  0.0005      \  # phân rã trọng số (paper=0.0005)
    --momentum      0.937       \  # động lượng (paper=0.937)
    --fliplr        0.5         \  # lật ngang ngẫu nhiên xác suất 50%
    --hsv-h         0.015       \  # biên độ hue HSV ±1.5%
    --hsv-s         0.7         \  # biên độ saturation HSV ±70%
    --hsv-v         0.4         \  # biên độ brightness HSV ±40%
    --scale         0.5         \  # co giãn ngẫu nhiên 0.5–1.0
    --translate     0.1         \  # dịch chuyển ngẫu nhiên tối đa 10%
    --erasing       0.4         \  # random erasing kết hợp RandAugment (p=0.4)
    --weights       yolo11n.pt  \  # pretrained khởi tạo (bỏ qua khi --scratch)
    --device        0           \  # GPU id (mặc định: td["device"])
    --workers       4           \  # số DataLoader workers (mặc định: td["workers"])
    --project       benchmark/runs \  # thư mục lưu kết quả
    --name          F3M         \  # tên run (thực tế = F3M_s<seed>)
    --logfile       path/to/log.txt   # tuỳ chọn: ghi log ra file riêng
```

### Multi-seed (báo mean ± std)

```bash
for s in 0 1 2; do
  python -m benchmark.f3m.train_f3m --data data/scoraldet_fold0.yaml --seed $s
done
```

### Đánh giá trên test split

```bash
# Cơ bản
python -m benchmark.f3m.eval_f3m \
    --data    data/scoraldet_fold0.yaml \
    --weights benchmark/runs/F3M_s42/weights/best.pt

# Đầy đủ args
python -m benchmark.f3m.eval_f3m \
    --data    data/scoraldet_fold0.yaml \
    --weights benchmark/runs/F3M_s42/weights/best.pt \
    --split   test   \   # test | val | train
    --imgsz   640    \
    --batch   16     \
    --device  0      \
    --project benchmark/runs \
    --name    F3M_eval
```

> **Log:** lưu tại `benchmark/runs/F3M_s<seed>/train_log.txt` (tee-log giống SC-YOLO12).  
> **Kết quả eval:** `metrics_test.csv` + `metrics_test.json` trong `benchmark/runs/F3M_eval/`.

### Siêu tham số paper (cố định)

| Tham số | Giá trị | Ghi chú |
|---|---|---|
| `imgsz` | 640 | 640×640 px |
| `batch` | 16 | cố định |
| `optimizer` | auto | Adam-based |
| `lr0` | 0.01 | học ban đầu |
| `lrf` | 0.01 | học cuối |
| `weight_decay` | 0.0005 | phân rã |
| `momentum` | 0.937 | động lượng |
| `seed` | 42 | toàn bộ lượt chạy |
| `epochs` | **linh hoạt** | khác nhau theo dataset |
| `fliplr` | 0.5 | lật ngang 50% |
| `hsv_h/s/v` | 0.015 / 0.7 / 0.4 | HSV augment |
| `scale` | 0.5 | co giãn 0.5–1.0 |
| `translate` | 0.1 | dịch chuyển ≤10% |
| `auto_augment` | randaugment | RandAugment (hardcode) |
| `erasing` | 0.4 | random erasing p=0.4 |

> **Lưu ý process:** registry chỉ patch `parse_model` **một lần** mỗi process. Chạy F3M ở process riêng (đừng import chung SF-YOLO/SCoralDet trong cùng phiên) để module F3M kịp vào `frozenset(CUSTOM_MODULES)` trước khi patch.
> 

## Kiến trúc (Separate–Project–Fuse)

| Stage | Công thức | Ghi chú |
| --- | --- | --- |
| Separate (Eq 1) | `Xlf = AvgPool3x3(X)`, `Xhf = X - Xlf` | low-pass cố định, KHÔNG param |
| Project (Eq 2) | `X~lf = Plf(Xlf)`, `X~hf = Phf(Xhf)` | 2 conv 1×1 riêng, `C → C'=max(8,⌊rC⌋)` |
| Fuse (Eq 3) | `Ymid = Conv1x1(X~lf + X~hf)` | `C' → C` (Upsample nếu ds>1, không dùng) |
| Gate (Eq 4) | `G = σ(Conv1x1([X,Ymid]))`, `Y = X + G⊙Ymid` | gate=False → `Y = X + Ymid` |
| SA (Eq 5–6) | `Ỹ = Y⊙σ(Conv7x7([avg,max]))` | chỉ trong `F3MWithSA`, pool theo kênh |

## Điểm cắm vào YOLO11n (Fig 7)

| Vị trí | Layer | Module | r | gate | SA | kênh |
| --- | --- | --- | --- | --- | --- | --- |
| Stem | idx 1 (sau Conv#0) | `F3MWithSA` | 0.33 | True | Có | 16 |
| Deep | idx 10 (trước SPPF) | `F3M` | 0.125 | False | Không | 256 |

## Đối chiếu paper ↔ code

| Paper | Code |
| --- | --- |
| Eq 1–4 Separate-Project-Fuse | `F3M.forward` (`modules_f3m.py`) |
| Eq 5–6 Spatial Attention | `SpatialAttention`  • `F3MWithSA` |
| Fig 7 vị trí cắm (insert) | `base_f3m_yolo11n.yaml`  • `F3M_NODES` |
| Table 1 ngân sách +0.03M | smoke-test params trong `modules_f3m.py` |

## Mốc verify (Table 1, SCoralDet test split)

| Model | P | R | mAP50 | mAP50-95 | Params | GFLOPs |
| --- | --- | --- | --- | --- | --- | --- |
| YOLO11n (baseline) | 0.763 | 0.686 | 0.762 | 0.513 | 2.58M | 6.3 |
| YOLO11n-F3M (dual) | **0.861** | 0.708 | **0.797** | **0.539** | **2.61M** | **6.5** |

Reimpl đạt **mAP50 ≈ 0.797 / mAP50-95 ≈ 0.539 / ~2.61M / ~6.5 GFLOPs** là khớp. Có thể tái hiện ablation `onlyF3M` (chỉ deep) và `onlyF3MWithSA` (chỉ stem) bằng cách sửa `F3M_NODES` trong `build_f3m.py`.

## Ghi chú fairness

- Dùng **cùng split cố định** và `train_defaults` như các model khác trong `benchmark/` (100ep, optimizer=auto, imgsz640, batch16) — gần như trùng protocol paper F3M.
- F3M **không đổi loss/assigner** nên không cần subclass trainer — rủi ro reimpl thấp hơn SCoralDet nhiều.

> [!WARNING]
> 

> **Phát hiện chéo cho SCoralDet:** paper F3M (§3.4/§3.6) reproduce SCoralDet chỉ đạt 0.724 mAP50 / 0.483 mAP50-95 (so 0.819 paper gốc), quy cho thiếu *"Wasserstein loss"*. Paper SCoralDet gốc (đã đọc) chỉ mô tả MPFB + GSConv/VoVGSCSP + APT — KHÔNG nhắc Wasserstein. Cần rà lại repo SCoralDet xem có NWD/Wasserstein loss ẩn không.
>