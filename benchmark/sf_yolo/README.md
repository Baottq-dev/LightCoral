<aside>
🐟

**SF-YOLO faithful** reimplement trên base **YOLOv11n** để benchmark đối chiếu với SC-YOLO12 trên SCoralDet. **Tái dùng** nguyên lớp `SFDF`/`AMCF` và `models/registry.py`; chỉ khác base (v11n: C3k2 + SPPF + C2PSA).

</aside>

## Đối chiếu paper ↔ code (3 đóng góp)

| Đóng góp SF-YOLO | Paper | Lớp tái dùng | Lắp vào đâu |
| --- | --- | --- | --- |
| Shallow design (bỏ downsample cuối + 2 head) | §3.2 | `AMCF` (`models/shallow_p2.py`) | idx 7 (stride 1) + neck 2-head trong YAML |
| SFDF (Spatial-Frequency Dual-domain Fusion) | §3.3, Eq 4-11 | `SFDF` (`models/sfdf.py`) | idx 2/4/6/8 (swap 6/8) |
| AMCF (Adaptive Multi-context Fusion) | §3.4, Eq 12-15 | `AMCF` (`models/shallow_p2.py`) | idx 7 |

<aside>
🧩

SFDF nội bộ dùng lại `ChannelAttention`, `DSA`, `PinwheelConv`, `HaarDWT`/`HaarIDWT` (`models/common.py`); AMCF dùng `_StarBlock` + `_PinwheelPConv` (`models/shallow_p2.py`). **Không viết mới module nào** — chỉ thêm YAML v11n + builder + train wrapper.

</aside>

## Kiến trúc `sf_yolo11n`

- **Backbone v11n:** Conv×2 → C3k2(→SFDF#1)@2 → Conv → C3k2(→SFDF#2)@4 → Conv → C3k2(→SFDF#3,swap)@6 → **Conv@7(→AMCF stride 1)** → C3k2(→SFDF#4,swap)@8 → SPPF@9 → C2PSA@10.
- **Shallow:** AMCF stride 1 bỏ downsample cuối ⇒ backbone dừng ở **stride 16** (640→320→160→80→40).
- **Neck:** top-down vươn tới stride 4 (160×160), **fuse layer 2 & 4**; PAN trả **2 head** P4 @80×80 (stride 8) + P5 @40×40 (stride 16) ⇒ `Detect.stride == [8, 16]`.

## Cách chạy (từ root repo `sc-yolo12/`)

```bash
# 1) Smoke-test build: forward 640, kiem 2 head + stride [8,16] + NaN, in params
python -m benchmark.sf_yolo.build_sf_yolo
```

### Train — tham số cơ bản

```bash
# Chạy mặc định theo paper (imgsz=736, batch=16, SGD, 300ep, seed=0)
python -m benchmark.sf_yolo.train_sf_yolo --data data/scoraldet_fold0.yaml

# Dùng seed khác
python -m benchmark.sf_yolo.train_sf_yolo --data data/scoraldet_fold0.yaml --seed 1

# Train từ scratch (giống điều kiện gốc paper — không pretrained)
python -m benchmark.sf_yolo.train_sf_yolo --data data/scoraldet_fold0.yaml --seed 0 --scratch

# Chỉ định GPU
python -m benchmark.sf_yolo.train_sf_yolo --data data/scoraldet_fold0.yaml --device 0
```

### Train — tất cả CLI args (ghi đè hyperparameter paper khi cần)

```bash
python -m benchmark.sf_yolo.train_sf_yolo \
    --data         data/scoraldet_fold0.yaml \  # [BẮT BUỘC] data YAML với split cố định
    --seed         0            \  # hạt giống (mặc định=0; dùng 0/1/2 cho multi-seed)
    --epochs       300          \  # số epoch (paper=300, cố định)
    --imgsz        736          \  # kích thước ảnh đầu vào (paper=736×736, cố định)
    --batch        16           \  # kích thước lô (paper=16, cố định)
    --optimizer    SGD          \  # bộ tối ưu (paper=SGD, cố định)
    --lr0          0.01         \  # tốc độ học ban đầu (paper=0.01)
    --lrf          0.01         \  # tốc độ học cuối cùng (paper=0.01)
    --weight-decay 0.0005       \  # phân rã trọng số (paper=0.0005)
    --momentum     0.937        \  # động lượng SGD (paper=0.937)
    --weights      yolo11n.pt   \  # pretrained khởi tạo — stem + vài Conv khớp tên
    --scratch                   \  # flag: train từ đầu (bỏ --weights)
    --device       0            \  # GPU id (mặc định: td["device"])
    --workers      4            \  # số DataLoader workers
    --project      benchmark/runs \  # thư mục lưu kết quả
    --name         SF_YOLO      \  # tên run (thực tế = SF_YOLO_s<seed>)
    --logfile      path/to/log.txt   # tuỳ chọn: ghi log ra file riêng
```

### Multi-seed (báo mean ± std)

```bash
for s in 0 1 2; do
  python -m benchmark.sf_yolo.train_sf_yolo --data data/scoraldet_fold0.yaml --seed $s
done
```

### Đánh giá trên test split

```bash
# Cơ bản
python -m benchmark.sf_yolo.eval_sf_yolo \
    --data    data/scoraldet_fold0.yaml \
    --weights benchmark/runs/SF_YOLO_s0/weights/best.pt

# Đầy đủ args
python -m benchmark.sf_yolo.eval_sf_yolo \
    --data    data/scoraldet_fold0.yaml \
    --weights benchmark/runs/SF_YOLO_s0/weights/best.pt \
    --split   test   \   # test | val | train
    --imgsz   640    \   # dùng 640 khi eval dù train ở 736 (tránh OOM)
    --batch   16     \
    --device  0      \
    --project benchmark/runs \
    --name    SF_YOLO_eval
```

> **Log:** lưu tại `benchmark/runs/SF_YOLO_s<seed>/train_log.txt` (tee-log giống SC-YOLO12).  
> **Kết quả eval:** `metrics_test.csv` + `metrics_test.json` trong `benchmark/runs/SF_YOLO_eval/`.

### Siêu tham số paper (cố định)

| Tham số | Giá trị | Ghi chú |
|---|---|---|
| `imgsz` | 736 | 736×736 px |
| `batch` | 16 | cố định |
| `optimizer` | SGD | cố định |
| `lr0` | 0.01 | học ban đầu |
| `lrf` | 0.01 | học cuối |
| `weight_decay` | 0.0005 | phân rã |
| `momentum` | 0.937 | động lượng SGD |
| `epochs` | 300 | cố định |

Log console lưu tại `runs/benchmark/SFYOLO_s<seed>/train_log.txt` (tee-log giống SC-YOLO12). Sau khi chạy đủ seed, dùng `eval/` (bootstrap CI, corrected t-test) để so với cấu hình SC-YOLO12 tốt nhất (M5).

## Fairness & lưu ý

<aside>
⚖️

Cùng **split cố định** (517/64/64), cùng **imgsz 640**, cùng epochs + early-stop, **3 seed [0,1,2]**. KHÔNG dùng 736×736/300ep của paper cho bảng chính; KHÔNG trộn số gốc paper (khác dataset DUO/UTDAC2020/TrashCan).

</aside>

<aside>
⚠️

**OOM 8GB:** 2 head ở 80×80 (stride 8) nặng hơn head stride 8 thông thường. Nếu OOM → giảm `--batch`, cân nhắc `--imgsz 512`, hoặc gradient-checkpoint; ghi rõ mọi ngoại lệ trong bảng.

</aside>

<aside>
🔍

Index layer **nhạy theo phiên bản Ultralytics** — luôn chạy smoke-test `build_sf_yolo` trước khi train thật, và **pin đúng phiên bản** trong `requirements.txt` (bản vá `parse_model` dựa trên regex đọc mã nguồn).

</aside>