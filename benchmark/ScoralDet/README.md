# SCoralDet (reimplemented) — benchmark package

<aside>
🪸

Reimplement **SCoralDet** (Lu et al., *Ecological Informatics* 85 (2025) 102937) trên base **YOLOv10n**, làm đối thủ benchmark cho SC-YOLO12 trên Soft-Coral. Paper **không release code** module/hyperparams → đây là **faithful approximation**, luôn nhãn "SCoralDet (reimplemented)" trong bảng kết quả.

</aside>

## Cách chạy (từ root repo `sc-yolo12/`)

```bash
# 1) Smoke-test build (khong train): kiem 3 head + stride [8,16,32] + dem params
python -m benchmark.scoraldet.build_scoraldet

# 2) Test nhanh tung module
python -m benchmark.scoraldet.modules_mpfb    # MPFB reparam (train == deploy)
python -m benchmark.scoraldet.modules_neck    # GSConv + VoVGSCSP
python -m benchmark.scoraldet.apt_assigner    # APT transform
python -m benchmark.scoraldet.apt_loss        # SoftCls + SoftCenterConf
```

### Train — tham số cơ bản

```bash
# Chạy mặc định (seed=42, epochs=300, imgsz=640, batch=32, SGD)
python -m benchmark.scoraldet.train_scoraldet --data data/scoraldet_fold0.yaml

# Paper protocol: 300ep / batch32 / SGD lr0.01 momentum0.937 (để đối chiếu số paper)
python -m benchmark.scoraldet.train_scoraldet --data data/scoraldet_fold0.yaml --paper-protocol

# Train từ scratch
python -m benchmark.scoraldet.train_scoraldet --data data/scoraldet_fold0.yaml --scratch

# Chỉ định seed và GPU
python -m benchmark.scoraldet.train_scoraldet --data data/scoraldet_fold0.yaml --seed 1 --device 0
```

### Train — tất cả CLI args (ghi đè khi cần)

```bash
python -m benchmark.scoraldet.train_scoraldet \
    --data          data/scoraldet_fold0.yaml \  # [BẮT BUỘC] data YAML với split cố định
    --seed          42           \  # hạt giống (mặc định=42)
    --epochs        300          \  # số epoch (paper=300, cố định)
    --imgsz         640          \  # kích thước ảnh (paper=640)
    --batch         32           \  # kích thước lô (paper=32)
    --optimizer     SGD          \  # bộ tối ưu (paper=SGD)
    --lr0           0.01         \  # tốc độ học ban đầu (paper=0.01)
    --lrf           0.01         \  # tốc độ học cuối (paper=0.01)
    --weight_decay  0.0005       \  # phân rã trọng số (paper=0.0005)
    --warmup_epochs 3            \  # số epoch warmup
    --momentum      0.937        \  # động lượng SGD (paper=0.937)
    --paper-protocol             \  # flag: ghi đè về 300ep/batch32/SGD lr0.01 mom0.937
    --scratch                    \  # flag: train từ đầu (không pretrained)
    --weights       yolov10n.pt  \  # pretrained khởi tạo (mặc định=yolov10n.pt)
    --apt-power     2.0          \  # số mũ p của APT (paper=2)
    --apt-thr       0.5          \  # ngưỡng T của APT (giả định=0.5)
    --soft-chi      10.0         \  # χ=10 cho SoftCenterConfLoss (paper Sec 4.2)
    --soft-delta    3.0          \  # δ=3 cho SoftCenterConfLoss (paper Sec 4.2)
    --fliplr        0.5          \  # lật ngang ngẫu nhiên xác suất 50%
    --hsv-h         0.015        \  # biên độ hue HSV ±1.5%
    --hsv-s         0.4          \  # biên độ saturation HSV ±40%
    --hsv-v         0.4          \  # biên độ brightness HSV ±40%
    --scale         0.5          \  # co giãn ngẫu nhiên 50%–100%
    --translate     0.1          \  # dịch chuyển tối đa 10%
    --device        0            \  # GPU id
    --workers       4            \  # số DataLoader workers
    --project       benchmark/runs \  # thư mục lưu kết quả
    --name          SCoralDet    \  # tên run (thực tế = SCoralDet_s<seed>)
    --logfile       path/to/log.txt   # tuỳ chọn: ghi log ra file riêng
```

### Multi-seed (báo mean ± std)

```bash
for s in 0 1 2; do
  python -m benchmark.scoraldet.train_scoraldet --data data/scoraldet_fold0.yaml --seed $s
done
```

### Đánh giá trên test split

```bash
# Cơ bản (không reparam — đo Params/GFLOPs chế độ train)
python -m benchmark.scoraldet.eval_scoraldet \
    --data    data/scoraldet_fold0.yaml \
    --weights benchmark/runs/SCoralDet_s42/weights/best.pt

# Với --reparam: switch_to_deploy MPFB → đo Params/GFLOPs deploy (paper: ~2.4M / ~5.9G)
python -m benchmark.scoraldet.eval_scoraldet \
    --data    data/scoraldet_fold0.yaml \
    --weights benchmark/runs/SCoralDet_s42/weights/best.pt \
    --reparam \
    --split   test   \   # test | val | train
    --imgsz   640    \
    --batch   16     \
    --device  0      \
    --project benchmark/runs \
    --name    SCoralDet_eval
```

> **Log:** lưu tại `benchmark/runs/SCoralDet_s<seed>/train_log.txt`.  
> **Kết quả eval:** `metrics_test.csv` + `metrics_test.json` trong `benchmark/runs/SCoralDet_eval/`.

### Siêu tham số paper (cố định, Sec 4.2)

| Tham số | Giá trị | Ghi chú |
|---|---|---|
| `imgsz` | 640 | 640×640 px |
| `batch` | 32 | cố định |
| `optimizer` | SGD | cố định |
| `lr0` | 0.01 | học ban đầu |
| `lrf` | 0.01 | học cuối |
| `weight_decay` | 0.0005 | phân rã |
| `momentum` | 0.937 | động lượng |
| `epochs` | 300 | cố định |
| `fliplr` | 0.5 | lật ngang 50% |
| `hsv_h/s/v` | 0.015 / 0.4 / 0.4 | HSV augment |
| `scale` | 0.5 | co giãn 50%–100% |
| `translate` | 0.1 | dịch chuyển ≤10% |
| `apt_power` | 2.0 | số mũ APT (p) |
| `soft_chi` | 10.0 | χ soft center loss |
| `soft_delta` | 3.0 | δ soft center loss |

## Cấu trúc

| File | Vai trò |
| --- | --- |
| `modules_mpfb.py` | MPFB (5 nhánh + reparameterize) + C2f_MPFB — đóng góp ① |
| `modules_neck.py` | GSConv + VoVGSCSP (vendored, slim-neck) — đóng góp ② |
| `apt_assigner.py` | APTAssigner (Eq.2–3) — đóng góp ③ (label assignment) |
| `apt_loss.py` | SoftClsLoss/SoftCenterConfLoss/v10APTDetectionLoss — đóng góp ③ (loss) |
| `base_scoraldet.yaml` | YOLOv10n chuẩn + điểm cắm |
| `build_scoraldet.py` | vá module + kênh scale tường minh + smoke-test |
| `train_scoraldet.py` | train wrapper + inject APT loss |

## Đối chiếu paper ↔ code

| Đóng góp | Paper | Code này | Trạng thái |
| --- | --- | --- | --- |
| MPFB | 5 nhánh (Fig.2) + reparam (Eq.8–9) | `MPFB` 5 nhánh + `switch_to_deploy()` | ✅ faithful |
| GSConv/VoV-GSCSP | slim-neck (Fig.3) | vendored `modules_neck.py` | ✅ faithful |
| APT assign | u_trans + t=s^α·u^β (Eq.2–3) | `APTAssigner.get_box_metrics` | ✅ cốt lõi; T,α,β,k giả định |
| Soft cls / center loss | Eq.5–7 (χ=10 δ=3) | `SoftClsLoss`/`SoftCenterConfLoss` | ⚠️ có sẵn, **không wire mặc định** |
| Backbone MPFB | "tất cả C2f" (Fig.1) | idx 2/4/6 (giữ C2fCIB@8) | ⚠️ giả định |

## Mốc verify (Soft-Coral, paper Table 1)

<aside>
🎯

SCoralDet full: **mAP50 ≈ 0.819, mAP50-95 ≈ 0.532, Param ≈ 2.4M, GFLOPs ≈ 5.9** (300 epoch). Đo Param/GFLOPs **sau reparameterize MPFB** (`switch_to_deploy()`) để khớp. Nếu lệch nhiều → kiểm MPFB (reparam) hoặc APT (α/β/T/k).

</aside>

## Wiring soft-center loss (tùy chọn, thực nghiệm)

Mặc định chỉ bật **APT assigner** (chạy ổn định). Để thử nghiệm thêm $L_{conf}$ (Eq.7): override `__call__` của `v8APTDetectionLoss`, lấy `target_bboxes` từ assigner, tính `SoftCenterConfLoss` trên anchor positive (toạ độ chuẩn hóa [0,1]) và cộng vào `loss[0]` với trọng số nhỏ. **Cảnh báo:** số hạng $\chi^{(|x|-\delta)}$ dễ tràn — clamp exponent hoặc chuẩn hóa trước khi bật.

## Ghi chú fairness

- Mặc định dùng `cfg/module_specs.yaml → train_defaults` (giống SC-YOLO12 & SF-YOLO): **cùng split, imgsz, optimizer, seed**.
- `--paper-protocol` chỉ để **đối chiếu con số paper**, không dùng cho bảng so sánh chính.
- Multi-seed `[0,1,2]` → báo mean±std như các model khác.