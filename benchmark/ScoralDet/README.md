# SCoralDet (reimplemented) — benchmark package

<aside>
🪸

Reimplement **SCoralDet** (Lu et al., *Ecological Informatics* 85 (2025) 102937) trên base **YOLOv10n**, làm đối thủ benchmark cho SC-YOLO12 trên Soft-Coral. Paper **không release code** module/hyperparams → đây là **faithful approximation**, luôn nhãn "SCoralDet (reimplemented)" trong bảng kết quả.

</aside>

## Cách chạy (từ root repo `sc-yolo12/`)

```jsx
# 1) Smoke-test build (khong train): kiem 3 head + stride [8,16,32] + dem params
python benchmark/scoraldet/build_scoraldet.py
python -m benchmark.scoraldet.build_scoraldet

# 2) Test nhanh tung module
python benchmark/scoraldet/modules_mpfb.py     # MPFB reparam (train == deploy)
python benchmark/scoraldet/modules_neck.py     # GSConv + VoVGSCSP
python benchmark/scoraldet/apt_assigner.py     # APT transform
python benchmark/scoraldet/apt_loss.py         # SoftCls + SoftCenterConf

# 3) Train (cung split + protocol voi SC-YOLO12)
python -m benchmark.scoraldet.train_scoraldet --data data/scoraldet_fold0.yaml --seed 0
python -m benchmark.scoraldet.train_scoraldet --data ... --seed 0 --paper-protocol   # 300ep/SGD nhu paper
python -m benchmark.scoraldet.train_scoraldet --data ... --seed 0 --scratch          # tu scratch
```

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