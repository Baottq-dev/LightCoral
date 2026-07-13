# SC-YOLO12 — Hướng dẫn sử dụng

Mã nguồn ablation 5 module trên baseline **YOLOv12n** cho bài toán phát hiện san hô mềm
(**SCoralDet**, 646 ảnh gốc, 6 lớp, split 8-1-1 có sẵn: train=517 / val=64 / test=65).
Mọi tổ hợp từ 0–5 module đều chạy được qua cờ CLI `--modules`.

Ngoài pipeline ablation chính (`train.py`/`test.py`), repo còn có:
- `benchmark/` — so sánh SC-YOLO12 với các model YOLO khác (v8/v10/11/12/26, RT-DETR).
- `scripts/` — driver đa-seed cho ablation, sweep hyperparameter, đánh giá đa-seed.

> **Khủng hoảng tái lập (đọc trước khi kết luận bất kỳ số liệu nào):** nhiễu huấn luyện
> giữa các seed (~0.02–0.03 mAP) có độ lớn ngang với hiệu ứng module công bố. Riêng
> `workers` khác 0 cũng có thể lệch test mAP ~0.028. **Không bao giờ kết luận từ 1 seed.**
> Luôn chạy ≥3 seed, báo cáo mean±std (sample std, ddof=1), giữ `workers=0`.

---

## 1. Cài đặt

```bash
git clone <repo> && cd LightCoral-YOLO
pip install -r requirements.txt
```

`requirements.txt` pin `torch==2.7.1+cu128` (RTX 4060, CUDA 12.8) và `ultralytics==8.4.52`.
Máy khác/CUDA khác: đổi `cu128` trong `requirements.txt` theo [pytorch.org](https://pytorch.org/get-started/locally/).

Kiểm tra GPU:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Trọng số pretrained cần có sẵn ở gốc repo (đã kèm theo repo): `yolo12n.pt` (dùng cho
`--modules`/`--preset`), `yolo11n.pt`, `yolov10n.pt`, `yolo26n.pt` (dùng cho `benchmark/`).

### Cấu trúc thư mục

```
LightCoral-YOLO/
├─ cfg/                    # base_yolov12n*.yaml, module_specs.yaml, coral_soft_yolo.yaml
├─ models/                 # common.py, registry.py, shallow_p2.py, sfdf.py, pg_dam.py, fga2.py
├─ engine/                 # build_model.py (dung YAML dong), losses.py (DegradationLoss)
├─ augment/                # physics_degradation.py (module 3)
├─ utils/                  # seed.py (set_seed/make_generator), flops_fps.py
├─ eval/                   # group_kfold.py, bootstrap_ci.py, corrected_ttest.py
├─ train.py                # CLI huan luyen 1 config (goc pipeline ablation)
├─ test.py                 # CLI danh gia 1 checkpoint (val/test)
├─ scripts/                # driver da-seed: ablation, sweep HP, eval da-seed
├─ tools/                  # visualize_physics_aug.py
├─ benchmark/              # pipeline benchmark doc lap (model khac: v8/v10/11/26, RT-DETR)
├─ data/, datasets/        # du lieu goc + du lieu da convert format YOLO
└─ requirements.txt
```

---

## 2. Bảng module

| **#** | **Module** | **Loại** | **Vị trí tác động** | **File** |
| --- | --- | --- | --- | --- |
| ① | Shallow-feature (AMCF + shallow-fusion) | Kiến trúc | AMCF thay downsample idx 7 + fuse layer 2→head P3 | `models/shallow_p2.py` • `cfg/base_yolov12n_shallow.yaml` |
| ② | SFDF (spatial–frequency DWT fusion) | Kiến trúc | Thay 4 khối tại idx 2/4/6/8 | `models/sfdf.py` |
| ③ | Physics degradation aug | Train-only | `preprocess_batch` (không đổi kiến trúc) | `augment/physics_degradation.py` |
| ④ | PG-DAM (FiLM + L_deg) | Kiến trúc + loss | Chèn sau stem idx 0 | `models/pg_dam.py` • `engine/losses.py` |
| ⑤ | FGA² (frequency-gated area attention) | Kiến trúc | Thay A2C2f tại idx 11, 14 (neck) | `models/fga2.py` |

Tham số mặc định của từng module nằm trong `cfg/module_specs.yaml` (`modules.*`), bao gồm
`train_defaults` (HP mặc định khi không truyền cờ CLI: `optimizer: auto, lr0: 0.01, ...`).
**Lưu ý:** `optimizer: auto` trong file này KHÔNG phải HP đang thực chạy trong bất kỳ pipeline
nào bên dưới — mọi driver (`run_ablation.py`, `run_s1_sweep.py`, `3_train_baseline.py`) đều
truyền `--optimizer` tường minh (SGD hoặc AdamW) để đảm bảo tái lập.

---

## 3. Quickstart — chạy 1 config bằng `train.py`

```bash
# B1. Baseline B0
python train.py --data cfg/coral_soft_yolo.yaml --modules "" --seed 0

# B2. To hop tuy chon (vd module 1,2,4)
python train.py --data cfg/coral_soft_yolo.yaml --modules 1,2,4 --seed 0

# B3. Preset day du (xem bang preset o muc 4)
python train.py --data cfg/coral_soft_yolo.yaml --preset E8 --seed 0

# B4. Ghi de HP (mac dinh lay tu module_specs.yaml.train_defaults neu bo qua)
python train.py --data cfg/coral_soft_yolo.yaml --modules 3,4 --seed 0 \
    --optimizer SGD --lr0 0.001 --lrf 0.01 --weight_decay 0.0005 \
    --warmup_epochs 3 --epochs 100 --imgsz 640 --batch 16 --workers 0

# B5. Danh gia tren tap TEST sau khi train xong
python test.py --weights runs/scyolo12/<run_name>/weights/best.pt \
    --data cfg/coral_soft_yolo.yaml --split test --conf 0.001 --iou 0.7 \
    --csv runs/scyolo12/<run_name>_test.csv

# B6. Chi phi tinh toan (Params/GFLOPs/FPS)
python -m utils.flops_fps --modules 1,2,4,5 --imgsz 640 --device 0
```

`run_name` do `train.py` tự sinh theo công thức
`f"{name}_{data_tag}_s{seed}_ep{epochs}_lr0{lr0}_lrf{lrf}_config1"` (xem `train.py:185`),
với `name` = `M<tag chữ số module đã sort>` (vd `--modules 3,4` → `M34`) hoặc tên preset.

Mỗi lần train tự ghi log đầy đủ (banner Ultralytics + bảng kiến trúc) vào
`runs/scyolo12/<run_name>/train_log.txt`, dòng đầu là lệnh CLI đã chạy.

Toàn bộ HP override đều **tùy chọn** (`type=..., default=None`) — nếu không truyền,
`train.py` lấy từ `train_defaults` trong `cfg/module_specs.yaml`.

### Multi-seed thủ công + đánh giá tổng hợp

```bash
for s in 0 1 2; do
    python train.py --data cfg/coral_soft_yolo.yaml --preset E8 --seed $s
done

# Gom mean+/-std qua seed tren tap test (chi 1 config)
python scripts/eval_seeds.py --run E8 --seeds 0,1,2 --data cfg/coral_soft_yolo.yaml
# -> runs/scyolo12/E8_test_agg/{per_seed_test.csv, per_class_agg_test.csv, summary_test.json}
```

`eval_seeds.py --run` cần khớp tên run **không kèm** hậu tố `_s<seed>`; script tự tìm
`runs/scyolo12/{run}_s{seed}/weights/best.pt` cho từng seed.

### GroupKFold chống rò rỉ near-duplicate (tùy chọn)

```bash
python -m eval.group_kfold --images datasets/coral_soft_yolo/images/train --k 5
python train.py --data data/folds/coral_fold0.yaml --preset E8 --seed 0
```

---

## 4. Bảng preset (khớp `cfg/module_specs.yaml`)

| **Preset** | **Modules** | **Ý nghĩa** |
| --- | --- | --- |
| B0 | — | YOLOv12n thuần |
| E1 | 3 | chỉ physics aug |
| E2 | 1 | chỉ AMCF + shallow-fusion |
| E3 | 2 | chỉ SFDF |
| E4 | 1,2 | Shallow + SFDF |
| E5 | 1,2,3 | Shallow + SFDF + physics aug |
| E6 | 1,2,3,4 | + PG-DAM |
| E7 | 1,2,3,5 | + FGA² (không PG-DAM) |
| E8 | 1,2,3,4,5 | full SC-YOLO12 |

Ngoài preset, `--modules` nhận **mọi tổ hợp** con của {1..5} (32 tổ hợp), vd `--modules 3,4`.

---

## 5. Pipeline ablation chính (đa-seed, tự động) — `scripts/run_ablation.py`

Driver train + eval end-to-end cho **B0 / M1 / M2 / M12 / M3 / M4 / M34 × 3 seed (0,1,2)**,
dùng chung 1 bộ HP cố định (khớp mặc định `benchmark/scripts/3_train_baseline.py`, để so sánh
module công bằng — **không phải HP tối ưu**, xem mục 7 nếu muốn tìm HP tốt nhất):

```
optimizer=SGD, lr0=0.001, lrf=0.01, weight_decay=0.0005, warmup_epochs=3,
epochs=100, imgsz=640, batch=16, workers=0
```

```bash
# Chay day du 7 config x 3 seed = 21 run (train qua subprocess train.py, eval in-process)
python scripts/run_ablation.py

# Chi 1 vai config / seed (vd sanity check nhanh)
python scripts/run_ablation.py --configs B0,M34 --seeds 0,1

# Sanity check 1 epoch (khong ton gio)
python scripts/run_ablation.py --configs B0,M34 --seeds 0 --epochs 1
```

Idempotent: nếu `best.pt` của 1 (config, seed) đã tồn tại, driver tự skip train, chỉ eval lại.
Eval chuẩn hoá `conf=0.001, iou=0.7` (tập test) cho mọi config. Kết quả ghi ra:
- `runs/ablation/per_seed_test.csv` — mAP50/mAP50-95 từng (config, seed).
- `runs/ablation/ablation_agg_test.csv` — mean±std (ddof=1) mỗi config + Δ so với B0.

---

## 6. Benchmark model khác — `benchmark/scripts/run_benchmark.ps1`

So sánh SC-YOLO12/YOLOv12n với các model YOLO khác trên cùng dataset (mục đích: xác nhận
YOLOv12n là baseline hợp lý, không phải để tune HP). Chạy trên PowerShell:

```powershell
# Mac dinh: cac model duoc bo comment trong $Models, moi model x seed 0,1,2
.\benchmark\scripts\run_benchmark.ps1

# Tuy chinh
.\benchmark\scripts\run_benchmark.ps1 -Epochs 50 -ImgSz 1280
.\benchmark\scripts\run_benchmark.ps1 -Seeds 0,1,2
.\benchmark\scripts\run_benchmark.ps1 -EvalSplit val      # mac dinh test
.\benchmark\scripts\run_benchmark.ps1 -SkipEval           # chi train
.\benchmark\scripts\run_benchmark.ps1 -DryRun             # in lenh, khong chay
```

Mỗi model×seed: train qua `benchmark/scripts/3_train_baseline.py` (hỗ trợ hub model
yolov8/v10/11/12/26, rtdetr, hoặc `--weights` cho checkpoint tùy chỉnh), rồi eval qua
`benchmark/scripts/4_evaluate.py` (`conf=0.001, iou=0.7`, cùng chuẩn với ablation).
Log tổng hợp + CSV mean±std (`benchmark_agg_<timestamp>.csv`) ghi vào `$Project/_logs/`.

---

## 7. Nghiên cứu / tune hyperparameter tốt nhất — `scripts/run_s1_sweep.py`

**Tách biệt hoàn toàn** với HP cố định của mục 5/6 (dùng để so sánh module công bằng).
Mục này dành riêng cho việc **tìm bộ HP tốt nhất** cho 1 config cụ thể (mặc định M34,
sửa `MODULES` trong file nếu muốn áp cho config khác).

### Giai đoạn 1 — Screen (grid rộng, 1 seed)

```bash
python scripts/run_s1_sweep.py
```

Grid hiện tại (`OPTLR × LRFS × WDS × COSES` = 4×2×3×2 = **48 config**, seed=42, 200 epoch):

| Trục | Giá trị |
| --- | --- |
| (optimizer, lr0) — bind theo cặp, không cross bừa | (SGD, 0.01), (SGD, 0.005), (AdamW, 0.002), (AdamW, 0.001) |
| lrf | 0.01, 0.001 |
| weight_decay | 0.0005, 0.001, **0.005** (mở rộng biên trên — dataset chỉ 517 ảnh train, dễ overfit) |
| cos_lr | False, True |

Mỗi run gọi `scripts/train_tune.py` (bản mở rộng của `train.py`, thêm `--cos_lr`/`--momentum`
và tự đặt tên run mô tả HP, vd `M34__AdamW_lr0-0p001_lrf-0p01_cos-0_wd-0p0005_s42`).
`patience=0` (tắt early stop) nhưng `best.pt` vẫn lưu theo epoch fitness cao nhất
(`fitness = 0.2*mAP50 + 0.8*mAP50-95`) — train đủ 200 epoch không có nghĩa checkpoint bị
overfit, vì best checkpoint được chọn tại đúng epoch tốt nhất.

```bash
# Gom ket qua, sap xep theo fitness
python scripts/collect_screen.py
# -> in bang + ghi screen_summary.csv
```

### Giai đoạn 2 — Confirm (top ứng viên, thêm seed)

Sau khi xem `screen_summary.csv`, chọn top ứng viên và sửa `CONFIGS` trong
`scripts/run_confirm.py` (list `(optimizer, lr0, lrf, cos_lr, weight_decay)`), rồi:

```bash
python scripts/run_confirm.py     # chay them SEEDS=[0,1] cho moi ung vien (co seed=42 tu screen la 3)
python scripts/collect_confirm.py # gom mean+/-std (ddof=1) qua >=2 seed -> confirm_summary.csv
```

`collect_confirm.py` tự loại config có <2 seed (không đủ để tính std đáng tin).

> **Lưu ý quan trọng khi đọc `screen_summary.csv`/`confirm_summary.csv` đã có sẵn trong
> repo:** các run trước đây (khi `train_tune.py:270` còn thiếu
> `generator=make_generator(seed)`) có RNG của physics aug (module 3) KHÔNG khóa theo seed
> — nghĩa là kết quả M34 cũ **không tái lập được** dù cùng `--seed`. Bug này đã được sửa;
> **cần chạy lại toàn bộ sweep M34** để có số liệu đáng tin, không dùng lại CSV cũ.

---

## 8. So sánh thống kê 2 cấu hình

```bash
# Corrected resampled t-test (Nadeau & Bengio) — muc fold x seed
python -m eval.corrected_ttest --a a.json --b b.json --n-train 517 --n-test 65

# Bootstrap CI muc anh (paired per-image metric)
python -m eval.bootstrap_ci --a a.json --b b.json --n-boot 10000 --ci 0.95 --seed 0
```

---

## 9. Smoke tests (chạy trước khi train thật)

```bash
# 1) Build + forward du kien truc (bat gay khi sua parse_model monkey-patch)
python -m engine.build_model
```

```python
# 2) DWT kha nghich: HaarIDWT(HaarDWT(x)) ~ x
import torch
from models.common import HaarDWT, HaarIDWT
x = torch.randn(2, 16, 64, 64)
assert torch.allclose(HaarIDWT()(HaarDWT()(x)), x, atol=1e-5)

# 3) SFDF (4 khoi idx 2/4/6/8) + AMCF (idx 7) giu dung shape
from models.sfdf import SFDF
from models.shallow_p2 import AMCF
assert SFDF(64, 128)(torch.randn(2, 64, 80, 80)).shape == (2, 128, 80, 80)
assert SFDF(256, 256, swap=True)(torch.randn(2, 256, 20, 20)).shape == (2, 256, 20, 20)
assert AMCF(128, 256, 2)(torch.randn(2, 128, 40, 40)).shape == (2, 256, 20, 20)

# 4) FGA2: lambda=0 => tuong duong attention goc; lambda co gradient
from models.fga2 import FGA2_A2C2f
m = FGA2_A2C2f(384, 128, n=1, area=4, lambda_init=0.0)
out = m(torch.randn(2, 384, 40, 40))
assert out.shape == (2, 128, 40, 40)
out.sum().backward()
lams = [p for n_, p in m.named_parameters() if n_.endswith("lam")]
assert all(p.grad is not None for p in lams), "lambda phai nhan gradient"

# 5) Physics aug: z_gt dung shape, anh trong [0,1]
import yaml
from augment.physics_degradation import from_specs
specs = yaml.safe_load(open("cfg/module_specs.yaml"))
aug = from_specs(specs)
imgs, z = aug(torch.rand(4, 3, 640, 640))
assert z.shape == (4, 7) and imgs.min() >= 0 and imgs.max() <= 1
```

```bash
# 6) Xem truc quan physics aug tren anh that
python tools/visualize_physics_aug.py --n 8 --seed 42
```

---

## 10. Sanity checks trước khi báo cáo kết quả

- [ ] GroupKFold: không group nào xuất hiện ở cả train và val (assert sẵn trong `eval/group_kfold.py`).
- [ ] Mỗi cấu hình chạy đủ seeds [0, 1, 2]; báo cáo mean ± std (ddof=1), **không cherry-pick seed tốt nhất**.
- [ ] `workers=0` cho mọi run cần so sánh (workers khác 0 tự nó lệch test mAP ~0.028).
- [ ] Eval dùng cùng protocol `conf=0.001, iou=0.7` giữa các config muốn so sánh.
- [ ] So sánh A vs B: `eval/corrected_ttest.py` (mức fold×seed) **và** `eval/bootstrap_ci.py` (mức ảnh).
- [ ] Báo cáo kèm Params/GFLOPs/FPS từ `utils/flops_fps.py` cho mọi cấu hình trong bảng chính.
- [ ] Kiểm tra log `L_deg` giảm dần khi dùng ④+③ (nếu không giảm → xem lại chuẩn hóa z trong `augment/physics_degradation.py`).
- [ ] Nếu CI của 2 config chồng nhau (vd B0 vs M34) → kết luận trung thực là "chưa khác biệt có ý nghĩa", không phải "module tốt hơn".

---

## Hạn chế đã biết

- Dùng ④ (PG-DAM) làm lệch index layer → pretrained COCO chỉ nạp lại đúng qua cơ chế remap
  `_load_pretrained_shifted` (trong `train.py`/`scripts/train_tune.py`, dịch key
  `model.<i>.*` → `model.<i+1>.*` cho `i >= insert_after_idx`).
- `torch.use_deterministic_algorithms` có thể cảnh báo với vài op CUDA — đã đặt `warn_only=True`
  (`utils/seed.py`).
- Phiên bản Ultralytics cần khớp `requirements.txt` (`8.4.52`); nếu nâng cấp, kiểm tra lại chữ
  ký `DetectionModel.loss` và `preprocess_batch`, cùng logic `build_optimizer`/`optimizer=auto`.
- `scripts/` chưa được track trong git tại thời điểm viết README này — nếu clone lại repo,
  xác nhận các file `scripts/*.py` có mặt trước khi chạy mục 5/7.
