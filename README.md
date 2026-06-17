# SC-YOLO12 — Hướng dẫn sử dụng

Mã nguồn ablation 5 module trên baseline **YOLOv12n** cho bài toán phát hiện san hô mềm (SCoralDet, 646 ảnh, 6 lớp). Mọi tổ hợp từ 0–5 module đều chạy được qua cờ CLI.

## 1. Cài đặt

```bash
git clone <repo> && cd sc-yolo12
pip install -r requirements.txt
# tai trong so pretrained
# yolov12n.pt tu repo YOLOv12 chinh thuc (sunsmarterjie/yolov12)
```

Cấu trúc thư mục (khớp 1-1 với các trang con ở đây):

```jsx
sc-yolo12/
├─ cfg/            # base_yolov12n.yaml, base_yolov12n_shallow.yaml, module_specs.yaml
├─ models/         # common.py, registry.py, shallow_p2.py, sfdf.py, pg_dam.py, fga2.py
├─ train.py        # CLI huan luyen (chay: python train.py)
├─ augment/        # physics_degradation.py
├─ engine/         # build_model.py, losses.py
├─ eval/           # group_kfold.py, bootstrap_ci.py, corrected_ttest.py
├─ utils/          # seed.py, flops_fps.py
└─ requirements.txt
```

## 2. Bảng module

| **#** | **Module** | **Loại** | **Vị trí tác động** | **File** |
| --- | --- | --- | --- | --- |
| ① | Shallow-feature (AMCF + shallow-fusion) | Kiến trúc | AMCF thay downsample idx 7 + fuse layer 2→head P3 (giữ 3 head) | `models/shallow_p2.py`  • `cfg/base_yolov12n_shallow.yaml` |
| ② | SFDF (spatial–frequency DWT fusion) | Kiến trúc | Thay 4 khối tại idx 2/4/6/8 | `models/sfdf.py` |
| ③ | Physics degradation aug | Train-only | `preprocess_batch` (không đổi kiến trúc) | `augment/physics_degradation.py` |
| ④ | PG-DAM (FiLM + L_deg) | Kiến trúc + loss | Chèn sau stem idx 0 | `models/pg_dam.py`  • `engine/losses.py` |
| ⑤ | FGA² (frequency-gated area attention) | Kiến trúc | Thay A2C2f tại idx 11, 14 (neck) | `models/fga2.py` |

## 3. Quickstart

```bash
# ===== BUOC DAU: split CUNG 8-1-1 da chia san (cfg/coral_soft_yolo.yaml) =====
# B1. Baseline B0
python train.py --data cfg/coral_soft_yolo.yaml --modules "" --seed 0

# B2. To hop tuy chon (vd module 1,2,4)
python train.py --data cfg/coral_soft_yolo.yaml --modules 1,2,4 --seed 0

# B3. Preset day du
python train.py --data cfg/coral_soft_yolo.yaml --preset E8 --seed 0

# B4. Multi-seed (bao cao mean +- std)
for s in 0 1 2; do python train.py --data cfg/coral_soft_yolo.yaml --preset E8 --seed $s; done

# B5. Danh gia tren tap TEST (sau khi train xong)
python test.py --weights runs/scyolo12/E8_s0/weights/best.pt --data cfg/coral_soft_yolo.yaml --csv runs/scyolo12/E8_s0_test.csv

# B6. Chi phi tinh toan
python -m utils.flops_fps --modules 1,2,4,5 --device 0

# ===== (TUY CHON, ve sau) GroupKFold chong leakage near-duplicate =====
# python -m eval.group_kfold --images datasets/coral_soft_yolo/images/train --k 5
# python train.py --data data/folds/coral_fold0.yaml --preset E8 --seed 0
```

## 4. Bảng preset (khớp `cfg/module_specs.yaml`)

| **Preset** | **Modules** | **Ý nghĩa** |
| --- | --- | --- |
| B0 | — | YOLOv12n thuần |
| E1 | 3 | chỉ physics aug |
| E2 | 1 | chỉ AMCF + shallow-fusion |
| E3 | 2 | chỉ SFDF |
| E4 | 1,2 | Shallow + SFDF |
| E5 | 1,2,3 |   • physics aug |
| E6 | 1,2,3,4 |   • PG-DAM |
| E7 | 1,2,3,5 |   • FGA² (không PG-DAM) |
| E8 | 1,2,3,4,5 | full SC-YOLO12 |

Ngoài preset, `--modules` nhận **mọi tổ hợp** con của {1..5} (32 tổ hợp).

## 5. Smoke tests (chạy trước khi train thật)

```bash
# 1) Build + forward du 16 kien truc (to hop {1,2,4,5})
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

## 6. Sanity checks trước khi báo cáo kết quả

- [ ]  GroupKFold: không group nào xuất hiện ở cả train và val (assert sẵn trong `eval/group_kfold.py`).
- [ ]  Mỗi cấu hình chạy đủ seeds [0, 1, 2]; báo cáo mean ± std, không cherry-pick seed tốt nhất.
- [ ]  So sánh A vs B: `eval/corrected_ttest.py` (mức fold×seed) **và** `eval/bootstrap_ci.py` (mức ảnh).
- [ ]  Báo cáo kèm Params/GFLOPs/FPS từ `utils/flops_fps.py` cho mọi cấu hình trong bảng chính.
- [ ]  Kiểm tra log `L_deg` giảm dần khi dùng ④+③ (nếu không giảm → xem lại chuẩn hóa z).

<aside>
⚠️

**Hạn chế đã biết:**

- Dùng ④ (PG-DAM) làm lệch index layer → pretrained COCO chỉ nạp được một phần theo tên/shape khớp; nếu cần tối đa hóa transfer, remap state_dict theo thứ tự layer trước khi load.
- `torch.use_deterministic_algorithms` có thể cảnh báo với vài op CUDA — đã đặt `warn_only=True`.
- Phiên bản Ultralytics cần khớp `requirements.txt`; nếu nâng cấp, kiểm tra lại chữ ký `DetectionModel.loss` và `preprocess_batch`.
</aside>