# Huong dan Train YOLOv12 + F3M

## 0. Tom tat 3 buoc nhanh

```bash
# 1) Cai dat
pip install ultralytics torch torchvision

# 2) Copy module + dang ky (xem muc 2)
#    - copy f3m_modules.py vao ultralytics/nn/modules/
#    - them khoi elif vao parse_model trong ultralytics/nn/tasks.py

# 3) Train
python train_f3m.py --cfg yolo12n-f3m.yaml --data coral_dataset.yaml --epochs 300 --batch 16 --device 0
```

---

## 1. Cai dat moi truong

```bash
conda create -n f3m python=3.10 -y && conda activate f3m
pip install ultralytics            # keo theo torch, torchvision
# Kiem tra GPU:
python -c "import torch; print(torch.cuda.is_available())"
```

## 2. Dang ky module (2 cach)

### Cach A - sua source (chac chan nhat)
1. Copy `f3m_modules.py`  ->  `ultralytics/nn/modules/f3m_modules.py`
2. Trong `ultralytics/nn/modules/__init__.py` them:
   ```python
   from .f3m_modules import F3M, F3MWithSA
   ```
   va them `"F3M", "F3MWithSA"` vao `__all__`.
3. Trong `ultralytics/nn/tasks.py`:
   - Dau file: `from ultralytics.nn.modules import F3M, F3MWithSA`
   - Trong `parse_model(...)`, them vao chuoi if/elif:
     ```python
     elif m in {F3M, F3MWithSA}:
         c1 = c2 = ch[f]        # residual: out == in channels
         args = [c1, *args]     # chen in_channels vao dau args
     ```

### Cach B - runtime, it sua hon (dung f3m_register.py)
De `f3m_modules.py` + `f3m_register.py` cung thu muc lam viec, roi:
```python
import f3m_register             # tu dong tiem F3M/F3MWithSA vao namespace
from ultralytics import YOLO
model = YOLO("yolo12n-f3m.yaml")
```
Cach B bo qua buoc sua `__init__.py`, NHUNG van phai them khoi `elif`
trong `parse_model` (channel-handling) mot lan nhu Cach A buoc 3.

> Tai sao van can elif? Vi F3M/F3MWithSA bao toan so kenh; parser mac dinh
> khong chen `in_channels` vao args. Khoi elif tu suy ra `c1 = c2 = ch[f]`
> (da tinh ca width scaling theo scale n/s/m/l/x), nho do YAML chi can ghi
> hyperparam: `- [-1, 1, F3M, [0.125, False]]`.

## 3. Chuan bi dataset

Dataset theo dinh dang YOLO. Sua `coral_dataset.yaml`:
```yaml
path: /duong/dan/SCoralDet
train: images/train
val:   images/val
nc: 6
names: [Euphyllia, Favites, Platygyra, Acropora, Montipora, Porites]
```
Cau truc thu muc:
```
SCoralDet/
  images/train/*.jpg   labels/train/*.txt
  images/val/*.jpg     labels/val/*.txt
```
Moi dong nhan: `<class_id> <xc> <yc> <w> <h>` (chuan hoa 0..1).

## 4. Train

### Dung script co san
```bash
python train_f3m.py --cfg yolo12n-f3m.yaml --data coral_dataset.yaml \
    --epochs 300 --imgsz 640 --batch 16 --device 0 --name yolo12n_f3m
```

### Hoac dung CLI Ultralytics (sau khi dang ky Cach A)
```bash
yolo detect train model=yolo12n-f3m.yaml data=coral_dataset.yaml \
    epochs=300 imgsz=640 batch=16 device=0 optimizer=SGD lr0=0.01
```

### Transfer learning tu trong so YOLOv12 goc (khuyen nghi voi dataset nho)
```bash
python train_f3m.py --cfg yolo12n-f3m.yaml --weights yolo12n.pt \
    --data coral_dataset.yaml --epochs 300
```
`.load()` se nap cac lop trung khop; rieng F3M/F3MWithSA khoi tao moi.

## 5. Cac thi nghiem ablation (doi chieu Table 2 bai bao)

| Cau hinh        | YAML                          | Y nghia            |
|-----------------|-------------------------------|--------------------|
| Baseline        | yolo12n.yaml (goc)            | khong F3M          |
| Chi Stem        | yolo12n-f3m-stem-only.yaml    | onlyF3MWithSA      |
| Chi Neck        | yolo12n-f3m-neck-only.yaml    | onlyF3M            |
| Day du (Dual)   | yolo12n-f3m.yaml             | F3M + F3MWithSA    |

```bash
for cfg in yolo12n-f3m-stem-only yolo12n-f3m-neck-only yolo12n-f3m; do
    python train_f3m.py --cfg ${cfg}.yaml --data coral_dataset.yaml \
        --epochs 300 --name ${cfg}
done
```

## 6. Danh gia / Suy luan / Xuat

```bash
# Validate
yolo detect val model=runs/detect/yolo12n_f3m/weights/best.pt data=coral_dataset.yaml

# Predict
yolo detect predict model=runs/detect/yolo12n_f3m/weights/best.pt source=test.jpg

# Export ONNX
yolo export model=runs/detect/yolo12n_f3m/weights/best.pt format=onnx opset=12
```

## 7. Kiem tra so tham so / GFLOPs (doi chieu bai bao)

```python
import f3m_register
from ultralytics import YOLO
YOLO("yolo12n-f3m.yaml").info(detailed=False, verbose=True)
# Bai bao (tren YOLO11n): +0.03M params, +0.2 GFLOPs so voi baseline.
```

## 8. Loi thuong gap

| Loi | Nguyen nhan / cach xu ly |
|-----|--------------------------|
| `KeyError: 'F3M'` khi build model | Chua dang ky ten -> import f3m_register hoac sua __init__.py |
| `TypeError: __init__() missing 'reduction_ratio'` hoac channel sai | Chua them khoi `elif m in {F3M, F3MWithSA}` vao parse_model |
| `shape mismatch` khi Concat o head | Sua so layer index trong YAML cho khop sau khi them/bot layer |
| mAP thap, loss khong giam | Thu transfer learning (--weights yolo12n.pt), tang epochs, kiem tra nhan |

## 9. Cau hinh dung trong cau hinh ket hop day du (theo bai bao)

| Vi tri        | Module     | r     | gate  | spatial attn |
|---------------|------------|-------|-------|--------------|
| Stem (early)  | F3MWithSA  | 0.33  | True  | Co (k=7)     |
| Pre-SPPF      | F3M        | 0.125 | False | Khong        |
