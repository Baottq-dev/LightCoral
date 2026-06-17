# Huong dan tich hop F3M vao YOLOv12 (Ultralytics)

Gom 3 phan theo yeu cau:
1. **F3M tai Neck** (pre-SPPF) — lite, `gate=False`
2. **F3MWithSA tai Stem** (early) — full + spatial attention, `gate=True`
3. **Su ket hop day du** — file `yolo12n-f3m.yaml`

---

## Buoc 1 — Copy file module

Copy `f3m_modules.py` vao:
```
ultralytics/nn/modules/f3m_modules.py
```

## Buoc 2 — Export module

Mo `ultralytics/nn/modules/__init__.py`, them:
```python
from .f3m_modules import F3M, F3MWithSA

__all__ = (
    # ... cac export co san ...
    "F3M",
    "F3MWithSA",
)
```

## Buoc 3 — Dang ky trong parser

Mo `ultralytics/nn/tasks.py`:

(a) Them import gan dau file:
```python
from ultralytics.nn.modules.f3m_modules import F3M, F3MWithSA
```

(b) Trong ham `parse_model(...)`, tim vong lap xu ly tung layer.
F3M/F3MWithSA giu nguyen so channel (residual), nen them nhanh sau:
```python
if m in (F3M, F3MWithSA):
    c1 = ch[f]          # in_channels lay tu layer truoc
    c2 = c1             # out_channels == in_channels (residual)
    args = [c1, *args]  # chen in_channels vao dau danh sach args
```
Dat doan nay TRUOC khi `args` duoc dung de khoi tao module
(cung cho voi nhung module khac nhu Conv/C3k2 trong if/elif chain).

> Luu y: trong YAML, args cua F3MWithSA la `[reduction_ratio, use_gate, sa_kernel_size]`
> va cua F3M la `[reduction_ratio, use_gate]`. Sau khi parser chen `c1` vao dau,
> chu ky constructor `F3M(in_channels, reduction_ratio, use_gate, stride)` va
> `F3MWithSA(in_channels, reduction_ratio, use_gate, sa_kernel_size, stride)` se khop.

## Buoc 4 — Train

```python
from ultralytics import YOLO

model = YOLO("yolo12n-f3m.yaml")
model.train(
    data="scoraldet.yaml", epochs=300, imgsz=640, batch=16,
    device=0, seed=42, optimizer="auto",
    lr0=0.01, momentum=0.937, weight_decay=0.0005,
    fliplr=0.5, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    scale=0.5, translate=0.1, mosaic=1.0, erasing=0.4,
)
```

## Buoc 5 — Kiem tra nhanh module

```bash
python f3m_modules.py
# In ra shape dau vao/ra + so tham so cua tung module
```

---

## Bang tom tat

| Module       | Vi tri      | reduction_ratio | gate  | Spatial Attn |
|--------------|-------------|-----------------|-------|--------------|
| F3MWithSA    | Stem (early)| 0.33            | True  | Co (Conv7x7) |
| F3M          | Neck (SPPF) | 0.125           | False | Khong        |

Thiet ke residual: `in_channels == out_channels` -> plug-and-play, khong
lam thay doi phan con lai cua mang.
