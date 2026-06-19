# F3M-YOLO11n (reimplemented)

Reimplement **F3M** (Frequency-domain Feature Fusion Module, Wang et al., *J. Mar. Sci. Eng.* 2026, 14, 20) trên base **YOLO11n** làm đối thủ benchmark cho SC-YOLO12 trên Soft-Coral. F3M là khối *plug-and-play* giữ nguyên kênh, **không thêm loss phụ** — tối ưu hoàn toàn qua detection loss chuẩn.

## Cách chạy

```bash
# Smoke-test build (forward 640, kiem 3 head + stride + params ~2.61M)
python benchmark/f3m/build_f3m.py

# Smoke-test rieng module (params ~0.026M, giu shape)
python benchmark/f3m/modules_f3m.py

# Train (cung split + protocol voi SC-YOLO12)
python -m benchmark.f3m.train_f3m --data data/scoraldet_fold0.yaml --seed 0
python -m benchmark.f3m.train_f3m --data ... --seed 0 --scratch   # tu scratch nhu paper
```

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