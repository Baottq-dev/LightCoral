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

# 2) Train 1 seed (CUNG split/imgsz voi SC-YOLO12)
python -m benchmark.sf_yolo.train_sf_yolo --data data/scoraldet_fold0.yaml --seed 0

# 3) Multi-seed [0,1,2] (mean +/- std)
for s in 0 1 2; do
  python -m benchmark.sf_yolo.train_sf_yolo --data data/scoraldet_fold0.yaml --seed $s
done

# 4) Tuy chon: tu scratch giong dieu kien goc paper (danh dau rieng trong bang)
python -m benchmark.sf_yolo.train_sf_yolo --data data/scoraldet_fold0.yaml --seed 0 --scratch
```

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