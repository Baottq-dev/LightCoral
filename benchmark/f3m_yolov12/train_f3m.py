# train_f3m.py
"""
Train YOLOv12 + F3M.

Vi du:
    python train_f3m.py --cfg yolo12n-f3m.yaml --data coral_dataset.yaml --epochs 300
    python train_f3m.py --cfg yolo12n-f3m-stem-only.yaml --name f3m_stem
    python train_f3m.py --cfg yolo12n-f3m-neck-only.yaml --name f3m_neck

Luu y: import f3m_register TRUOC khi tao YOLO de dang ky module.
"""

import argparse

import torch
import f3m_register            # noqa: F401  (dang ky F3M / F3MWithSA)
from ultralytics import YOLO

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("OMP_NUM_THREADS", "1")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg",  default="yolo12n-f3m.yaml", help="model yaml (kien truc)")
    p.add_argument("--data", default="coral_dataset.yaml", help="dataset yaml")
    p.add_argument("--weights", default="", help="pretrained .pt (de trong = train from scratch)")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--imgsz",  type=int, default=640)
    p.add_argument("--batch",  type=int, default=16)
    p.add_argument("--device", default="auto", help="'auto' (tu chon GPU/CPU) / '0' / '0,1' / 'cpu'")
    p.add_argument("--name",   default="yolo12n_f3m")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def main():
    a = parse_args()

    # Tu dong chon device: dung GPU neu kha dung, nguoc lai tu chuyen sang CPU.
    # Tranh loi "Invalid CUDA 'device=0'" khi cai ban torch CPU-only.
    device = a.device
    if device == "auto":
        device = 0 if torch.cuda.is_available() else "cpu"
    elif device not in ("cpu", "") and not torch.cuda.is_available():
        print(f"[train_f3m] CUDA khong kha dung -> chuyen device='{device}' sang 'cpu'.")
        device = "cpu"
    print(f"[train_f3m] Su dung device = {device} (cuda.is_available={torch.cuda.is_available()})")

    # Khoi tao model. Co the load pretrained backbone bang .load() neu can.
    model = YOLO(a.cfg)
    if a.weights:
        model = model.load(a.weights)   # nap trong so tuong thich (transfer learning)

    model.train(
        data=a.data,
        epochs=a.epochs,
        imgsz=a.imgsz,
        batch=a.batch,
        device=device,
        name=a.name,
        resume=a.resume,
        # ---- optimizer / lr (canh chuan theo cau hinh YOLO trong bai bao) ----
        optimizer="SGD",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        # ---- data augmentation ----
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        translate=0.1, scale=0.5, fliplr=0.5,
        mosaic=1.0, close_mosaic=10,
        erasing=0.4,
        # ---- misc ----
        seed=42,
        patience=50,
        val=True,
        plots=True,
    )

    # Danh gia tren tap val sau khi train
    metrics = model.val()
    print("mAP50    :", metrics.box.map50)
    print("mAP50-95 :", metrics.box.map)


if __name__ == "__main__":
    main()