"""
Quick script: in Params va GFLOPs cua 10 model benchmark
Chay tu: F:\LightCoral-YOLO\
    python scripts/model_info.py
"""

import re
import logging
from ultralytics import YOLO
import ultralytics.utils as ult_utils
from pathlib import Path

BASE = Path('runs/detect/runs/coral_benchmark')

MODELS = [
    'yolov8n_imgsz640_ep100',
    'yolov8s_imgsz640_ep100',
    'yolov10n_imgsz640_ep100',
    'yolov10s_imgsz640_ep100',
    'yolo11n_imgsz640_ep100',
    'yolo11s_imgsz640_ep100',
    'yolo12n_imgsz640_ep100',
    'yolo12s_imgsz640_ep100',
    'yolo26n_imgsz640_ep100',
    'yolo26s_imgsz640_ep100',
    'rtdetr-r50_imgsz640_ep100',
]


class _CaptureHandler(logging.Handler):
    """Capture ultralytics LOGGER output vao string."""
    def __init__(self):
        super().__init__()
        self.records = []
    def emit(self, record):
        self.records.append(self.format(record))
    def flush_text(self):
        txt = '\n'.join(self.records)
        self.records.clear()
        return txt


capture = _CaptureHandler()
ult_utils.LOGGER.addHandler(capture)

print(f'\n{"Model":<22} {"Params (M)":>12} {"GFLOPs":>10}')
print('-' * 46)

for run_name in MODELS:
    weights = BASE / run_name / 'weights' / 'best.pt'
    label   = run_name.replace('_imgsz640_ep100', '')

    if not weights.exists():
        print(f'{label:<22} {"NOT FOUND":>12}')
        continue

    try:
        capture.records.clear()
        model = YOLO(str(weights))
        model.fuse()  # merge BN vao Conv -- khop voi cach ultralytics report

        # Params tu fused model (giong ultralytics official)
        n_params = sum(p.numel() for p in model.model.parameters()) / 1e6

        # Trigger LOGGER output, capture GFLOPs
        capture.records.clear()
        model.info(verbose=True, imgsz=640)
        log_text = capture.flush_text()

        match = re.search(r'([\d.]+)\s*GFLOPs', log_text)
        gflops = float(match.group(1)) if match else '?'

        print(f'{label:<22} {round(n_params, 3):>12} {gflops:>10}')

    except Exception as e:
        print(f'{label:<22} ERROR: {e}')

ult_utils.LOGGER.removeHandler(capture)
print('-' * 46)
