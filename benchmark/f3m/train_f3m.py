# benchmark/f3m/train_f3m.py
# Train F3M-YOLO11n tren Soft-Coral - CUNG protocol voi SC-YOLO12 (so sanh cong bang).
# Chay tu ROOT repo:
#   python -m benchmark.f3m.train_f3m --data data/scoraldet_fold0.yaml --seed 0
#   python -m benchmark.f3m.train_f3m --data ... --seed 0 --scratch     # tu scratch (giong paper)
# TAI DUNG: build_f3m (builder), _Tee + train_defaults + set_seed.
# F3M KHONG them loss -> dung DetectionTrainer chuan (khong subclass).

import argparse
import sys
from pathlib import Path

# chay truc tiep: dam bao root repo (sc-yolo12/) trong sys.path
_ROOT_INIT = Path(__file__).resolve().parents[2]
if str(_ROOT_INIT) not in sys.path:
    sys.path.insert(0, str(_ROOT_INIT))

import yaml
from ultralytics.models.yolo.detect import DetectionTrainer

ROOT = Path(__file__).resolve().parents[2]          # sc-yolo12/

# ho tro chay ca '-m benchmark.f3m.train_f3m' lan 'python benchmark/f3m/train_f3m.py'
try:
    from benchmark.f3m.build_f3m import build_f3m_yaml, register_f3m_modules
except ImportError:
    from build_f3m import build_f3m_yaml, register_f3m_modules
from train import _Tee                      # tai dung tee-log (root/train.py)
from utils.seed import set_seed


def main():
    ap = argparse.ArgumentParser("F3M trainer (benchmark)")
    ap.add_argument("--data", required=True, help="data YAML (CUNG split co dinh voi SC-YOLO12)")
    ap.add_argument("--seed", type=int, default=42,
                    help="hat giong ngau nhien co dinh (paper=42, toan bo luot chay)")
    # ---- Sieu tham so co dinh theo paper F3M ----
    ap.add_argument("--epochs", type=int, default=None,
                    help="so epoch (linh hoat tuy dataset; cac tham so khac giu co dinh)")
    ap.add_argument("--imgsz", type=int, default=640,
                    help="kich thuoc anh dau vao (paper=640x640, co dinh)")
    ap.add_argument("--batch", type=int, default=16,
                    help="kich thuoc lo (paper=16, co dinh)")
    ap.add_argument("--optimizer", type=str, default="auto",
                    help="bo toi uu hoa (paper=auto/Adam-based, co dinh)")
    ap.add_argument("--lr0", type=float, default=0.01,
                    help="toc do hoc ban dau (paper=0.01, co dinh)")
    ap.add_argument("--lrf", type=float, default=0.01,
                    help="toc do hoc cuoi cung (paper=0.01)")
    ap.add_argument("--weight-decay", type=float, default=0.0005,
                    help="he so phan ra trong so (paper=0.0005, co dinh)")
    ap.add_argument("--momentum", type=float, default=0.937,
                    help="dong luong (paper=0.937, co dinh)")
    # ---- Augmentation (co dinh theo paper F3M) ----
    ap.add_argument("--fliplr", type=float, default=0.5,
                    help="lat ngang ngau nhien xac suat 50%% (paper=0.5)")
    ap.add_argument("--hsv-h", type=float, default=0.015,
                    help="bien do hue HSV +/-1.5%% (paper=0.015)")
    ap.add_argument("--hsv-s", type=float, default=0.7,
                    help="bien do saturation HSV +/-70%% (paper=0.7)")
    ap.add_argument("--hsv-v", type=float, default=0.4,
                    help="bien do value/brightness HSV +/-40%% (paper=0.4)")
    ap.add_argument("--scale", type=float, default=0.5,
                    help="co gian ngau nhien 0.5-1.0 (paper=0.5)")
    ap.add_argument("--translate", type=float, default=0.1,
                    help="dich chuyen ngau nhien toi da 10%% (paper=0.1)")
    ap.add_argument("--erasing", type=float, default=0.4,
                    help="xac suat random erasing ket hop RandAugment (paper=0.4)")
    # ---- Runtime ----
    ap.add_argument("--device", default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--scratch", action="store_true",
                    help="train tu dau (giong paper); mac dinh dung pretrained yolo11n.pt")
    ap.add_argument("--weights", default="yolo11n.pt",
                    help="pretrained khoi tao (chi nap layer khop ten+shape)")
    ap.add_argument("--specs", default=str(ROOT / "cfg" / "module_specs.yaml"))
    ap.add_argument("--project", default=str(ROOT / "benchmark" / "runs"))
    ap.add_argument("--name", default="F3M")
    ap.add_argument("--logfile", default=None)
    args = ap.parse_args()

    specs = yaml.safe_load(Path(args.specs).read_text())
    td = specs["train_defaults"]

    set_seed(args.seed)               # python/numpy/torch + cudnn deterministic
    register_f3m_modules()            # PHAI goi truoc khi parse YAML model
    model_yaml = build_f3m_yaml(nc=6)

    # ---- tee-log giong SC-YOLO12: phan chieu stdout/stderr ra file trong run dir ----
    run_dir = Path(args.project) / f"{args.name}_s{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.logfile) if args.logfile else run_dir / "train_log.txt"
    _log_fh = open(log_path, "w", encoding="utf-8", buffering=1)
    _log_fh.write("python " + " ".join(sys.argv) + "\n")
    if sys.__stdout__ is not None:
        sys.__stdout__.write(f"[log] Console -> {log_path}\n")
    sys.stdout = _Tee(sys.__stdout__, _log_fh)
    sys.stderr = _Tee(sys.__stderr__, _log_fh)
    from ultralytics.utils import LOGGER
    for _h in LOGGER.handlers:
        if hasattr(_h, "setStream") and getattr(_h, "stream", None) in (sys.__stdout__, sys.__stderr__):
            _h.setStream(sys.stderr)

    # ---- Sieu tham so co dinh theo paper F3M ----
    # imgsz=640, batch=16, optimizer=auto (Adam-based), lr0=0.01, momentum=0.937,
    # weight_decay=0.0005, seed=42 (tat ca luot chay); epochs linh hoat theo dataset.
    # Augmentation: fliplr=0.5, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    #               scale=0.5 (0.5-1.0), translate=0.1, mosaic,
    #               RandAugment + random erasing (erasing=0.4).
    overrides = dict(
        model=str(model_yaml),
        data=args.data,
        # --- Sieu tham so paper (co dinh) ---
        epochs=args.epochs or td["epochs"],   # linh hoat tuy dataset
        imgsz=args.imgsz,                     # 640
        batch=args.batch,                     # 16
        optimizer=args.optimizer,             # auto (Adam-based)
        lr0=args.lr0,                         # 0.01
        lrf=args.lrf,                         # 0.01
        weight_decay=args.weight_decay,       # 0.0005
        momentum=args.momentum,               # 0.937
        # --- Augmentation paper F3M (co dinh) ---
        fliplr=args.fliplr,                   # 0.5  - lat ngang 50%
        hsv_h=args.hsv_h,                     # 0.015 - hue +/-1.5%
        hsv_s=args.hsv_s,                     # 0.7   - saturation +/-70%
        hsv_v=args.hsv_v,                     # 0.4   - brightness +/-40%
        scale=args.scale,                     # 0.5   - co gian 0.5-1.0
        translate=args.translate,             # 0.1   - dich chuyen toi da 10%
        mosaic=td["mosaic"],                  # 1.0   - mosaic epoch dau
        auto_augment="randaugment",           # RandAugment
        erasing=args.erasing,                 # 0.4   - random erasing p=0.4
        flipud=td["flipud"],
        degrees=td["degrees"],
        # --- Lay tu train_defaults ---
        patience=td["patience"],
        warmup_epochs=td["warmup_epochs"],
        # --- Runtime ---
        device=args.device if args.device is not None else td["device"],
        workers=args.workers if args.workers is not None else td["workers"],
        pretrained=False if args.scratch else args.weights,
        seed=args.seed,
        deterministic=True,
        project=args.project,
        name=f"{args.name}_s{args.seed}",
        exist_ok=True,
        plots=True,
    )

    trainer = DetectionTrainer(overrides=overrides)
    try:
        trainer.train()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        # Khoi phuc stdout/stderr goc TRUOC khi dong file.
        # Neu khong, Python GC se co flush _Tee sau khi _log_fh da dong
        # => "Exception ignored in: <train._Tee object>" khi thoat.
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        if not _log_fh.closed:
            _log_fh.close()


if __name__ == "__main__":
    main()