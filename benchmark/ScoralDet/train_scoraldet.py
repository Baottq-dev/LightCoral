# benchmark/scoraldet/train_scoraldet.py
# Train SCoralDet tren Soft-Coral - CUNG protocol voi SC-YOLO12 (so sanh cong bang).
# Chay tu ROOT repo:
#   python -m benchmark.scoraldet.train_scoraldet --data data/scoraldet_fold0.yaml --seed 0
#   python -m benchmark.scoraldet.train_scoraldet --data ... --seed 0 --scratch          # tu scratch
#   python -m benchmark.scoraldet.train_scoraldet --data ... --seed 0 --paper-protocol   # 300ep/SGD nhu paper
# TAI DUNG: build_scoraldet (builder), _Tee + train_defaults + set_seed. Inject v10APTDetectionLoss.

import argparse
import sys
import types
from pathlib import Path

# chay truc tiep: dam bao root repo (sc-yolo12/) trong sys.path
_ROOT_INIT = Path(__file__).resolve().parents[2]
if str(_ROOT_INIT) not in sys.path:
    sys.path.insert(0, str(_ROOT_INIT))

import yaml
from ultralytics.models.yolo.detect import DetectionTrainer

ROOT = Path(__file__).resolve().parents[2]          # sc-yolo12/ (khong import tu build_scoraldet tranh circular)

# ho tro chay ca '-m benchmark.scoraldet.train_scoraldet' lan 'python benchmark/scoraldet/train_scoraldet.py'
try:
    from benchmark.scoraldet.build_scoraldet import build_scoraldet_yaml, register_scoraldet_modules
    from benchmark.scoraldet.apt_loss import v10APTDetectionLoss
except ImportError:
    from build_scoraldet import build_scoraldet_yaml, register_scoraldet_modules
    from apt_loss import v10APTDetectionLoss
from train import _Tee                      # tai dung tee-log (root/train.py)
from utils.seed import set_seed


class SCoralDetTrainer(DetectionTrainer):
    """DetectionTrainer chuan nhung thay loss = v10APTDetectionLoss (APT label assignment).
    APT_POWER/APT_THR doc tu class attr (set boi main truoc khi tao trainer)."""

    APT_POWER = 2.0
    APT_THR = 0.5

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
        power, thr = self.APT_POWER, self.APT_THR
        # override init_criterion -> v10APTDetectionLoss (giu chu ky preds dict one2many/one2one)
        model.init_criterion = types.MethodType(
            lambda m: v10APTDetectionLoss(m, power=power, thr=thr), model
        )
        return model


def main():
    ap = argparse.ArgumentParser("SCoralDet trainer (benchmark)")
    ap.add_argument("--data", required=True, help="data YAML (CUNG split co dinh voi SC-YOLO12)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--scratch", action="store_true",
                    help="train tu dau (giong paper); mac dinh dung pretrained yolov10n.pt")
    ap.add_argument("--weights", default="yolov10n.pt",
                    help="pretrained khoi tao (chi nap layer khop ten+shape)")
    ap.add_argument("--paper-protocol", action="store_true",
                    help="tai hien dieu kien paper: 300ep, batch32, SGD lr0.01 mom0.937")
    ap.add_argument("--apt-power", type=float, default=2.0, help="so mu p cua APT (paper=2)")
    ap.add_argument("--apt-thr", type=float, default=0.5, help="nguong T cua APT (gia dinh 0.5)")
    ap.add_argument("--specs", default=str(ROOT / "cfg" / "module_specs.yaml"))
    ap.add_argument("--project", default=str(ROOT / "benchmark" / "runs"))
    ap.add_argument("--name", default="SCoralDet")
    ap.add_argument("--logfile", default=None)
    args = ap.parse_args()

    specs = yaml.safe_load(Path(args.specs).read_text())
    td = specs["train_defaults"]

    set_seed(args.seed)               # python/numpy/torch + cudnn deterministic
    register_scoraldet_modules()      # PHAI goi truoc khi parse YAML model
    model_yaml = build_scoraldet_yaml(nc=6)

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

    # ---- protocol: mac dinh = train_defaults (fair voi SC-YOLO12); --paper-protocol = dieu kien goc ----
    overrides = dict(
        model=str(model_yaml),
        data=args.data,
        epochs=args.epochs or td["epochs"],
        imgsz=args.imgsz or td["imgsz"],
        batch=args.batch or td["batch"],
        patience=td["patience"],
        optimizer=td["optimizer"],
        lr0=td["lr0"], lrf=td["lrf"],
        weight_decay=td["weight_decay"],
        warmup_epochs=td["warmup_epochs"],
        mosaic=td["mosaic"], flipud=td["flipud"], fliplr=td["fliplr"],
        degrees=td["degrees"], hsv_h=td["hsv_h"], hsv_s=td["hsv_s"], hsv_v=td["hsv_v"],
        device=args.device if args.device is not None else td["device"],
        workers=args.workers if args.workers is not None else td["workers"],
        pretrained=False if args.scratch else args.weights,
        seed=args.seed,
        deterministic=True,
        project=args.project,
        name=f"{args.name}_s{args.seed}",
        exist_ok=True,
    )
    if args.paper_protocol:
        # SCoralDet Sec 4.2: 300ep, batch32, SGD lr0.01 momentum0.937, scale 0.5, translate 0.1
        overrides.update(
            epochs=args.epochs or 300,
            batch=args.batch or 32,
            optimizer="SGD",
            lr0=0.01, momentum=0.937,
            scale=0.5, translate=0.1,
        )

    SCoralDetTrainer.APT_POWER = args.apt_power
    SCoralDetTrainer.APT_THR = args.apt_thr
    trainer = SCoralDetTrainer(overrides=overrides)
    try:
        trainer.train()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        if not _log_fh.closed:
            _log_fh.close()


if __name__ == "__main__":
    main()