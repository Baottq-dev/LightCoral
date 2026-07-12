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

# ho tro chay ca 3 tinh huong:
#   1) python -m benchmark.ScoralDet.train_scoraldet  (ten thu muc chinh xac)
#   2) python -m benchmark.scoraldet.train_scoraldet  (sau khi doi ten ve lowercase)
#   3) python benchmark/ScoralDet/train_scoraldet.py  (flat, can sys.path)
_HERE = Path(__file__).resolve().parent  # .../benchmark/ScoralDet/
try:
    from benchmark.ScoralDet.build_scoraldet import build_scoraldet_yaml, register_scoraldet_modules
    from benchmark.ScoralDet.apt_loss import v10APTDetectionLoss
except ModuleNotFoundError:
    try:
        from benchmark.scoraldet.build_scoraldet import build_scoraldet_yaml, register_scoraldet_modules
        from benchmark.scoraldet.apt_loss import v10APTDetectionLoss
    except ModuleNotFoundError:
        import sys as _sys
        if str(_HERE) not in _sys.path:
            _sys.path.insert(0, str(_HERE))
        from build_scoraldet import build_scoraldet_yaml, register_scoraldet_modules
        from apt_loss import v10APTDetectionLoss
from train import _Tee                      # tai dung tee-log (root/train.py)
from utils.seed import set_seed

# Gia tri mac dinh APT (khop voi SCoralDetTrainer.APT_POWER / APT_THR)
APT_POWER_DEFAULT: float = 2.0
APT_THR_DEFAULT: float = 0.5


def _apt_init_criterion(model):
    """Module-level function (picklable) thay cho lambda."""
    return v10APTDetectionLoss(model, power=model._apt_power, thr=model._apt_thr)


def _patch_apt(model, power: float, thr: float):
    """Gan _apt_power/_apt_thr va override init_criterion len model instance.
    Phai goi sau moi lan load model (get_model, final_eval, eval script)."""
    model._apt_power = power
    model._apt_thr = thr
    model.init_criterion = types.MethodType(_apt_init_criterion, model)



class SCoralDetTrainer(DetectionTrainer):
    """DetectionTrainer chuan nhung thay loss = v10APTDetectionLoss (APT label assignment).
    APT_POWER/APT_THR doc tu class attr (set boi main truoc khi tao trainer).
    SCR_CHI/SCR_DELTA: tham so vung trung tam mem (soft center region) - paper Sec 4.2: chi=10, delta=3."""

    APT_POWER = 2.0
    APT_THR = 0.5
    # Soft center region loss hyperparameters (SCoralDet Sec 4.2)
    SCR_CHI: float = 10.0   # chi (χ) - co so mu phat lech tam
    SCR_DELTA: float = 3.0  # delta (δ) - nguong vung trung tam

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
        _patch_apt(model, self.APT_POWER, self.APT_THR)
        return model

    def save_model(self):
        """Strip init_criterion truoc khi save checkpoint.

        Ultralytics save_model() serialize deepcopy(self.ema.ema), KHONG phai
        self.model truc tiep. ema.ema la deep-copy cua model sau _patch_apt,
        nen no cung co init_criterion/_apt_* trong __dict__.
        Phai strip ca 2 truoc khi super().save_model() chay, khoi phuc sau.
        """
        _STRIP = ("init_criterion", "_apt_power", "_apt_thr")
        targets = [self.model]
        if hasattr(self, "ema") and hasattr(self.ema, "ema") and self.ema.ema is not None:
            targets.append(self.ema.ema)
        saved = [{k: t.__dict__.pop(k) for k in _STRIP if k in t.__dict__} for t in targets]
        try:
            super().save_model()
        finally:
            for t, s in zip(targets, saved):
                t.__dict__.update(s)   # khoi phuc de tiep tuc train


def main():
    ap = argparse.ArgumentParser("SCoralDet trainer (benchmark)")
    ap.add_argument("--data", required=True, help="data YAML (CUNG split co dinh voi SC-YOLO12)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--optimizer", type=str, default="SGD")
    ap.add_argument("--lr0", type=float, default=0.01)
    ap.add_argument("--lrf", type=float, default=0.01)
    ap.add_argument("--weight_decay", type=float, default=0.0005)
    ap.add_argument("--warmup_epochs", type=int, default=3)
    ap.add_argument("--momentum", type=float, default=0.937)
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
    # Soft center region loss (SCoralDet Eq 6-7, Sec 4.2)
    ap.add_argument("--soft-chi", type=float, default=10.0,
                    help="chi (chi) - co so mu phat lech tam cua SoftCenterConfLoss (paper=10)")
    ap.add_argument("--soft-delta", type=float, default=3.0,
                    help="delta - nguong vung trung tam cua SoftCenterConfLoss (paper=3)")
    # Data augmentation (co dinh theo SCoralDet paper Sec 4.2)
    ap.add_argument("--fliplr", type=float, default=0.5,
                    help="xac suat lat ngang ngau nhien (paper=0.5)")
    ap.add_argument("--hsv-h", type=float, default=0.015,
                    help="bien doi ton mau HSV hue (+/-1.5%%, paper=0.015)")
    ap.add_argument("--hsv-s", type=float, default=0.4,
                    help="tang cuong do bao hoa HSV saturation (+/-40%%, paper=0.4)")
    ap.add_argument("--hsv-v", type=float, default=0.4,
                    help="tang cuong do sang HSV brightness (+/-40%%, paper=0.4)")
    ap.add_argument("--scale", type=float, default=0.5,
                    help="thay doi kich thuoc ngau nhien (50%%~100%% goc, paper scale=0.5)")
    ap.add_argument("--translate", type=float, default=0.1,
                    help="dich chuyen anh ngau nhien toi da 10%% (paper=0.1)")
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
    # Data Augmentation (SCoralDet paper Sec 4.2):
    #   fliplr=0.5  : lat ngang ngau nhien 50%
    #   hsv_h=0.015 : bien doi ton mau +/-1.5%
    #   hsv_s=0.4   : tang bao hoa +/-40% (ghi de td hsv_s=0.7)
    #   hsv_v=0.4   : tang do sang +/-40%
    #   scale=0.5   : thay doi kich thuoc ngau nhien 50%~100%
    #   translate=0.1: dich chuyen toi da 10%
    overrides = dict(
        model=str(model_yaml),
        data=args.data,
        epochs=args.epochs or td["epochs"],
        imgsz=args.imgsz or td["imgsz"],
        batch=args.batch or td["batch"],
        patience=td["patience"],
        optimizer=args.optimizer or td["optimizer"],
        lr0=args.lr0 or td["lr0"], lrf=args.lrf or td["lrf"],
        weight_decay=args.weight_decay or td["weight_decay"],
        warmup_epochs=args.warmup_epochs or td["warmup_epochs"],
        momentum=args.momentum or td["momentum"],
        # --- Augmentation (SCoralDet Sec 4.2, ghi de train_defaults neu khac) ---
        mosaic=td["mosaic"],
        flipud=td["flipud"],
        fliplr=args.fliplr,           # 0.5  - lat ngang 50%
        degrees=td["degrees"],
        hsv_h=args.hsv_h,             # 0.015 - +/-1.5% hue
        hsv_s=args.hsv_s,             # 0.4   - +/-40% saturation (paper; ghi de td=0.7)
        hsv_v=args.hsv_v,             # 0.4   - +/-40% brightness
        scale=args.scale,             # 0.5   - co gian 50%~100%
        translate=args.translate,     # 0.1   - dich chuyen toi da 10%
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
    if args.paper_protocol:
        # SCoralDet Sec 4.2: 300ep, batch32, SGD lr0.01 momentum0.937
        # Augmentation da co san trong overrides mac dinh (scale/translate/fliplr/hsv_*)
        overrides.update(
            epochs=args.epochs or 300,
            batch=args.batch or 32,
            optimizer="SGD",
            lr0=0.01,
            momentum=0.937,
        )

    SCoralDetTrainer.APT_POWER = args.apt_power
    SCoralDetTrainer.APT_THR = args.apt_thr
    # Soft center region loss (chi=10, delta=3 theo paper Sec 4.2)
    SCoralDetTrainer.SCR_CHI = args.soft_chi
    SCoralDetTrainer.SCR_DELTA = args.soft_delta
    trainer = SCoralDetTrainer(overrides=overrides)
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