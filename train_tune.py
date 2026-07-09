# train_tune.py  (dat o ROOT repo sc-yolo12/, giong train.py)
# Ban sao train.py + them --cos_lr / --momentum + tu dat ten thu muc theo config tune.
# Vi du:
#   python train_tune.py --data cfg/coral_soft_yolo.yaml --modules 3,4 --seed 0 \
#       --epochs 200 --optimizer SGD --lr0 0.01 --lrf 0.001 --weight_decay 0.001 --cos_lr
#   -> runs/scyolo12/M34__SGD_lr0-0p01_lrf-0p001_cos-1_wd-0p001_s0/

import argparse
import re
import sys
from pathlib import Path

import yaml

from ultralytics.models.yolo.detect import DetectionTrainer

from augment.physics_degradation import from_specs
from engine.build_model import ROOT, build_yaml_for_modules
from engine.losses import DegradationLoss, SCDetectionModel
from models.registry import register_custom_modules
from utils.seed import set_seed


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


class _Tee:
    """Phan chieu stdout/stderr ra file GIONG console (giong M1_s42.txt).

    - Xu ly '\r' cua tqdm: chi giu trang thai CUOI cung cua moi dong (khong nhoe).
    - Loai ma mau ANSI => file sach.
    - Console giu nguyen (isatty uy quyen cho stream goc => tqdm van chay live).
    """

    def __init__(self, console, fh):
        self.console = console
        self.fh = fh
        self._line = ""          # buffer dong hien tai (sau '\r' cuoi cung)

    def write(self, data):
        if self.console is not None:
            self.console.write(data)
        for ch in data:
            if ch == "\r":
                self._line = ""
            elif ch == "\n":
                self.fh.write(_ANSI_RE.sub("", self._line) + "\n")
                self.fh.flush()
                self._line = ""
            else:
                self._line += ch
        return len(data)

    def flush(self):
        if self.console is not None:
            self.console.flush()
        self.fh.flush()

    def isatty(self):
        return getattr(self.console, "isatty", lambda: False)()


def parse_modules(arg_modules: str, arg_preset: str, presets: dict) -> set[int]:
    if arg_preset:
        assert not arg_modules, "Chi dung MOT trong --modules hoac --preset"
        assert arg_preset in presets, f"Preset la: {list(presets)}"
        return set(presets[arg_preset])
    if not arg_modules.strip():
        return set()
    return {int(x) for x in arg_modules.split(",") if x.strip()}


def _fmt(x) -> str:
    """Format so gon & an toan cho ten thu muc: 0.01->0p01, 0.0005->0p0005, 1e-3->0p001, -1->m1."""
    return f"{x:g}".replace("-", "m").replace(".", "p")


class SCTrainer(DetectionTrainer):
    """DetectionTrainer + (3) physics aug trong preprocess_batch + (4) L_deg."""

    sc_modules: set = set()
    sc_specs: dict = {}
    physics_aug = None  # BatchPhysicsDegradation | None

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = SCDetectionModel(cfg, nc=self.data["nc"], verbose=verbose)
        if 4 in self.sc_modules:
            w = self.sc_specs["modules"]["pg_dam"]["loss_weight"]
            model.deg_loss = DegradationLoss(weight=w)
        if weights:
            if 4 in self.sc_modules:
                # (4) da chen 1 layer => moi index >= pos bi +1. Phai doi ten key
                # checkpoint cho khop, neu khong toan bo backbone se KHONG nap duoc.
                self._load_pretrained_shifted(model, weights)
            else:
                model.load(weights)  # COCO-pretrained: chi nap layer khop ten/shape
        return model

    def _load_pretrained_shifted(self, model, weights):
        """Nap COCO-pretrained khi (4) PG-DAM da chen 1 layer tai `pos`.

        build_model.py chen PGDAM_FiLM tai idx `pos` (=insert_after_idx+1) va +1
        cho moi index >= pos. Checkpoint yolo12n.pt van danh so theo kien truc GOC,
        nen ten key 'model.<i>.*' lech 1 so voi model da reindex => model.load mac
        dinh chi khop moi layer 0 (stem). Ta doi 'model.<i>.*' -> 'model.<i+1>.*'
        voi moi i >= pos truoc khi intersect => backbone duoc khoi phuc pretrained.
        Cac layer moi (PG-DAM/SFDF/FGA2/nhanh P2) van random nhu mong doi.
        """
        import re
        from ultralytics.utils import LOGGER
        from ultralytics.utils.torch_utils import intersect_dicts

        pos = self.sc_specs["modules"]["pg_dam"]["insert_after_idx"] + 1
        csd = weights.float().state_dict() if hasattr(weights, "float") else weights
        remapped = {}
        for k, v in csd.items():
            mobj = re.match(r"^(model\.)(\d+)(\..*)$", k)
            if mobj:
                idx = int(mobj.group(2))
                if idx >= pos:
                    idx += 1
                k = f"{mobj.group(1)}{idx}{mobj.group(3)}"
            remapped[k] = v
        msd = model.state_dict()
        inter = intersect_dicts(remapped, msd)  # khop ten + shape
        model.load_state_dict(inter, strict=False)
        LOGGER.info(
            f"PG-DAM remap pretrained: nap {len(inter)}/{len(msd)} tensor (pos={pos}) "
            f"- backbone khoi phuc COCO; cac layer moi giu random."
        )
        return model

    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)  # img -> float[0,1] tren device
        if self.physics_aug is not None:
            batch["img"], batch["z_gt"] = self.physics_aug(batch["img"])
        return batch


def main():
    ap = argparse.ArgumentParser("SC-YOLO12 tuning trainer (S1 sweep)")
    ap.add_argument("--data", required=True, help="data YAML (1 fold cua GroupKFold)")
    ap.add_argument("--modules", default="", help="vd: 1,2,4 (rong = baseline)")
    ap.add_argument("--preset", default="", help="B0 | E1..E8 (thay cho --modules)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--weights", default=None, help="mac dinh theo module_specs.yaml")
    ap.add_argument("--specs", default=str(ROOT / "cfg" / "module_specs.yaml"))
    ap.add_argument("--project", default=str(ROOT / "runs" / "scyolo12"))  # tuyet doi -> tranh bi long runs/detect/...
    ap.add_argument("--device", default=None)
    # Training hyperparameters
    ap.add_argument("--optimizer",     type=str,   default=None,
                    choices=["SGD", "Adam", "AdamW", "NAdam", "RAdam", "RMSProp", "auto"])
    ap.add_argument("--lr0",           type=float, default=None, help="Initial learning rate")
    ap.add_argument("--lrf",           type=float, default=None, help="Final LR factor (lr0 * lrf)")
    ap.add_argument("--weight_decay",  type=float, default=None)
    ap.add_argument("--warmup_epochs", type=int,   default=None)
    # >>> MOI so voi train.py: cosine LR + momentum <<<
    ap.add_argument("--cos_lr", action="store_true",
                    help="Bat cosine LR scheduler (mac dinh: linear nhu Ultralytics)")
    ap.add_argument("--momentum", type=float, default=None,
                    help="SGD momentum / Adam beta1 (chi tune o buoc cuoi tren winner)")
    # Augmentation
    ap.add_argument("--mosaic",  type=float, default=None, help="Mosaic probability")
    ap.add_argument("--flipud",  type=float, default=None, help="Vertical flip probability")
    ap.add_argument("--fliplr",  type=float, default=None, help="Horizontal flip probability")
    ap.add_argument("--degrees", type=float, default=None, help="Rotation degrees")
    ap.add_argument("--hsv_h",   type=float, default=None, help="HSV hue shift")
    ap.add_argument("--hsv_s",   type=float, default=None, help="HSV saturation shift")
    ap.add_argument("--hsv_v",   type=float, default=None, help="HSV value shift")
    # Runtime
    ap.add_argument("--workers", type=int, default=None, help="Dataloader workers (0 on Windows)")
    ap.add_argument("--logfile", default=None,
                    help="File log console. Mac dinh: <project>/<run_name>/train_log.txt")
    args = ap.parse_args()

    specs = yaml.safe_load(Path(args.specs).read_text())
    td = specs["train_defaults"]
    mods = parse_modules(args.modules, args.preset, specs["presets"])

    set_seed(args.seed)               # python/numpy/torch + cudnn deterministic
    register_custom_modules()         # PHAI goi truoc khi parse YAML model

    # doc nc tu data YAML (linh hoat, giong train.py) thay vi cung nc=6
    data_cfg = yaml.safe_load(Path(args.data).read_text(encoding="utf-8"))
    nc = data_cfg.get("nc", 6)
    # so module 3 (aug) khong doi kien truc; YAML chi phu thuoc {1,2,4,5}
    model_yaml = build_yaml_for_modules(mods, specs, nc=nc)

    tag = "".join(str(m) for m in sorted(mods)) or "0"
    name = args.preset or f"M{tag}"

    # ---- Gia tri hieu luc (CLI override > train_defaults) de dua vao ten & overrides ----
    eff_opt = args.optimizer or td["optimizer"]
    eff_lr0 = args.lr0 if args.lr0 is not None else td["lr0"]
    eff_lrf = args.lrf if args.lrf is not None else td["lrf"]
    eff_wd  = args.weight_decay if args.weight_decay is not None else td["weight_decay"]

    # ---- Ten run TU DONG theo config tune (de tien theo doi 32 run cua grid S1) ----
    tune_parts = [
        str(eff_opt),
        f"lr0-{_fmt(eff_lr0)}",
        f"lrf-{_fmt(eff_lrf)}",
        f"cos-{int(args.cos_lr)}",
        f"wd-{_fmt(eff_wd)}",
    ]
    if args.momentum is not None:
        tune_parts.append(f"mom-{_fmt(args.momentum)}")
    run_name = f"{name}__{'_'.join(tune_parts)}_s{args.seed}"

    # ---- Luu log console giong M1_s42.txt: tee stdout+stderr ra file trong run dir ----
    run_dir = Path(args.project) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.logfile) if args.logfile else run_dir / "train_log.txt"
    _log_fh = open(log_path, "w", encoding="utf-8", buffering=1)
    _log_fh.write("python " + " ".join(sys.argv) + "\n")        # dong dau = lenh da chay
    if sys.__stdout__ is not None:
        sys.__stdout__.write(f"[log] Console -> {log_path}\n")
    sys.stdout = _Tee(sys.__stdout__, _log_fh)
    sys.stderr = _Tee(sys.__stderr__, _log_fh)
    # Ultralytics tao StreamHandler luc import (bind stderr GOC) => phai tro lai tee,
    # neu khong banner + bang kien truc (LOGGER.info) se KHONG vao file.
    from ultralytics.utils import LOGGER
    for _h in LOGGER.handlers:
        if hasattr(_h, "setStream") and getattr(_h, "stream", None) in (sys.__stdout__, sys.__stderr__):
            _h.setStream(sys.stderr)

    overrides = dict(
        model=str(model_yaml),
        data=args.data,
        epochs=args.epochs or td["epochs"],
        imgsz=args.imgsz or td["imgsz"],
        batch=args.batch or td["batch"],
        patience=td["patience"],
        optimizer=eff_opt,
        lr0=eff_lr0,
        lrf=eff_lrf,
        weight_decay=eff_wd,
        warmup_epochs=args.warmup_epochs if args.warmup_epochs is not None else td["warmup_epochs"],
        cos_lr=args.cos_lr,          # MOI: bat/tat cosine LR
        # Augmentation
        mosaic=args.mosaic if args.mosaic is not None else td["mosaic"],
        flipud=args.flipud if args.flipud is not None else td["flipud"],
        fliplr=args.fliplr if args.fliplr is not None else td["fliplr"],
        degrees=args.degrees if args.degrees is not None else td["degrees"],
        hsv_h=args.hsv_h if args.hsv_h is not None else td["hsv_h"],
        hsv_s=args.hsv_s if args.hsv_s is not None else td["hsv_s"],
        hsv_v=args.hsv_v if args.hsv_v is not None else td["hsv_v"],
        # Runtime
        device=args.device if args.device is not None else td["device"],
        workers=args.workers if args.workers is not None else td["workers"],
        # Pretrained / output
        pretrained=args.weights or td["pretrained"],
        seed=args.seed,
        deterministic=True,
        project=args.project,
        name=run_name,
        exist_ok=True,
    )
    # momentum: chi dua vao overrides khi duoc chi dinh (con lai de Ultralytics tu chon theo optimizer)
    if args.momentum is not None:
        overrides["momentum"] = args.momentum

    trainer = SCTrainer(overrides=overrides)
    trainer.sc_modules = mods
    trainer.sc_specs = specs
    if 3 in mods:
        trainer.physics_aug = from_specs(specs)
    try:
        trainer.train()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        if not _log_fh.closed:
            _log_fh.close()


if __name__ == "__main__":
    main()