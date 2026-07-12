# scripts/run_ablation.py  (dat o repo root/scripts/, ngang hang run_confirm.py)
# Pipeline ablation chinh: B0/M1/M2/M12/M3/M4/M34 x 3 seed, workers=0.
# Train qua subprocess train.py (HP dong bo voi run_benchmark.ps1 mac dinh,
# DE VE train_defaults trong module_specs.yaml vi do dung optimizer=auto/lr0=0.01).
# Eval in-process (conf=0.001, iou=0.7) tren tap test, gom mean+/-std (ddof=1) qua seed.
#
# Usage:
#   python scripts/run_ablation.py
#   python scripts/run_ablation.py --configs B0,M34 --seeds 0,1
import argparse
import csv
import os
import statistics as st
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TRAIN = REPO / "train.py"
assert TRAIN.exists(), f"Khong thay train.py tai {TRAIN}"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))  # de import duoc 'models.registry' cho eval in-process

DATA    = "cfg/coral_soft_yolo.yaml"
PROJECT = REPO / "runs" / "ablation"

CONFIGS = {
    "B0":  "",
    "M1":  "1",
    "M2":  "2",
    "M12": "1,2",
    "M3":  "3",
    "M4":  "4",
    "M34": "3,4",
}
SEEDS = [0, 1, 2]

# HP chuan (khop mac dinh run_benchmark.ps1 / 3_train_baseline.py), DE VE train_defaults
HP = dict(
    optimizer="SGD", lr0=0.001, lrf=0.01, weight_decay=0.0005, warmup_epochs=3,
    epochs=100, imgsz=640, batch=16, workers=0,
)

EVAL_KW = dict(split="test", imgsz=640, batch=16, device="0", conf=0.001, iou=0.7)


def run_name_for(mods_arg, seed):
    """Tai tao dung cong thuc run_name cua train.py:185."""
    mods = {int(x) for x in mods_arg.split(",") if x.strip()}
    tag = "".join(str(m) for m in sorted(mods)) or "0"
    name = f"M{tag}"
    data_tag = Path(DATA).stem
    return f"{name}_{data_tag}_s{seed}_ep{HP['epochs']}_lr0{HP['lr0']}_lrf{HP['lrf']}_config1"


def train_one(mods_arg, seed, env):
    rn = run_name_for(mods_arg, seed)
    best = PROJECT / rn / "weights" / "best.pt"
    if best.exists():
        print(f"[skip] {rn} (best.pt da co)")
        return rn, best

    cmd = [
        sys.executable, str(TRAIN),
        "--data", DATA, "--modules", mods_arg, "--seed", str(seed),
        "--optimizer", HP["optimizer"], "--lr0", str(HP["lr0"]), "--lrf", str(HP["lrf"]),
        "--weight_decay", str(HP["weight_decay"]), "--warmup_epochs", str(HP["warmup_epochs"]),
        "--epochs", str(HP["epochs"]), "--imgsz", str(HP["imgsz"]), "--batch", str(HP["batch"]),
        "--workers", str(HP["workers"]), "--project", str(PROJECT),
    ]
    print(f"[run] {rn}")
    ret = subprocess.run(cmd, cwd=str(REPO), env=env)
    if not best.exists():
        print(f"  !! that bai THAT SU {rn} rc={ret.returncode}")
        sys.exit(ret.returncode)
    if ret.returncode != 0:
        print(f"  (bo qua) rc={ret.returncode} nhung co best.pt -> tiep tuc")
    return rn, best


def eval_one(weights):
    from ultralytics import YOLO
    from models.registry import register_custom_modules

    register_custom_modules()  # PHAI goi truoc khi load checkpoint co custom module (SFDF/AMCF/PG-DAM)
    model = YOLO(str(weights))
    box = model.val(data=DATA, **EVAL_KW, plots=False, verbose=False).box
    return float(box.map50), float(box.map)


def mean_std(vals):
    m = st.mean(vals)
    s = st.stdev(vals) if len(vals) > 1 else 0.0  # sample std (ddof=1)
    return m, s


def main():
    ap = argparse.ArgumentParser("SC-YOLO12 ablation driver (B0/M1/M2/M12/M3/M4/M34 x seeds)")
    ap.add_argument("--configs", default=",".join(CONFIGS.keys()),
                     help="Danh sach config, phay ngan cach. Mac dinh: tat ca 7 config.")
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS),
                     help="Danh sach seed, phay ngan cach.")
    ap.add_argument("--epochs", type=int, default=None,
                     help="Ghi de HP['epochs'] (dung cho sanity check nhanh, vd --epochs 1).")
    args = ap.parse_args()

    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    for c in configs:
        assert c in CONFIGS, f"Config khong hop le: {c} (co: {list(CONFIGS)})"
    if args.epochs is not None:
        HP["epochs"] = args.epochs

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")

    PROJECT.mkdir(parents=True, exist_ok=True)

    # ── Train tat ca (name, mods) x seed ──────────────────────────────────────
    weights_map = {}  # (name, seed) -> best.pt path
    for name in configs:
        mods_arg = CONFIGS[name]
        for seed in seeds:
            rn, best = train_one(mods_arg, seed, env)
            weights_map[(name, seed)] = best

    # ── Eval in-process tren tat ca checkpoint ────────────────────────────────
    per_seed_rows = []  # (config, seed, mAP50, mAP50-95)
    for name in configs:
        for seed in seeds:
            best = weights_map[(name, seed)]
            print(f"\n==== EVAL {name} seed={seed}: {best} ====")
            map50, map50_95 = eval_one(best)
            print(f"  mAP50={map50:.4f}  mAP50-95={map50_95:.4f}")
            per_seed_rows.append((name, seed, map50, map50_95))

    # ── CSV 1: per-seed ────────────────────────────────────────────────────────
    p1 = PROJECT / "per_seed_test.csv"
    with p1.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["config", "seed", "mAP50", "mAP50_95"])
        for name, seed, map50, map50_95 in per_seed_rows:
            w.writerow([name, seed, f"{map50:.6f}", f"{map50_95:.6f}"])
    print(f"\nDa luu: {p1}")

    # ── Gom mean+/-std moi config, Delta vs B0 ────────────────────────────────
    by_config = {name: [] for name in configs}
    for name, seed, map50, map50_95 in per_seed_rows:
        by_config[name].append((map50, map50_95))

    agg = {}
    for name in configs:
        map50_vals = [m for m, _ in by_config[name]]
        map5095_vals = [m for _, m in by_config[name]]
        agg[name] = {
            "n": len(map50_vals),
            "mAP50_mean": mean_std(map50_vals)[0], "mAP50_std": mean_std(map50_vals)[1],
            "mAP50_95_mean": mean_std(map5095_vals)[0], "mAP50_95_std": mean_std(map5095_vals)[1],
        }

    b0_map50 = agg.get("B0", {}).get("mAP50_mean")
    b0_map5095 = agg.get("B0", {}).get("mAP50_95_mean")

    p2 = PROJECT / "ablation_agg_test.csv"
    with p2.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["config", "n", "mAP50_mean", "mAP50_std", "mAP50_95_mean", "mAP50_95_std",
                    "delta_mAP50_vs_B0", "delta_mAP50_95_vs_B0"])
        for name in configs:
            a = agg[name]
            d50 = f"{a['mAP50_mean'] - b0_map50:.6f}" if b0_map50 is not None else ""
            d5095 = f"{a['mAP50_95_mean'] - b0_map5095:.6f}" if b0_map5095 is not None else ""
            w.writerow([name, a["n"], f"{a['mAP50_mean']:.6f}", f"{a['mAP50_std']:.6f}",
                        f"{a['mAP50_95_mean']:.6f}", f"{a['mAP50_95_std']:.6f}", d50, d5095])
    print(f"Da luu: {p2}")

    print(f"\n{'Config':<8}{'n':>3}  {'mAP50':>18}  {'mAP50-95':>18}")
    for name in configs:
        a = agg[name]
        print(f"{name:<8}{a['n']:>3}  "
              f"{a['mAP50_mean']:.4f} +/- {a['mAP50_std']:.4f}   "
              f"{a['mAP50_95_mean']:.4f} +/- {a['mAP50_95_std']:.4f}")


if __name__ == "__main__":
    main()
