# run_s1_sweep.py  (dat o ROOT repo sc-yolo12/)
import os
import subprocess
import sys
from pathlib import Path

DATA    = "cfg/coral_soft_yolo.yaml"
MODULES = "3,4"
SEED    = 42
EPOCHS  = 200
PROJECT = Path("runs/scyolo12")

def fmt(x):
    return f"{x:g}".replace("-", "m").replace(".", "p")

# lr0 LONG theo optimizer (KHONG cross bua)
OPTLR = [("SGD", 0.01), ("SGD", 0.005), ("AdamW", 0.002), ("AdamW", 0.001)]
LRFS  = [0.01, 0.001]
WDS   = [0.0005, 0.001, 0.005]  # bien tren mo rong: 517 anh train de overfit, can test wd manh hon
COSES = [False, True]

tag  = "".join(sorted(m.strip() for m in MODULES.split(",") if m.strip()))
name = f"M{tag}"

configs = [(o, l, lrf, wd, cos)
           for (o, l) in OPTLR for lrf in LRFS for wd in WDS for cos in COSES]

print(f"Tong {len(configs)} config\n")
for i, (opt, lr0, lrf, wd, cos) in enumerate(configs, 1):
    parts = [opt, f"lr0-{fmt(lr0)}", f"lrf-{fmt(lrf)}", f"cos-{int(cos)}", f"wd-{fmt(wd)}"]
    run_name = f"{name}__{'_'.join(parts)}_s{SEED}"
    if (PROJECT / run_name / "weights" / "best.pt").exists():
        print(f"[{i}/{len(configs)}] SKIP (da xong): {run_name}")
        continue
    cmd = [sys.executable, "train_tune.py",
           "--data", DATA, "--modules", MODULES, "--seed", str(SEED),
           "--epochs", str(EPOCHS),
           "--optimizer", opt, "--lr0", str(lr0), "--lrf", str(lrf),
           "--weight_decay", str(wd)]
    if cos:
        cmd.append("--cos_lr")
    print(f"[{i}/{len(configs)}] RUN: {run_name}")
    ret = subprocess.run(cmd)
    ok = (PROJECT / run_name / "weights" / "best.pt").exists()
    if not ok:
        print(f"  !! That bai THAT SU o {run_name} (rc={ret.returncode}); dung lai.")
        sys.exit(ret.returncode)
    if ret.returncode != 0:
        print(f"  (bo qua) rc={ret.returncode} nhung best.pt da co -> chay tiep")
print(f"\nHoan tat {len(configs)} run.")