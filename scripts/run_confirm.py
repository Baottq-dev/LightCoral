import os, subprocess, sys
from pathlib import Path

REPO    = Path("F:/LightCoral-YOLO")
TRAIN   = REPO / "scripts" / "train_tune.py"
assert TRAIN.exists(), f"Khong thay train_tune.py tai {TRAIN}"
PROJECT = REPO / "runs" / "scyolo12"
DATA    = "cfg/coral_soft_yolo.yaml"
NAME, EPOCHS, SEEDS = "M34", 200, [0, 1]

# de train_tune.py import duoc 'augment', 'engine', 'models', ... o repo root
env = os.environ.copy()
env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")

# (optimizer, lr0, lrf, cos_lr, weight_decay)
CONFIGS = [
    ("AdamW", 0.001, 0.01,  True,  0.0005),
    ("AdamW", 0.001, 0.01,  False, 0.0005),
    ("AdamW", 0.001, 0.001, False, 0.0005),
    ("SGD",   0.005, 0.001, False, 0.0005),
    ("SGD",   0.01,  0.001, False, 0.001),
]

def fmt(x): return f"{x:g}".replace("-", "m").replace(".", "p")
def run_name(opt, lr0, lrf, cos, wd, seed):
    return f"{NAME}__{opt}_lr0-{fmt(lr0)}_lrf-{fmt(lrf)}_cos-{int(cos)}_wd-{fmt(wd)}_s{seed}"

for seed in SEEDS:
    for opt, lr0, lrf, cos, wd in CONFIGS:
        rn = run_name(opt, lr0, lrf, cos, wd, seed)
        if (PROJECT / rn / "weights" / "best.pt").exists():
            print(f"[skip] {rn}"); continue
        cmd = [sys.executable, str(TRAIN),
               "--data", DATA, "--modules", "3,4", "--seed", str(seed),
               "--epochs", str(EPOCHS),
               "--optimizer", opt, "--lr0", str(lr0), "--lrf", str(lrf),
               "--weight_decay", str(wd)]        # <-- --weight_decay, KHONG con --name/--wd
        if cos: cmd.append("--cos_lr")
        print("[run]", rn)
        ret = subprocess.run(cmd, cwd=str(REPO), env=env)
        ok = (PROJECT / rn / "weights" / "best.pt").exists()
        if not ok:
            print(f"  !! that bai THAT SU {rn} rc={ret.returncode}"); sys.exit(ret.returncode)
        if ret.returncode != 0:
            print(f"  (bo qua) rc={ret.returncode} nhung co best.pt -> tiep tuc")
print("DONE confirm sweep")