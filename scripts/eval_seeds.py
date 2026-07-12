# eval_seeds.py  (dat o ROOT repo sc-yolo12/, ngang hang test.py)
# Danh gia MOT config qua NHIEU seed tren tap test -> bao cao mean+/-std (tong + per-class).
# Dua tren test.py, chi khac: lap qua seeds, gom trung binh, tu dat ten thu muc ket qua.
# Vi du:
#   python eval_seeds.py --run M34__AdamW_lr0-0p001_lrf-0p01_cos-0_wd-0p0005 \
#       --seeds 0,1,42 --data cfg/coral_soft_yolo.yaml
#   -> runs/scyolo12/M34__..._test_agg/{per_seed_test.csv, per_class_agg_test.csv, summary_test.json}

import argparse
import csv
import json
import statistics as st
import sys
from pathlib import Path

_here = Path(__file__).resolve()
for _p in [_here.parent, *_here.parents]:
    if (_p / "engine").is_dir():
        sys.path.insert(0, str(_p))
        break

from ultralytics import YOLO

from engine.build_model import ROOT
from models.registry import register_custom_modules



def _mean_std(vals):
    m = st.mean(vals)
    s = st.stdev(vals) if len(vals) > 1 else 0.0   # sample std (ddof=1) cho paper
    return m, s


def eval_one(weights, args):
    """Chay model.val tren 1 checkpoint -> (overall dict, per_class dict)."""
    model = YOLO(weights)
    val_kwargs = dict(
        data=args.data, split=args.split, imgsz=args.imgsz, batch=args.batch,
        conf=args.conf, iou=args.iou, plots=False, verbose=False,
    )
    if args.device is not None:
        val_kwargs["device"] = args.device
    box = model.val(**val_kwargs).box
    names = model.names if isinstance(model.names, dict) else {i: n for i, n in enumerate(model.names)}
    overall = {"mAP50-95": float(box.map), "mAP50": float(box.map50),
               "mAP75": float(box.map75), "P": float(box.mp), "R": float(box.mr)}
    per_class = {}
    for i, ci in enumerate(box.ap_class_index):
        p, r, ap50, ap = box.class_result(i)
        per_class[names.get(int(ci), str(ci))] = {"P": float(p), "R": float(r),
                                                   "mAP50": float(ap50), "mAP50-95": float(ap)}
    return overall, per_class


def main():
    ap = argparse.ArgumentParser("SC-YOLO12 multi-seed evaluator (test mean+/-std)")
    ap.add_argument("--run", required=True,
                    help="ten config KHONG kem _s<seed>, vd M34__AdamW_lr0-0p001_lrf-0p01_cos-0_wd-0p0005")
    ap.add_argument("--seeds", default="0,1,42", help="danh sach seed, vd 0,1,42")
    ap.add_argument("--data", required=True, help="data YAML (vd cfg/coral_soft_yolo.yaml)")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None, help="vd 0 | cpu (mac dinh: tu chon)")
    ap.add_argument("--conf", type=float, default=0.001, help="chuan do mAP (KHONG phai inference)")
    ap.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    ap.add_argument("--project", default=str(ROOT / "runs" / "scyolo12"))
    ap.add_argument("--name", default=None, help="ten thu muc ket qua; mac dinh <run>_<split>_agg")
    ap.add_argument("--weights-name", default="best.pt", help="ten file weight trong weights/ (mac dinh best.pt)")
    args = ap.parse_args()

    register_custom_modules()       # PHAI goi truoc khi load model co custom module (SFDF/PG-DAM/FGA2)

    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
    project = Path(args.project)
    run_name = args.name or f"{args.run}_{args.split}_agg"
    save_dir = project / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # ---- chay tung seed ----
    # LUU Y: "seed" o day CHI chon checkpoint train ({run}_s{seed}). model.val()
    # voi conf=0.001, khong augmentation -> hoan toan deterministic, khong can set_seed.
    # Bien dong giua cac seed la do checkpoint TRAIN khac nhau, KHONG phai eval.
    per_seed = []          # list[(seed, overall, per_class)]
    for seed in seeds:
        wpath = project / f"{args.run}_s{seed}" / "weights" / args.weights_name
        assert wpath.exists(), f"Khong thay checkpoint: {wpath}"
        print(f"\n==== EVAL seed {seed}: {wpath} ====")
        overall, per_class = eval_one(str(wpath), args)
        print(f"  mAP50={overall['mAP50']:.4f}  mAP50-95={overall['mAP50-95']:.4f}  "
              f"P={overall['P']:.3f}  R={overall['R']:.3f}")
        per_seed.append((seed, overall, per_class))

    n = len(per_seed)
    metric_keys = ["mAP50-95", "mAP50", "mAP75", "P", "R"]

    # ---- gom overall mean+/-std ----
    agg_overall = {k: _mean_std([o[k] for _, o, _ in per_seed]) for k in metric_keys}
    print(f"\n================ TEST '{args.split}' — {n} seed {seeds} ================")
    for k in metric_keys:
        m, s = agg_overall[k]
        print(f"{k:9s}: {m:.4f} +/- {s:.4f}")

    # ---- gom per-class mean+/-std ----
    class_names = list(per_seed[0][2].keys())
    agg_class = {}
    for cname in class_names:
        agg_class[cname] = {k: _mean_std([pc[cname][k] for _, _, pc in per_seed])
                            for k in ["mAP50-95", "mAP50", "P", "R"]}
    print("\nPer-class mAP50-95 (mean +/- std):")
    for cname in class_names:
        m, s = agg_class[cname]["mAP50-95"]
        print(f"  {cname:<22} {m:.3f} +/- {s:.3f}")

    # ---- CSV 1: tung seed (overall) + 2 dong mean/std ----
    p1 = save_dir / f"per_seed_{args.split}.csv"
    with p1.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["seed"] + metric_keys)
        for seed, o, _ in per_seed:
            w.writerow([seed] + [f"{o[k]:.6f}" for k in metric_keys])
        w.writerow(["mean"] + [f"{agg_overall[k][0]:.6f}" for k in metric_keys])
        w.writerow(["std"]  + [f"{agg_overall[k][1]:.6f}" for k in metric_keys])
    print(f"\nDa luu: {p1}")

    # ---- CSV 2: per-class mean+/-std ----
    p2 = save_dir / f"per_class_agg_{args.split}.csv"
    with p2.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class", "mAP50-95_mean", "mAP50-95_std", "mAP50_mean", "mAP50_std",
                    "P_mean", "P_std", "R_mean", "R_std"])
        for cname in class_names:
            a = agg_class[cname]
            w.writerow([cname,
                        f"{a['mAP50-95'][0]:.6f}", f"{a['mAP50-95'][1]:.6f}",
                        f"{a['mAP50'][0]:.6f}",    f"{a['mAP50'][1]:.6f}",
                        f"{a['P'][0]:.6f}",        f"{a['P'][1]:.6f}",
                        f"{a['R'][0]:.6f}",        f"{a['R'][1]:.6f}"])
    print(f"Da luu: {p2}")

    # ---- JSON tom tat (cho script tong hop ablation doc lai) ----
    summary = {
        "run": args.run, "split": args.split, "seeds": seeds, "n": n,
        "overall_mean": {k: agg_overall[k][0] for k in metric_keys},
        "overall_std":  {k: agg_overall[k][1] for k in metric_keys},
        "per_class_map50_95_mean": {c: agg_class[c]["mAP50-95"][0] for c in class_names},
        "per_class_map50_95_std":  {c: agg_class[c]["mAP50-95"][1] for c in class_names},
        "per_seed": [{"seed": s, **o} for s, o, _ in per_seed],
    }
    p3 = save_dir / f"summary_{args.split}.json"
    p3.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Da luu: {p3}")
    print(f"\nTat ca ket qua o: {save_dir}")


if __name__ == "__main__":
    main()