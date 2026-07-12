# collect_screen.py  (ROOT repo)
import csv, glob
from pathlib import Path

PROJECT = Path("runs/scyolo12")

def load_best(csv_path):
    with open(csv_path) as f:
        rows = [{k.strip(): v.strip() for k, v in r.items()} for r in csv.DictReader(f)]
    if not rows: return None
    fit = lambda r: 0.2*float(r["metrics/mAP50(B)"]) + 0.8*float(r["metrics/mAP50-95(B)"])
    b = max(rows, key=fit)
    return {"epoch": int(float(b["epoch"])),
            "P": float(b["metrics/precision(B)"]), "R": float(b["metrics/recall(B)"]),
            "mAP50": float(b["metrics/mAP50(B)"]), "mAP50_95": float(b["metrics/mAP50-95(B)"]),
            "fitness": fit(b)}

out = []
for res in sorted(glob.glob(str(PROJECT / "M34__*" / "results.csv"))):
    m = load_best(res)
    if m: m["run"] = Path(res).parent.name; out.append(m)

out.sort(key=lambda r: r["fitness"], reverse=True)
print(f"{'run':58s} {'mAP50':>7} {'mAP50-95':>9} {'P':>6} {'R':>6} {'fit':>7}")
for r in out:
    print(f"{r['run']:58s} {r['mAP50']:7.4f} {r['mAP50_95']:9.4f} {r['P']:6.3f} {r['R']:6.3f} {r['fitness']:7.4f}")
with open("screen_summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["run","epoch","P","R","mAP50","mAP50_95","fitness"])
    w.writeheader(); [w.writerow(r) for r in out]
print(f"\n{len(out)} run -> screen_summary.csv")