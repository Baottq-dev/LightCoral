import csv, glob, re, statistics as st
from pathlib import Path

PROJECT = Path("runs/scyolo12")
OUT     = "confirm_summary.csv"
METRICS = ["fit", "mAP50", "mAP50_95", "P", "R"]

def best_row(p):
    rows = [{k.strip(): v.strip() for k, v in r.items()} for r in csv.DictReader(open(p))]
    if not rows: return None
    fit = lambda r: 0.2*float(r["metrics/mAP50(B)"]) + 0.8*float(r["metrics/mAP50-95(B)"])
    b = max(rows, key=fit)
    return {"mAP50": float(b["metrics/mAP50(B)"]), "mAP50_95": float(b["metrics/mAP50-95(B)"]),
            "P": float(b["metrics/precision(B)"]), "R": float(b["metrics/recall(B)"]),
            "fit": fit(b), "epoch": int(float(b["epoch"]))}

# gom theo config (bo hau to _s42/_s0/_s1), luu ca seed de in chi tiet
groups = {}
for res in glob.glob(str(PROJECT / "*" / "results.csv")):
    name = Path(res).parent.name
    mseed = re.search(r"_s(\d+)$", name)
    cfg = re.sub(r"_s\d+$", "", name)
    m = best_row(res)
    if m:
        m["seed"] = int(mseed.group(1)) if mseed else -1
        groups.setdefault(cfg, []).append(m)

out = []
for cfg, ms in groups.items():
    if len(ms) < 2: continue        # chi xet config >=2 seed (bo 32 run screen 1-seed)
    agg = {"cfg": cfg, "n": len(ms), "seeds": ",".join(str(m["seed"]) for m in sorted(ms, key=lambda x: x["seed"]))}
    for k in METRICS:
        v = [m[k] for m in ms]
        agg[f"{k}_mean"] = round(st.mean(v), 4)
        agg[f"{k}_std"]  = round(st.stdev(v) if len(v) > 1 else 0.0, 4)
    agg["_detail"] = ms
    out.append(agg)

out.sort(key=lambda a: a["fit_mean"], reverse=True)

# ---- in bang tong hop ----
hdr = f"{'cfg':52s} {'n':>2} {'seeds':>8}"
for k in METRICS: hdr += f" {k+'_mean':>13} {'std':>7}"
print(hdr)
for a in out:
    line = f"{a['cfg']:52s} {a['n']:2d} {a['seeds']:>8}"
    for k in METRICS: line += f" {a[f'{k}_mean']:13.4f} {a[f'{k}_std']:7.4f}"
    print(line)

# ---- in chi tiet tung seed (kiem tra do dao dong) ----
print("\n--- chi tiet tung seed ---")
for a in out:
    print(a["cfg"])
    for m in sorted(a["_detail"], key=lambda x: x["seed"]):
        print(f"   s{m['seed']:<3} fit={m['fit']:.4f}  mAP50={m['mAP50']:.4f}  "
              f"mAP50-95={m['mAP50_95']:.4f}  P={m['P']:.3f}  R={m['R']:.3f}  (best@ep{m['epoch']})")

# ---- ghi CSV ----
fields = ["cfg", "n", "seeds"] + [f"{k}_{s}" for k in METRICS for s in ("mean", "std")]
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    for a in out: w.writerow(a)
print(f"\n-> da ghi {len(out)} config vao {OUT}")