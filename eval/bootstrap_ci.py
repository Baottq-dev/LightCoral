# eval/bootstrap_ci.py
# Bootstrap CI muc anh cho hieu so metric giua 2 cau hinh (paired by image).
#
# Input: 2 mang per-image metric (vd AP@0.5 tinh san cho tung anh) cua
#        model A va model B, CUNG thu tu anh (paired).
# Output: diff trung binh, CI phan vi, p-value 2 phia (gia thuyet diff=0).

import argparse
import json

import numpy as np


def bootstrap_diff(a, b, n_boot=10000, ci=0.95, seed=0):
    """a, b: (N,) per-image metric, paired. Tra ve dict ket qua."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    assert a.shape == b.shape, "a va b phai cung so anh (paired)"
    n = len(a)
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))     # resample anh (co hoan lai)
    boot = d[idx].mean(axis=1)                       # (n_boot,)
    lo = np.percentile(boot, (1 - ci) / 2 * 100)
    hi = np.percentile(boot, (1 + ci) / 2 * 100)
    # p-value 2 phia: ti le boot vuot qua 0 ve phia nguoc dau (pivotal don gian)
    p = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
    return {
        "mean_diff": float(d.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "ci_level": ci,
        "p_value": float(min(p, 1.0)),
        "n_images": int(n),
        "n_boot": int(n_boot),
        "significant": bool(lo > 0 or hi < 0),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser("Bootstrap CI muc anh cho delta mAP")
    ap.add_argument("--a", required=True, help="JSON list per-image metric model A")
    ap.add_argument("--b", required=True, help="JSON list per-image metric model B")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--ci", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    a = json.loads(open(args.a).read())
    b = json.loads(open(args.b).read())
    res = bootstrap_diff(a, b, args.n_boot, args.ci, args.seed)
    print(json.dumps(res, indent=2, ensure_ascii=False))