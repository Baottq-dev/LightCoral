# eval/corrected_ttest.py
# Corrected resampled t-test (Nadeau & Bengio 2003).
# Dung khi so sanh 2 cau hinh tren K fold (x nhieu seed) cua CUNG mot CV.
#
# Hieu chinh: variance thuong bi danh gia thap vi cac tap train/test
# chong lan. He so hieu chinh = (1/n + n_test/n_train).

import argparse
import json

import numpy as np
from scipy import stats


def corrected_ttest(diffs, n_train, n_test):
    """diffs: (n,) hieu metric tung lan resample (A - B), vd mAP_A - mAP_B/fold.
    n_train, n_test: so mau train/test moi lan (de tinh he so hieu chinh).
    Tra ve dict: mean, t, p_value (2 phia), df.
    """
    d = np.asarray(diffs, float)
    n = len(d)
    assert n >= 2, "Can it nhat 2 lan do (fold x seed)"
    mean = d.mean()
    var = d.var(ddof=1)
    # he so hieu chinh Nadeau-Bengio
    corr = (1.0 / n) + (n_test / float(n_train))
    denom = np.sqrt(corr * var)
    if denom == 0:
        return {"mean_diff": float(mean), "t_stat": float("inf"),
                "p_value": 0.0, "df": n - 1, "significant": True}
    t = mean / denom
    df = n - 1
    p = 2 * stats.t.sf(abs(t), df)
    return {
        "mean_diff": float(mean),
        "t_stat": float(t),
        "p_value": float(p),
        "df": int(df),
        "correction": float(corr),
        "significant": bool(p < 0.05),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser("Corrected resampled t-test")
    ap.add_argument("--a", required=True, help="JSON list metric model A theo fold/seed")
    ap.add_argument("--b", required=True, help="JSON list metric model B (cung thu tu)")
    ap.add_argument("--n-train", type=int, required=True)
    ap.add_argument("--n-test", type=int, required=True)
    args = ap.parse_args()

    a = np.array(json.loads(open(args.a).read()), float)
    b = np.array(json.loads(open(args.b).read()), float)
    assert a.shape == b.shape, "A va B phai cung so lan do"
    res = corrected_ttest(a - b, args.n_train, args.n_test)
    print(json.dumps(res, indent=2, ensure_ascii=False))