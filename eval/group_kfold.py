# eval/group_kfold.py
# Tao K fold theo GroupKFold (chong ro ri near-duplicate) va xuat data YAML/fold.
#
# Group-id lay tu (uu tien cao -> thap):
#   1) file groups.csv (cot: image, group) neu co
#   2) tien to ten file truoc dau '_' hoac so frame (vd IMG_0423 -> IMG)
# Day la diem MAU CHOT chong leakage: anh gan trung lap phai cung 1 fold.

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import yaml
from sklearn.model_selection import GroupKFold


def default_group_id(stem: str) -> str:
    """Heuristic group-id khi khong co groups.csv.
    'IMG_0423' -> 'IMG'; 'site3-frame0007' -> 'site3'. Tinh chinh theo dataset.
    """
    m = re.split(r"[_\-]", stem)
    return m[0] if m else stem


def load_groups(images, groups_csv=None):
    if groups_csv and Path(groups_csv).exists():
        mp = {}
        with open(groups_csv) as f:
            for row in csv.DictReader(f):
                mp[Path(row["image"]).stem] = row["group"]
        return [mp.get(Path(p).stem, default_group_id(Path(p).stem)) for p in images]
    return [default_group_id(Path(p).stem) for p in images]


def make_folds(image_dir, n_splits=5, groups_csv=None, pattern="*.jpg"):
    images = sorted(str(p) for p in Path(image_dir).rglob(pattern))
    assert images, f"Khong tim thay anh trong {image_dir}"
    groups = load_groups(images, groups_csv)
    n_groups = len(set(groups))
    assert n_groups >= n_splits, (
        f"So group ({n_groups}) < n_splits ({n_splits}); giam K hoac gop group."
    )
    gkf = GroupKFold(n_splits=n_splits)
    folds = []
    for tr, va in gkf.split(images, groups=groups):
        folds.append(([images[i] for i in tr], [images[i] for i in va]))
        # kiem chung: khong group nao xuat hien o ca 2 phia
        gtr = {groups[i] for i in tr}
        gva = {groups[i] for i in va}
        assert not (gtr & gva), "RO RI: group xuat hien o ca train va val!"
    return folds


def write_fold_yaml(folds, names, out_dir, base_data):
    """Ghi train/val list + data YAML cho moi fold (dung duong dan tuyet doi)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for k, (tr, va) in enumerate(folds):
        ftr = out_dir / f"fold{k}_train.txt"
        fva = out_dir / f"fold{k}_val.txt"
        ftr.write_text("\n".join(tr))
        fva.write_text("\n".join(va))
        data = dict(base_data)
        data.update(train=str(ftr), val=str(fva), nc=len(names), names=names)
        fy = out_dir / f"scoraldet_fold{k}.yaml"
        fy.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
        paths.append(str(fy))
    return paths


if __name__ == "__main__":
    ap = argparse.ArgumentParser("GroupKFold splitter cho SCoralDet")
    ap.add_argument("--images", required=True)
    ap.add_argument("--groups-csv", default=None)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", default="data/folds")
    ap.add_argument("--pattern", default="*.jpg")
    args = ap.parse_args()

    names = ["Euphyllia", "Favosites", "Platygyra", "Sarcophyton", "Sinularia", "Wavinghand"]
    folds = make_folds(args.images, args.k, args.groups_csv, args.pattern)
    paths = write_fold_yaml(folds, names, args.out, base_data={"path": "."})
    print("Da tao", len(paths), "fold:")
    for p in paths:
        print(" -", p)