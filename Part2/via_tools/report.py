"""Result tables (pandas DataFrames) for the notebook; nothing is written to disk.

Two metrics: the released code's action MSE (raw units) and the normalised action MSE
of this work (see eval_offline.py), over all dims and per part.
"""

import numpy as np
import pandas as pd

from . import DEFAULT_OUT_DIR, PARTS, out_path

ROWS = [("action_mse", "action MSE, released code (raw units)"),
        ("nmse", "normalised action MSE, all dims"),
        ("nmse_head", "  head (7 dims)"),
        ("nmse_arm", "  arm pose (14 dims)"),
        ("nmse_gripper", "  gripper (2 dims)")]


def _pm(m, s, d=5):
    return f"{m:.{d}f} ± {s:.{d}f}"


def results_table(r):
    """r: dict returned by eval_offline.evaluate. mean ± std over seeds (and over episodes)."""
    multi = len(r["episodes"]) > 1
    rows = []
    for key, label in ROWS:
        o = r["overall"][key]
        row = {"metric": label, "mean ± std (3 seeds)": _pm(o["mean"], o["std_over_seeds"])}
        if multi:
            row["mean ± std (episodes)"] = _pm(o["mean"], o["std_over_episodes"])
        rows.append(row)
    return pd.DataFrame(rows)


def best_worst_table(out_dir=DEFAULT_OUT_DIR, name="bag_train", n=2):
    """Best and worst episodes by normalised MSE, from output/metrics/eval_<name>.csv."""
    df = pd.read_csv(out_path(out_dir, "metrics", f"eval_{name}.csv")).sort_values("nmse")
    sel = pd.concat([df.head(n).assign(case="best"), df.tail(n)[::-1].assign(case="worst")])
    cols = {key: label.strip() for key, label in ROWS}
    return sel[["case", "episode", "T", *cols]].rename(columns=cols).reset_index(drop=True)


SHARE_GROUPS = {"position": [0, 1, 2, 7, 8, 9, 15, 16, 17],
                "quaternion": [3, 4, 5, 6, 10, 11, 12, 13, 18, 19, 20, 21],
                "gripper": [14, 22]}


def per_dim_sq_error(out_dir, name, normalised=False, align_quat=False):
    """Mean squared error per action dim (23,), averaged over steps, seeds and episodes.

    normalised: errors in the policy's [-1, 1] space. align_quat: flip predicted
    quaternions into the GT hemisphere first (q and -q are the same rotation).
    """
    s = np.load(out_path(out_dir, "data", "stats_bag.npz"))
    half_range = (s["action_max"] - s["action_min"]) / 2
    per_ep = []
    for p in sorted(out_path(out_dir, "data").glob(f"pred_{name}_ep*.npz")):
        f = np.load(p)
        pred, gt = f["pred"].copy(), f["gt_action"][f["steps"]][None]
        if align_quat:
            for sl in PARTS.values():
                q = sl["quat"]
                pred[..., q] *= np.where((pred[..., q] * gt[..., q]).sum(-1, keepdims=True) < 0, -1, 1)
        d = pred - gt
        if normalised:
            d = d / half_range
        per_ep.append((d ** 2).mean(axis=(0, 1)))
    return np.mean(per_ep, axis=0)


def mse_variants_table(out_dir=DEFAULT_OUT_DIR):
    """Why the normalised MSE: share of position / quaternion / gripper dims in each variant."""
    variants = [("released code (raw units)", False, False),
                ("normalised [-1, 1]", True, False),
                ("normalised [-1, 1], quaternion sign aligned (this work)", True, True)]
    rows = []
    for name, split in (("bag", "val"), ("bag_train", "train")):
        if not list(out_path(out_dir, "data").glob(f"pred_{name}_ep*.npz")):
            continue
        for label, norm, align in variants:
            v = per_dim_sq_error(out_dir, name, norm, align)
            row = {"split": split, "action MSE": label, "value": f"{v.mean():.5f}"}
            for g, idx in SHARE_GROUPS.items():
                row[f"{g} share"] = f"{100 * v[idx].sum() / v.sum():.1f}%"
            rows.append(row)
    return pd.DataFrame(rows)
