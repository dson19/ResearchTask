"""Thin wrappers around the official Vision in Action code (Bag task, offline eval)."""
import sys
from pathlib import Path

PART2_DIR = Path(__file__).resolve().parents[1]
OFFICIAL_DIR = PART2_DIR / "external" / "vision-in-action" / "via_diffusion_policy"
DEFAULT_DATA_DIR = OFFICIAL_DIR / "dataset"
DEFAULT_CKPT_DIR = OFFICIAL_DIR / "ckpts"
DEFAULT_OUT_DIR = PART2_DIR / "output"
DEFAULT_CONFIG = OFFICIAL_DIR / "config" / "train_via_bag.yaml"
DEFAULT_CKPT_NAME = "via_bag_head_rgb_dinov2_cls_ema_500.pth"

# Every generated file goes to <out_dir>/<sub-folder by kind>/<name>.
OUT_KINDS = {
    "data": "data",                # *.npz: stats_bag.npz + pred_*_ep*.npz (not committed)
    "metrics": "metrics",          # eval_*.csv: per-episode metrics
    "videos": "videos",            # *.mp4
}


def out_path(out_dir, kind, name=None):
    """<out_dir>/<kind>/[name], creating the folder."""
    d = Path(out_dir) / OUT_KINDS[kind]
    d.mkdir(parents=True, exist_ok=True)
    return d / name if name else d

# Layout of the 23-d proprio / action vector (see NOTES.md for the sources).
# neck (head) = xyz + quat(x,y,z,w); each arm = xyz + quat(x,y,z,w) + gripper.
PARTS = {
    "head": {"pos": slice(0, 3), "quat": slice(3, 7), "gripper": None},
    "right": {"pos": slice(7, 10), "quat": slice(10, 14), "gripper": 14},
    "left": {"pos": slice(15, 18), "quat": slice(18, 22), "gripper": 22},
}


def add_official_to_path():
    """Make the official `models.py` / `utils.py` importable."""
    p = str(OFFICIAL_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)
