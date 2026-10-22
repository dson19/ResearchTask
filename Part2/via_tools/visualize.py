"""Result videos from `eval_offline.py` outputs.

Video (10 fps, 1280x720): head-camera frame | 3D head / end-effector
trajectories (ground truth solid, prediction dashed, current time as markers)
| running normalised action MSE of the head, the arms and the grippers.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import DEFAULT_DATA_DIR, DEFAULT_OUT_DIR, PARTS, out_path  # noqa: E402
from .data import Episodes  # noqa: E402

FPS = 10
COLORS = {"head": "#d62728", "right": "#1f77b4", "left": "#2ca02c"}
NMSE = {"nmse_head": ("head", "#d62728"), "nmse_arm": ("arm pose", "k"),
        "nmse_gripper": ("gripper", "#ff7f0e")}


def load_pred(path):
    f = np.load(path)
    return {k: f[k] for k in f.files}


def _plot_nmse(ax, t, err, alpha=1.0):
    """head / arm on the left axis, gripper (much larger) on a twin axis."""
    lines = {}
    for k in ("nmse_head", "nmse_arm"):
        label, c = NMSE[k]
        lines[k], = ax.plot(t, err[k], color=c, lw=1.4, alpha=alpha, label=label)
    ax2 = ax.twinx()
    label, c = NMSE["nmse_gripper"]
    lines["nmse_gripper"], = ax2.plot(t, err["nmse_gripper"], color=c, lw=1.2, alpha=alpha,
                                      ls="--", label=f"{label} (right axis)")
    ax.set_ylabel("normalised MSE")
    ax2.set_ylabel("gripper normalised MSE")
    ax.grid(alpha=0.3)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
    return lines, ax2


class EpisodeRenderer:
    def __init__(self, pred, images, sample="mean", split="val"):
        self.p = pred
        self.split = split
        self.images = images
        self.steps = pred["steps"]
        self.T = int(pred["T"])
        traj = pred["pred"] if sample == "mean" else pred["pred"][[int(sample)]]
        self.pred = traj.mean(axis=0)                       # (N, 23), positions only used
        self.gt = pred["gt_action"][self.steps]
        self.err = {k: pred[f"err_{k}"].mean(axis=0) for k in NMSE}
        self.mse = float(pred["err_action_mse"].mean())
        self.nmse = float(pred["err_nmse"].mean())
        self._build()

    def _build(self):
        fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
        gs = fig.add_gridspec(2, 2, height_ratios=[2.5, 1], width_ratios=[1, 1.25],
                              left=0.06, right=0.94, top=0.90, bottom=0.08,
                              hspace=0.12, wspace=0.02)
        self.fig = fig
        ax_img = fig.add_subplot(gs[0, 0])
        self.im = ax_img.imshow(self.images[self.steps[0]])
        ax_img.set_axis_off()
        ax_img.set_title("head camera (observation)", fontsize=10)

        ax3 = fig.add_subplot(gs[0, 1], projection="3d")
        self.markers = {}
        for part, sl in PARTS.items():
            g, q = self.gt[:, sl["pos"]], self.pred[:, sl["pos"]]
            c = COLORS[part]
            ax3.plot(*g.T, color=c, lw=1.6, label=f"{part} GT")
            ax3.plot(*q.T, color=c, lw=1.2, ls="--", label=f"{part} pred")
            self.markers[part] = (ax3.plot(*g[:1].T, "o", color=c, ms=7)[0],
                                  ax3.plot(*q[:1].T, "x", color=c, ms=8, mew=2)[0])
        ax3.set_xlabel("x [m]", fontsize=8)
        ax3.set_ylabel("y [m]", fontsize=8)
        ax3.set_zlabel("z [m]", fontsize=8)
        ax3.tick_params(labelsize=7)
        ax3.legend(fontsize=7, ncol=1, loc="upper left", bbox_to_anchor=(-0.08, 1.0))
        ax3.set_title("trajectories: solid = human demo, dashed = policy\n"
                      "(o / x = current GT / predicted pose)", fontsize=10)

        ax_e = fig.add_subplot(gs[1, :])
        t = self.steps / FPS
        _, ax2_faint = _plot_nmse(ax_e, t, self.err, alpha=0.15)           # full curves, faint
        self.lines, ax2 = _plot_nmse(ax_e, t[:1], {k: v[:1] for k, v in self.err.items()})
        for a in (ax2, ax2_faint):
            a.set_ylim(0, 1.05 * max(self.err["nmse_gripper"].max(), 1e-12))
        ax2_faint.set_yticks([])
        ax2_faint.set_ylabel("")
        self.cursor = ax_e.axvline(0, color="0.5", lw=1)
        ax_e.set_xlim(0, t[-1])
        ax_e.set_ylim(0, 1.05 * max(self.err["nmse_head"].max(), self.err["nmse_arm"].max()))
        ax_e.set_xlabel("time [s]")
        self.title = fig.suptitle("", fontsize=12)

    def render(self, i):
        s = self.steps[i]
        self.im.set_data(self.images[s])
        for part, sl in PARTS.items():
            mg, mp = self.markers[part]
            g, q = self.gt[i, sl["pos"]], self.pred[i, sl["pos"]]
            mg.set_data_3d([g[0]], [g[1]], [g[2]])
            mp.set_data_3d([q[0]], [q[1]], [q[2]])
        t = self.steps[: i + 1] / FPS
        for k, line in self.lines.items():
            line.set_data(t, self.err[k][: i + 1])
        self.cursor.set_xdata([s / FPS, s / FPS])
        self.title.set_text(
            f"ViA Bag {self.split} episode {int(self.p['episode'])}  |  action MSE {self.mse:.4f}, "
            f"normalised {self.nmse:.4f}  |  t = {s / FPS:5.1f} s  (frame {s}/{self.T})")
        self.fig.canvas.draw()
        return np.asarray(self.fig.canvas.buffer_rgba())[..., :3].copy()

    def video(self, path, fps=FPS):
        import imageio.v2 as imageio
        with imageio.get_writer(path, fps=fps, codec="libx264", quality=7,
                                macro_block_size=1, pixelformat="yuv420p") as w:
            for i in range(len(self.steps)):
                w.append_data(self.render(i))
        return Path(path)

    def snapshot(self, i, path):
        import imageio.v2 as imageio
        imageio.imwrite(path, self.render(i))
        return Path(path)

    def close(self):
        plt.close(self.fig)


def main():
    ap = argparse.ArgumentParser(description="ViA result videos.")
    ap.add_argument("--name", default="bag", help="eval name: renders the val episode, or the "
                    "best and worst episodes by normalised MSE for a multi-episode split")
    ap.add_argument("--pred", nargs="*", help="extra pred_<name>_ep<k>.npz files to render")
    ap.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--sample", default="mean", help="'mean' over seeds or a sample index")
    ap.add_argument("--split", default=None, choices=["val", "train"],
                    help="split of the predictions (default: train if the name ends in _train)")
    make_videos(**vars(ap.parse_args()))


def make_videos(name="bag", pred=None, data_dir=DEFAULT_DATA_DIR, out_dir=DEFAULT_OUT_DIR,
                sample="mean", split=None):
    """Videos of the episodes of eval `name` (from output/metrics/eval_<name>.csv): the only
    episode, or the best and worst by normalised MSE. Returns the list of files written."""
    import pandas as pd
    out = Path(out_dir)
    split = split or ("train" if name.endswith("_train") else "val")
    episodes = Episodes(data_dir, split)
    rank = pd.read_csv(out_path(out, "metrics", f"eval_{name}.csv")).sort_values("nmse")["episode"].tolist()
    jobs = [(name, rank[0])] if len(rank) == 1 else [(f"{name}_best", rank[0]), (f"{name}_worst", rank[-1])]
    paths = [(tag, out_path(out, "data", f"pred_{name}_ep{k}.npz")) for tag, k in jobs]
    paths += [("ep", Path(p_)) for p_ in pred or []]
    written = []
    for tag, pred_path in paths:
        p = load_pred(pred_path)
        r = EpisodeRenderer(p, episodes.images(int(p["episode"])), sample, split)
        v = r.video(out_path(out, "videos", f"video_{tag}_ep{int(p['episode'])}.mp4"))
        r.close()
        written.append(v)
        print(f"saved {v} ({v.stat().st_size / 1e6:.1f} MB)")
    episodes.close()
    return written


if __name__ == "__main__":
    main()
