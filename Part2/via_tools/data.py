"""Lazy zarr access for the ViA Bag data.

The official `load_from_zarr` reads every array into RAM and the dataset casts
images to float32; with the 13.6 GB train zip that does not fit. Here only
`proprio`, `action` and `episode_ends` are read from the train zip (for the
min-max stats, cached to `stats_bag.npz`), and validation images are read
episode by episode as uint8.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import zarr

from . import DEFAULT_DATA_DIR, DEFAULT_OUT_DIR, add_official_to_path, out_path

add_official_to_path()
from utils import get_data_stats  # noqa: E402  (official)

TRAIN_FILE = "bag_train.zarr.zip"
VAL_FILE = "bag_val.zarr.zip"
STATS_FILE = "stats_bag.npz"


class ZarrZip:
    """Read-only handle on a `*.zarr.zip`; arrays are only read when sliced."""

    def __init__(self, path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(
                f"{self.path} not found. Download it from the official Drive folder "
                f"and put it in {self.path.parent}")
        self.store = zarr.ZipStore(str(self.path), mode="r")
        self.root = zarr.open_group(store=self.store, mode="r")

    def __getitem__(self, key):
        return self.root[key]

    def keys(self):
        return list(self.root.array_keys())

    def tree(self):
        return self.root.tree()

    @property
    def episode_ends(self):
        return np.asarray(self.root["episode_ends"][:], dtype=np.int64)

    def episode_bounds(self):
        """[(start, end)) per episode; `episode_ends` are exclusive ends, as in
        the official `create_sample_indices` and validation slice `[0:ends[0]]`."""
        ends = self.episode_ends
        starts = np.concatenate([[0], ends[:-1]])
        return list(zip(starts.tolist(), ends.tolist()))

    def close(self):
        self.store.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def compute_train_stats(data_dir=DEFAULT_DATA_DIR):
    """Official stats: `get_data_stats` on float32 proprio ('agent_pos') and action."""
    with ZarrZip(Path(data_dir) / TRAIN_FILE) as z:
        train = {
            "agent_pos": np.asarray(z["proprio"][:]).astype(np.float32),
            "action": np.asarray(z["action"][:]).astype(np.float32),
        }
    return {k: get_data_stats(v) for k, v in train.items()}


def load_stats(data_dir=DEFAULT_DATA_DIR, out_dir=DEFAULT_OUT_DIR, recompute=False):
    """Train-set stats, cached so the 13.6 GB zip is opened only once."""
    cache = out_path(out_dir, "data", STATS_FILE)
    if cache.exists() and not recompute:
        f = np.load(cache)
        return {k: {"min": f[f"{k}_min"], "max": f[f"{k}_max"]} for k in ("agent_pos", "action")}
    stats = compute_train_stats(data_dir)
    np.savez(cache, **{f"{k}_{s}": v[s] for k, v in stats.items() for s in ("min", "max")})
    return stats


class Episodes:
    """Episodes of one split, read lazily one at a time (images stay uint8).

    `main-image` is chunked along time (4501 frames per chunk in the train zip),
    so slicing one episode decompresses a whole time block; the last two blocks
    are cached (~0.7 GB each for the train zip) to avoid decoding them again
    for every episode inside them.
    """

    def __init__(self, data_dir=DEFAULT_DATA_DIR, split="val"):
        self.split = split
        self.z = ZarrZip(Path(data_dir) / (VAL_FILE if split == "val" else TRAIN_FILE))
        self.bounds = self.z.episode_bounds()
        self.proprio = np.asarray(self.z["proprio"][:]).astype(np.float32)
        self.action = np.asarray(self.z["action"][:]).astype(np.float32)
        self._block = self.z["main-image"].chunks[0]
        self._cache = {}

    def __len__(self):
        return len(self.bounds)

    def lengths(self):
        return [e - s for s, e in self.bounds]

    def _image_block(self, b):
        if b not in self._cache:
            if len(self._cache) >= 2:
                self._cache.pop(next(iter(self._cache)))
            n = self.z["main-image"].shape[0]
            self._cache[b] = np.asarray(self.z["main-image"][b * self._block:min(n, (b + 1) * self._block)])
        return self._cache[b]

    def images(self, k):
        """(T,H,W,3) uint8 head-camera frames of episode k."""
        s, e = self.bounds[k]
        out = []
        for b in range(s // self._block, (e - 1) // self._block + 1):
            blk = self._image_block(b)
            lo = max(s, b * self._block) - b * self._block
            hi = min(e, (b + 1) * self._block) - b * self._block
            out.append(blk[lo:hi])
        return np.concatenate(out) if len(out) > 1 else out[0]

    def __getitem__(self, k):
        s, e = self.bounds[k]
        return {
            "episode": k,
            "start": s,
            "main-image": self.images(k),
            "agent_pos": self.proprio[s:e],
            "action": self.action[s:e],
        }

    def __iter__(self):
        for k in range(len(self)):
            yield self[k]

    def close(self):
        self._cache.clear()
        self.z.close()


ValEpisodes = Episodes


def describe(path):
    """Keys, shapes, dtypes, chunks and episode stats, without reading images."""
    with ZarrZip(path) as z:
        info = {"file": str(path), "size_bytes": Path(path).stat().st_size, "arrays": {}}
        for k in z.keys():
            a = z[k]
            info["arrays"][k] = {"shape": list(a.shape), "dtype": str(a.dtype),
                                 "chunks": list(a.chunks)}
        lens = np.diff(np.concatenate([[0], z.episode_ends]))
        info["n_episodes"] = int(len(lens))
        info["episode_len"] = {"min": int(lens.min()), "max": int(lens.max()),
                               "mean": float(lens.mean()), "all": lens.tolist()}
        info["tree"] = str(z.tree())
    return info


def quat_sign_jumps(path):
    """Consecutive frames (same episode) whose quaternions have a negative dot
    product, i.e. the stored sign flipped (scipy `as_quat` does not fix w >= 0)."""
    from . import PARTS
    with ZarrZip(path) as z:
        bounds = z.episode_bounds()
        arrays = {k: np.asarray(z[k][:]) for k in ("proprio", "action")}
    same_ep = np.ones(len(arrays["action"]) - 1, bool)
    same_ep[[e - 1 for _, e in bounds[:-1]]] = False
    out = {}
    for k, a in arrays.items():
        for part, sl in PARTS.items():
            d = np.sum(a[1:, sl["quat"]] * a[:-1, sl["quat"]], axis=-1)
            w = a[:, sl["quat"]][:, -1]
            out[f"{k}_{part}"] = {"sign_jumps": int(np.sum((d < 0) & same_ep)),
                                  "w_min": float(w.min()), "w_max": float(w.max())}
    return out


def save_raw_video(val, out_dir, episode=0, seconds=10, fps=10, scale=2):
    import cv2
    import imageio.v2 as imageio

    imgs = val.images(episode)[: seconds * fps]
    p = out_path(out_dir, "videos", "raw_episode.mp4")
    with imageio.get_writer(p, fps=fps, codec="libx264", quality=8,
                            macro_block_size=1) as w:
        for im in imgs:
            w.append_data(cv2.resize(im, None, fx=scale, fy=scale,
                                     interpolation=cv2.INTER_NEAREST))
    return p


def peak_rss_gb():
    import psutil
    mi = psutil.Process().memory_info()
    return getattr(mi, "peak_wset", mi.rss) / 1e9


def main():
    ap = argparse.ArgumentParser(description="Inspect ViA Bag data, cache train stats.")
    ap.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--episode", type=int, default=0, help="val episode for the raw video")
    ap.add_argument("--recompute_stats", action="store_true")
    inspect(**vars(ap.parse_args()))


def inspect(data_dir=DEFAULT_DATA_DIR, out_dir=DEFAULT_OUT_DIR, episode=0, recompute_stats=False):
    """Print the data layout, cache the train stats and save the raw-episode video.
    Returns the summary dict (not written to disk)."""
    args = argparse.Namespace(**locals())
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    summary = {}
    for name in (TRAIN_FILE, VAL_FILE):
        info = describe(Path(args.data_dir) / name)
        info["quat_sign_jumps"] = quat_sign_jumps(Path(args.data_dir) / name)
        summary[name] = info
        print(f"\n== {name}  ({info['size_bytes'] / 1e9:.2f} GB)")
        for k, a in info["arrays"].items():
            print(f"  {k:14s} shape={a['shape']} dtype={a['dtype']} chunks={a['chunks']}")
        el = info["episode_len"]
        print(f"  episodes={info['n_episodes']} len min/mean/max = "
              f"{el['min']}/{el['mean']:.1f}/{el['max']} frames")
        print("  quaternion sign jumps between consecutive frames: " + ", ".join(
            f"{k} {v['sign_jumps']}" for k, v in info["quat_sign_jumps"].items()))

    t0 = time.time()
    stats = load_stats(args.data_dir, out, recompute=args.recompute_stats)
    print(f"\ntrain stats ({time.time() - t0:.1f} s, cached in {out_path(out, 'data', STATS_FILE)}):")
    for k, v in stats.items():
        print(f"  {k} min={np.round(v['min'], 3).tolist()}")
        print(f"  {k} max={np.round(v['max'], 3).tolist()}")

    summary["train_stats"] = {k: {s: v[s].tolist() for s in ("min", "max")} for k, v in stats.items()}

    val = ValEpisodes(args.data_dir)
    print("\nsaved", save_raw_video(val, out, args.episode))
    val.close()
    print(f"peak RAM: {peak_rss_gb():.2f} GB")
    return summary


if __name__ == "__main__":
    main()
