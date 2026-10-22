"""Open-loop evaluation of the official ViA Bag checkpoint: action MSE + normalised MSE.

Receding horizon as in the official validation (train_via.py): observe frame
8c, predict 16 actions, keep the first 8, advance 8, with ground-truth
observations from the dataset and `int(T/8) - 2` chunks per episode. Because
the observations are ground truth, chunks do not depend on earlier
predictions, so all chunks (and sampling seeds) of an episode run as one batch;
this is the same computation as the sequential official loop.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import (DEFAULT_CKPT_DIR, DEFAULT_CKPT_NAME, DEFAULT_CONFIG, DEFAULT_DATA_DIR,
               DEFAULT_OUT_DIR, PARTS, add_official_to_path, out_path)
from .data import Episodes, load_stats, peak_rss_gb
from .model import load_policy

add_official_to_path()
from utils import normalize_data, normalize_image_imagenet, unnormalize_data  # noqa: E402

# action_mse: the released code's val_epoch_action_error (raw units, all 23 dims).
# nmse*: normalised action MSE (this work): error in the policy's [-1, 1] space, predicted
# quaternions sign-aligned to the human ones; over all dims and per part.
METRICS = ["action_mse", "nmse", "nmse_head", "nmse_arm", "nmse_gripper"]
DIMS = {"head": list(range(0, 7)),                          # head position + quaternion
        "arm": list(range(7, 14)) + list(range(15, 22)),    # both arms, position + quaternion
        "gripper": [14, 22]}                                # right, left gripper command


# ---------------------------------------------------------------- inference
def preprocess_images(images):
    """uint8 (B,H,W,3) -> float32 tensor-ready (B,3,H,W): /255 then ImageNet normalisation,
    as in the official training dataset and robot rollout script. (The official validation
    loop in train_via.py skips the /255; see NOTES.md.)"""
    x = images.astype(np.float32) / 255.0
    x = normalize_image_imagenet(x)
    return np.moveaxis(x, -1, 1)


@torch.no_grad()
def predict_chunks(nets, scheduler, cfg, stats, images, agent_pos, noise, device):
    """images (C,k,H,W,3) uint8, agent_pos (C,k,23) with k = obs_horizon (oldest first),
    noise (S,C,pred_horizon,23).

    Returns unnormalised actions (S, C, action_horizon, 23)."""
    S, C = noise.shape[:2]
    k = images.shape[1]
    img = torch.from_numpy(preprocess_images(images.reshape(C * k, *images.shape[2:]))).to(device)
    feats = nets["vision_encoder"].forward_features(img)["x_norm_clstoken"]
    nagent = torch.from_numpy(normalize_data(agent_pos.reshape(C * k, -1),
                                             stats=stats["agent_pos"])).to(device, dtype=torch.float32)
    obs_cond = torch.cat([feats, nagent], dim=-1).reshape(C, -1)   # (C, k*obs_dim)
    obs_cond = obs_cond.unsqueeze(0).expand(S, -1, -1).reshape(S * C, -1)

    naction = noise.reshape(S * C, *noise.shape[2:]).to(device)
    scheduler.set_timesteps(cfg["num_test_DDIM_timesteps"])
    for k in scheduler.timesteps:
        noise_pred = nets["noise_pred_net"](sample=naction, timestep=k, global_cond=obs_cond)
        naction = scheduler.step(model_output=noise_pred, timestep=k, sample=naction).prev_sample

    naction = naction.cpu().numpy().reshape(S, C, *noise.shape[2:])
    action_pred = unnormalize_data(naction, stats=stats["action"])
    start = cfg["obs_horizon"] - 1                 # official: first executed action
    return action_pred[:, :, start:start + cfg["action_horizon"]]


def episode_noise(cfg, n_chunks, n_samples, seed, episode):
    """Fixed Gaussian noise per (episode, sample); generated on CPU for reproducibility."""
    out = []
    for s in range(n_samples):
        g = torch.Generator().manual_seed(seed + 1000 * s + 100003 * episode)
        out.append(torch.randn((n_chunks, cfg["pred_horizon"], cfg["action_dim"]), generator=g))
    return torch.stack(out)


def run_episode(nets, scheduler, cfg, stats, ep, n_samples, seed, device,
                max_batch=128):
    T = len(ep["action"])
    H = cfg["action_horizon"]
    n_chunks = int(T / H) - 2                     # official: range(int(len/8) - 2)
    starts = np.arange(n_chunks) * H
    # observation window of the official obs deque: frames s-k+1 .. s, frame 0 repeated
    k = cfg["obs_horizon"]
    frames = np.maximum(starts[:, None] - np.arange(k - 1, -1, -1)[None], 0)   # (C, k)
    noise = episode_noise(cfg, n_chunks, n_samples, seed, ep["episode"])
    preds = []
    per_batch = max(1, max_batch // n_samples)
    for b in range(0, n_chunks, per_batch):
        idx = frames[b:b + per_batch]
        preds.append(predict_chunks(nets, scheduler, cfg, stats, ep["main-image"][idx],
                                    ep["agent_pos"][idx], noise[:, b:b + per_batch],
                                    device))
    pred = np.concatenate(preds, axis=1).reshape(n_samples, n_chunks * H, -1)
    steps = np.arange(n_chunks * H)
    return {"pred": pred.astype(np.float32), "steps": steps, "chunk_starts": starts, "T": T}


# ---------------------------------------------------------------- metrics
def half_range(stats):
    """Half the training range of each action dim: dividing an error by it gives the error
    in the policy's [-1, 1] space (the min/offset of the normalisation cancels)."""
    return (stats["action"]["max"] - stats["action"]["min"]) / 2


def normalised_sq_error(pred, gt, half_rng):
    """pred (S,N,23), gt (N,23) -> squared error (S,N,23) in [-1, 1] units, with the
    predicted quaternions flipped to the GT hemisphere (q and -q are the same rotation)."""
    pred = pred.copy()
    gt = gt[None]
    for sl in PARTS.values():
        q = sl["quat"]
        pred[..., q] *= np.where((pred[..., q] * gt[..., q]).sum(-1, keepdims=True) < 0, -1, 1)
    return ((pred - gt) / half_rng) ** 2


def step_errors(pred, gt, half_rng):
    """pred (S,N,23), gt (N,23) -> {metric: per-step error (S,N)}."""
    err = {"action_mse": np.mean((pred - gt[None]) ** 2, axis=-1)}   # = F.mse_loss per step
    nsq = normalised_sq_error(pred, gt, half_rng)
    err["nmse"] = nsq.mean(-1)
    for part, idx in DIMS.items():
        err[f"nmse_{part}"] = nsq[..., idx].mean(-1)
    return err


# ---------------------------------------------------------------- main
def aggregate(rows_seed):
    """rows_seed: list over episodes of {metric: (S,)}. Returns overall stats."""
    overall = {}
    for m in METRICS:
        a = np.stack([r[m] for r in rows_seed])          # (E, S)
        seed_means = a.mean(axis=0)                      # mean over episodes, per seed
        overall[m] = {"mean": float(a.mean()),
                      "std_over_seeds": float(seed_means.std()),
                      "std_over_episodes": float(a.mean(axis=1).std()),
                      "per_seed": seed_means.tolist()}
    return overall


def main():
    ap = argparse.ArgumentParser(description="Open-loop eval of ViA on Bag val episodes.")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--ckpt", default=DEFAULT_CKPT_NAME, help="file name inside --ckpt_dir")
    ap.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--ckpt_dir", default=str(DEFAULT_CKPT_DIR))
    ap.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--split", default="val", choices=["val", "train"],
                    help="val = held-out (official); train = episodes seen in training (supplementary)")
    ap.add_argument("--n_samples", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, nargs="*", help="subset of val episodes")
    ap.add_argument("--name", default="bag")
    evaluate(**vars(ap.parse_args()))


def evaluate(config=DEFAULT_CONFIG, ckpt=DEFAULT_CKPT_NAME, data_dir=DEFAULT_DATA_DIR,
             ckpt_dir=DEFAULT_CKPT_DIR, out_dir=DEFAULT_OUT_DIR, split="val", n_samples=3,
             seed=0, episodes=None, name="bag"):
    """Run the evaluation, write eval_<name>.csv (per episode) and pred_<name>_ep<k>.npz;
    return the summary dict (overall metrics, rankings), not written to disk."""
    args = argparse.Namespace(**locals())
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = False
    t0 = time.time()
    nets, scheduler, cfg, ckpt_info = load_policy(args.config, Path(args.ckpt_dir) / args.ckpt, device)
    stats = load_stats(args.data_dir, out)
    val = Episodes(args.data_dir, args.split)
    episodes = args.episodes if args.episodes else list(range(len(val)))
    print(f"model + stats ready in {time.time() - t0:.1f} s; {len(episodes)} episodes on {device}")

    episodes = [int(k) for k in episodes]
    result = {"config": str(args.config), "ckpt": ckpt_info,
              "n_samples": args.n_samples, "seed": args.seed, "split": args.split,
              "episodes": episodes}

    rows, rows_seed = [], []
    half_rng = half_range(stats)
    t1 = time.time()
    for k in episodes:
        ep = val[k]
        o = run_episode(nets, scheduler, cfg, stats, ep, args.n_samples, args.seed, device)
        gt = ep["action"][o["steps"]]
        err = step_errors(o["pred"], gt, half_rng)
        per_seed = {m: err[m].mean(axis=1) for m in METRICS}
        rows_seed.append(per_seed)
        row = {"episode": k, "T": o["T"], "n_eval_steps": len(o["steps"])}
        for m in METRICS:
            row[m] = float(per_seed[m].mean())
            row[m + "_std_seeds"] = float(per_seed[m].std())
        rows.append(row)
        np.savez_compressed(
            out_path(out, "data", f"pred_{args.name}_ep{k}.npz"), episode=k, T=o["T"], steps=o["steps"],
            chunk_starts=o["chunk_starts"], pred=o["pred"], gt_action=ep["action"],
            agent_pos=ep["agent_pos"], seeds=np.arange(args.n_samples) + args.seed,
            **{f"err_{m}": err[m] for m in METRICS})
        if len(episodes) <= 10 or k % 25 == 0 or k == episodes[-1]:
            print(f"ep {k:3d} T={o['T']:4d} action_mse={row['action_mse']:.5f} "
                  f"nmse={row['nmse']:.5f} (head {row['nmse_head']:.5f}, arm {row['nmse_arm']:.5f}, "
                  f"gripper {row['nmse_gripper']:.5f})")
    val.close()

    df = pd.DataFrame(rows)
    csv_path = out_path(out, "metrics", f"eval_{args.name}.csv")
    df.to_csv(csv_path, index=False, float_format="%.6f")
    result["overall"] = aggregate(rows_seed)
    result["ranking_by_action_mse"] = df.sort_values("action_mse")["episode"].tolist()
    result["ranking_by_nmse"] = df.sort_values("nmse")["episode"].tolist()
    result["runtime_s"] = time.time() - t1
    result["peak_ram_gb"] = peak_rss_gb()
    o = result["overall"]
    print(f"\nall {len(episodes)} episodes ({result['runtime_s']:.0f} s): "
          f"action_mse {o['action_mse']['mean']:.5f} (seeds ±{o['action_mse']['std_over_seeds']:.5f}, "
          f"episodes ±{o['action_mse']['std_over_episodes']:.5f}); "
          f"nmse {o['nmse']['mean']:.5f} (seeds ±{o['nmse']['std_over_seeds']:.5f})")
    print("saved", csv_path)
    return result


if __name__ == "__main__":
    main()
