"""Build `nets` exactly as the official `train_via.py` does and load a checkpoint.

`train_via.py` is a top-level training script (it loads the dataset, starts
wandb and trains on import), so it cannot be imported. The few lines that build
the networks and the DDIM scheduler are mirrored here with the official
classes and the config values; see NOTES.md.
"""
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from . import DEFAULT_CKPT_DIR, DEFAULT_CKPT_NAME, DEFAULT_CONFIG, add_official_to_path

add_official_to_path()
from models import ConditionalUnet1D  # noqa: E402  (official)


def load_config(path=DEFAULT_CONFIG):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_nets(cfg):
    """train_via.py: ConditionalUnet1D + DINOv2 ViT-S/14 in a ModuleDict."""
    noise_pred_net = ConditionalUnet1D(
        input_dim=cfg["action_dim"],
        global_cond_dim=cfg["obs_dim"] * cfg["obs_horizon"])
    head_dinov2_vits14 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    return nn.ModuleDict({
        "vision_encoder": head_dinov2_vits14,
        "noise_pred_net": noise_pred_net,
    })


def build_scheduler(cfg):
    """train_via.py: DDIMScheduler with the same arguments."""
    return DDIMScheduler(
        num_train_timesteps=cfg["num_train_DDIM_timesteps"],
        beta_start=.0001,
        beta_end=0.02,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        set_alpha_to_one=True,
        steps_offset=0,
        prediction_type="epsilon")


def read_state_dict(path):
    """Return (state_dict, info) and describe how the file is stored."""
    obj = torch.load(path, map_location="cpu", weights_only=True)
    info = {"file": str(path), "size_mb": Path(path).stat().st_size / 1e6}
    if isinstance(obj, dict) and all(isinstance(v, torch.Tensor) for v in obj.values()):
        info["format"] = "raw state_dict"
        sd = obj
    else:
        wrapper = next(k for k in ("state_dict", "model", "ema", "nets") if k in obj)
        info["format"] = f"wrapped dict, weights under '{wrapper}'"
        sd = obj[wrapper]
    info["has_module_prefix"] = any(k.startswith("module.") for k in sd)
    if info["has_module_prefix"]:
        sd = {k.removeprefix("module."): v for k, v in sd.items()}
    info["top_level_keys"] = sorted({k.split(".")[0] for k in sd})
    info["n_tensors"] = len(sd)
    info["n_params"] = int(sum(v.numel() for v in sd.values()))
    return sd, info


def load_policy(config=DEFAULT_CONFIG, ckpt=DEFAULT_CKPT_DIR / DEFAULT_CKPT_NAME,
                device="cuda"):
    """nets (eval mode, on device), noise scheduler, config dict, checkpoint info."""
    cfg = load_config(config)
    nets = build_nets(cfg)
    sd, info = read_state_dict(ckpt)
    nets.load_state_dict(sd, strict=True)
    nets.eval().to(device)
    return nets, build_scheduler(cfg), cfg, info


def main():
    ap = argparse.ArgumentParser(description="Build ViA nets and load a checkpoint (strict).")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--ckpt_dir", default=str(DEFAULT_CKPT_DIR))
    ap.add_argument("--ckpt", default=DEFAULT_CKPT_NAME)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    nets, _, cfg, info = load_policy(args.config, Path(args.ckpt_dir) / args.ckpt, args.device)
    for k, v in info.items():
        print(f"{k}: {v}")
    for name, m in nets.items():
        print(f"{name}: {sum(p.numel() for p in m.parameters()) / 1e6:.2f} M params")
    print("loaded with strict=True on", args.device)


if __name__ == "__main__":
    main()
