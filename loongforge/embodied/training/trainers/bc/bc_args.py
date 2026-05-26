"""CLI args specific to BC (Behavior Cloning) trainer."""
import argparse
import sys

from omegaconf import OmegaConf


def parse_bc_args():
    """Parse BC-specific training arguments."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--suite", type=str, default=None,
                   choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"])

    # ── Eval ──
    p.add_argument("--eval_interval", type=int, default=10,
                   help="Run deterministic eval every N iterations (0 = disable)")
    p.add_argument("--eval_n_episodes", type=int, default=10,
                   help="Number of episodes for deterministic eval")

    # ── Misc ──
    p.add_argument("--save_interval", type=int, default=50)
    p.add_argument("--save_video_interval", type=int, default=10)
    p.add_argument("--log_interval", type=int, default=1)

    args, _ = p.parse_known_args()
    return args


def merge_bc_args_to_cfg(args, cfg):
    """Merge explicitly-passed BC-specific CLI args into OmegaConf cfg."""
    raw_argv = sys.argv[1:]
    explicitly_set = set()
    for arg in raw_argv:
        if arg.startswith("--"):
            key = arg.lstrip("-").split("=")[0]
            explicitly_set.add(key)

    # Store BC-specific fields directly on cfg for trainer access
    for key in ("suite", "eval_interval", "eval_n_episodes",
                "save_interval", "save_video_interval", "log_interval"):
        if key in explicitly_set or getattr(args, key, None) is not None:
            OmegaConf.update(cfg, key, getattr(args, key), merge=True)

    return cfg