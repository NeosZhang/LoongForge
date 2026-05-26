"""
train.py - unified train entrypoint

Usage:
    # single card
    python train.py --config configs/paligemma_pi05.yaml

    # multi cards (accelerate launch)
    accelerate launch --config_file configs/deepspeed/accelerate_zero2.yaml \
        train.py --config configs/paligemma_pi05.yaml

    # CLI overrides
    python train.py --config configs/paligemma_pi05.yaml \
        trainer.max_train_steps=50000 trainer.learning_rate.base=1e-4
"""

import logging
import math
import os
import sys
from datetime import datetime
from pathlib import Path

# Suppress verbose DeepSpeed logs
logging.getLogger("deepspeed").setLevel(logging.WARNING)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from typing import Dict
from omegaconf import OmegaConf
from transformers import get_scheduler

from dataloader import build_dataloader
from model.compose.builder import build_framework
from training.trainers import build_trainer
from training.trainer_utils.trainer_tools import (
    build_param_lr_groups,
    normalize_dotlist_args,
    is_main_process,
    TrainerUtils,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Accelerator initialization and configuration
# ═══════════════════════════════════════════════════════════════

def _build_accelerator(gradient_accumulation_steps: int = 1):
    """
    Create Accelerator with correct gradient_accumulation_steps from config.
    """
    from accelerate import Accelerator, DeepSpeedPlugin, DistributedDataParallelKwargs

    if os.environ.get("USE_DDP") == "1":
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        acc = Accelerator(
            gradient_accumulation_steps=gradient_accumulation_steps,
            kwargs_handlers=[ddp_kwargs],
        )
    else:
        deepspeed_plugin = DeepSpeedPlugin()
        acc = Accelerator(
            gradient_accumulation_steps=gradient_accumulation_steps,
            deepspeed_plugin=deepspeed_plugin,
        )

    acc.print(acc.state)
    return acc


# ═══════════════════════════════════════════════════════════════
# Directory setup
# ═══════════════════════════════════════════════════════════════

def setup_directories(cfg) -> Path:
    """create output directory and save config"""
    output_root = cfg.get("output_root_dir", "outputs")
    run_id = cfg.get("run_id", cfg.framework.get("name", "train"))
    cfg.output_dir = os.path.join(output_root, run_id)
    output_dir = Path(cfg.output_dir)

    if is_main_process():
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir / "checkpoints", exist_ok=True)

        # Setup file logging
        log_dir = output_dir / "logs"
        os.makedirs(log_dir, exist_ok=True)
        log_file = log_dir / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(formatter)
        logging.getLogger().addHandler(file_handler)
        logger.info(f"Training logs will be saved to: {log_file}")

    return output_dir


def prepare_data(cfg, accelerator, output_dir) -> Dict[str, DataLoader]:
    """prepare training data"""
    # VLA data loader
    dataset_mix = getattr(cfg.datasets.vla_data, 'dataset_mix', 'N/A')
    logger.info(f"Creating VLA Dataset with Mixture `{dataset_mix}`")
    dataloader = build_dataloader(cfg=cfg, dataloader_module=cfg.datasets.vla_data.dataloader_module)

    accelerator.dataloader_config.dispatch_batches = False
    if dist.is_initialized():
        dist.barrier()

    return {cfg.datasets.vla_data.type : dataloader}

# ═══════════════════════════════════════════════════════════════
# LR Scheduler initialization
# ═══════════════════════════════════════════════════════════════

def _build_lambda_linear_scheduler(optimizer, cycle_lengths, warm_up_steps, f_start, f_max, f_min):
    """
    Multi-cycle LambdaWarmUpCosine scheduler matching the original cosmos_policy.

    Each cycle i has:
      - warm_up_steps[i] steps: linear ramp from f_start[i] -> f_max[i]
      - remaining steps: cosine decay from f_max[i] -> f_min[i]
    The last cycle repeats indefinitely at f_min[-1].
    """

    def lr_lambda(step: int) -> float:
        cumulative = 0
        for i, cycle_len in enumerate(cycle_lengths):
            cycle_start = cumulative
            if i == len(cycle_lengths) - 1:
                local_step = step - cycle_start
            else:
                if step < cumulative + cycle_len:
                    local_step = step - cycle_start
                else:
                    cumulative += cycle_len
                    continue

            wu = warm_up_steps[i]
            fs, fm, fn = f_start[i], f_max[i], f_min[i]

            if wu > 0 and local_step < wu:
                return fs + (fm - fs) * local_step / wu
            else:
                decay_steps = cycle_len - wu
                if decay_steps <= 0:
                    return fn
                decay_step = local_step - wu
                if i == len(cycle_lengths) - 1:
                    progress = min(decay_step / max(decay_steps, 1), 1.0)
                else:
                    progress = decay_step / decay_steps
                return fn + 0.5 * (fm - fn) * (1.0 + math.cos(math.pi * progress))

        return f_min[-1]

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def setup_optimizer_and_scheduler(model, cfg):
    """
    set optimizer and scheduler
    """
    # Per-module learning rate groups
    param_groups = build_param_lr_groups(model=model, cfg=cfg)
    base_lr = cfg.trainer.learning_rate.get("base", 1e-4)

    # Optimizer
    optimizer_cfg = cfg.trainer.get("optimizer", {})
    betas = tuple(optimizer_cfg.get("betas", [0.9, 0.95]))
    weight_decay = optimizer_cfg.get("weight_decay", 0.01)
    eps = optimizer_cfg.get("eps", 1e-8)

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=base_lr,
        betas=betas,
        weight_decay=weight_decay,
        eps=eps,
    )

    # Print optimizer group info
    if is_main_process():
        for group in optimizer.param_groups:
            logger.info(
                f"LR Group {group.get('name', '?')}: "
                f"lr={group['lr']}, num_params={len(group['params'])}"
            )

    # LR Scheduler
    scheduler_type = cfg.trainer.get("scheduler_type", None)
    if scheduler_type == "lambda_linear":
        cycle_lengths = list(OmegaConf.to_container(cfg.trainer.cycle_lengths, resolve=True))
        warm_up_steps = list(OmegaConf.to_container(cfg.trainer.warm_up_steps, resolve=True))
        f_start = list(OmegaConf.to_container(cfg.trainer.f_start, resolve=True))
        f_max = list(OmegaConf.to_container(cfg.trainer.f_max, resolve=True))
        f_min = list(OmegaConf.to_container(cfg.trainer.f_min, resolve=True))
        lr_scheduler = _build_lambda_linear_scheduler(
            optimizer, cycle_lengths, warm_up_steps, f_start, f_max, f_min
        )
    else:
        lr_scheduler_type = cfg.trainer.get("lr_scheduler_type", "cosine_with_min_lr")
        num_warmup_steps = cfg.trainer.get("warmup_steps", cfg.trainer.get("num_warmup_steps", 2000))
        max_train_steps = cfg.trainer.max_train_steps
        scheduler_kwargs = cfg.trainer.get("scheduler_specific_kwargs", {})
        if scheduler_kwargs:
            scheduler_kwargs = OmegaConf.to_container(scheduler_kwargs, resolve=True)
        else:
            scheduler_kwargs = {}

        lr_scheduler = get_scheduler(
            name=lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=max_train_steps,
            scheduler_specific_kwargs=scheduler_kwargs,
        )

    return optimizer, lr_scheduler


# ═══════════════════════════════════════════════════════════════
# configuration initialization
# ═══════════════════════════════════════════════════════════════

def load_config(argv):
    """load config from command line"""
    config_path = None
    extra_args = []

    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--config" and i + 1 < len(argv):
            config_path = argv[i + 1]
            i += 2
        elif arg.startswith("--config="):
            config_path = arg.split("=", 1)[1]
            i += 1
        else:
            extra_args.append(arg)
            i += 1

    if config_path is None:
        config_path = "configs/paligemma_pi05.yaml"

    cfg = OmegaConf.load(config_path)

    # CLI overrides (supports both --key value and key=value formats)
    normalized = normalize_dotlist_args(extra_args)
    if normalized:
        cli_cfg = OmegaConf.from_dotlist(normalized)
        cfg = OmegaConf.merge(cfg, cli_cfg)

    return cfg, config_path


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    """main function"""
    # load configuration
    cfg, config_path = load_config(sys.argv[1:])

    # Parse and merge common CLI arguments (shared across all trainers)
    from training.arguments import parse_common_args, merge_common_args_to_cfg
    common_args = parse_common_args()
    cfg = merge_common_args_to_cfg(common_args, cfg)

    logger.info(f"Config loaded: {config_path}")
    logger.info(f"Framework: {cfg.framework.get('name', 'unnamed')}")
    logger.info(f"Compose: {OmegaConf.to_container(cfg.framework.compose)}")

    # build Accelerator
    gradient_accumulation_steps = cfg.trainer.get("gradient_accumulation_steps", 1)
    accelerator = _build_accelerator(gradient_accumulation_steps)

    # create output directory
    output_dir = setup_directories(cfg)

    # build model（four layers assembly）
    logger.info("Building framework...")
    model = build_framework(cfg)
    logger.info(f"Model built: {model.__class__.__name__}")
    logger.info(f"  Architecture: {model.architecture.__class__.__name__}")
    logger.info(f"  Condition:    {model.condition.__class__.__name__}")
    logger.info(f"  Action:   {model.action.__class__.__name__}")

    # prepare data
    dataloaders = prepare_data(cfg=cfg, accelerator=accelerator, output_dir=output_dir)

    # setup optimizer + scheduler
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model, cfg)

    # build Trainer
    trainer = build_trainer(cfg, model, accelerator, optimizer, lr_scheduler, dataloaders)
    logger.info(f"Trainer: {trainer.__class__.__name__}")

    # prepare training
    trainer.prepare_training()

    # start training
    logger.info("Starting training...")
    trainer.train()


if __name__ == "__main__":
    main()
