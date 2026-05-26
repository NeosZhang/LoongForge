"""Common CLI arguments shared across all trainers."""
import argparse
import sys

from omegaconf import OmegaConf


def parse_common_args():
    """Parse common training arguments shared by all trainers."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    p.add_argument("--phase", type=str, default=None, choices=["pretrain", "finetune"])
    p.add_argument("--ckpt_path", type=str, default=None, help="SFT checkpoint path")

    # ── Trainer Config ──
    p.add_argument("--max_train_steps", type=int, default=150000)
    p.add_argument("--save_steps", type=int, default=10000)
    p.add_argument("--gradient_clipping", type=float, default=1.0)
    p.add_argument("--gradient_accumulation_steps", type=int, default=2)
    p.add_argument("--loss_spike_threshold", type=float, default=100.0)
    p.add_argument("--logging_frequency", type=int, default=50)
    p.add_argument("--save_format", type=str, default="safetensors")
    p.add_argument("--save_training_state", action="store_true")
    p.add_argument("--is_resume", action="store_true")
    p.add_argument("--pretrained_checkpoint", type=str, default=None)
    p.add_argument("--freeze_modules", type=str, default="vlm_interface.model.language_model")
    # Learning rates
    p.add_argument("--learning_rate", type=float, default=2.5e-05, help="Base learning rate")
    p.add_argument("--lr_vlm", type=float, default=1.0e-05, help="VLM interface learning rate")
    p.add_argument("--lr_action_model", type=float, default=1.0e-04, help="Action model learning rate")
    # Optimizer
    p.add_argument("--optimizer", type=str, default="AdamW")
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--adam_beta1", type=float, default=0.9)
    p.add_argument("--adam_beta2", type=float, default=0.95)
    p.add_argument("--adam_eps", type=float, default=1.0e-08)
    # LR scheduler
    p.add_argument("--lr_scheduler_type", type=str, default="cosine_with_min_lr")
    p.add_argument("--warmup_steps", type=int, default=2000)
    p.add_argument("--min_lr", type=float, default=1.0e-06)
    # EMA
    p.add_argument("--ema_enabled", action="store_true")
    p.add_argument("--ema_decay", type=float, default=0.9999)

    # ── Running Config ──
    p.add_argument("--seed", type=int, default=3047)
    p.add_argument("--output_dir", type=str, default="outputs")
    p.add_argument("--run_id", type=str, default=None)
    p.add_argument("--wandb_project", type=str, default="loongforge-vla")
    p.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])

    # ── Dataset Config ──
    p.add_argument("--dataloader_module", type=str, default="lerobot_datasets")
    p.add_argument("--data_root_dir", type=str, default="/data/lerobot")
    p.add_argument("--dataset_path", type=str, default="/data/lerobot/libero_spatial")
    p.add_argument("--dataset_mix", type=str, default=None)
    p.add_argument("--robot_type", type=str, default="libero_franka")
    p.add_argument("--per_device_batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--action_horizon", type=int, default=10)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--normalization_mode", type=str, default="q99")

    args, _ = p.parse_known_args()
    return args


# CLI arg name -> OmegaConf dotpath mapping (common args only)
_COMMON_ARG_TO_CFG = {
    # trainer
    "max_train_steps": "trainer.max_train_steps",
    "save_steps": "trainer.save_steps",
    "gradient_clipping": "trainer.gradient_clipping",
    "gradient_accumulation_steps": "trainer.gradient_accumulation_steps",
    "loss_spike_threshold": "trainer.loss_spike_threshold",
    "logging_frequency": "trainer.logging_frequency",
    "save_format": "trainer.save_format",
    "save_training_state": "trainer.save_training_state",
    "is_resume": "trainer.is_resume",
    "pretrained_checkpoint": "trainer.pretrained_checkpoint",
    "freeze_modules": "trainer.freeze_modules",
    "learning_rate": "trainer.learning_rate.base",
    "lr_vlm": "trainer.learning_rate.vlm_interface",
    "lr_action_model": "trainer.learning_rate.action_model",
    "optimizer": "trainer.optimizer.name",
    "weight_decay": "trainer.optimizer.weight_decay",
    "adam_beta1": "trainer.optimizer.betas.0",
    "adam_beta2": "trainer.optimizer.betas.1",
    "adam_eps": "trainer.optimizer.eps",
    "lr_scheduler_type": "trainer.lr_scheduler_type",
    "warmup_steps": "trainer.warmup_steps",
    "min_lr": "trainer.scheduler_specific_kwargs.min_lr",
    "ema_enabled": "trainer.ema.enabled",
    "ema_decay": "trainer.ema.decay",
    # running
    "seed": "seed",
    "output_dir": "output_root_dir",
    "run_id": "run_id",
    "wandb_project": "wandb_project",
    "wandb_mode": "wandb_mode",
    # dataset
    "dataloader_module": "datasets.vla_data.dataloader_module",
    "data_root_dir": "datasets.vla_data.data_root_dir",
    "dataset_path": "datasets.vla_data.dataset_path",
    "dataset_mix": "datasets.vla_data.dataset_mix",
    "robot_type": "datasets.vla_data.robot_type",
    "per_device_batch_size": "datasets.vla_data.per_device_batch_size",
    "num_workers": "datasets.vla_data.num_workers",
    "action_horizon": "datasets.vla_data.action_horizon",
    "image_size": "datasets.vla_data.image_size",
    "normalization_mode": "datasets.vla_data.normalization_mode",
}


def merge_common_args_to_cfg(args, cfg):
    """Merge explicitly-passed common CLI args into OmegaConf cfg. CLI takes precedence."""
    raw_argv = sys.argv[1:]
    explicitly_set = set()
    for arg in raw_argv:
        if arg.startswith("--"):
            key = arg.lstrip("-").split("=")[0]
            explicitly_set.add(key)

    overrides = []
    for arg_name, dotpath in _COMMON_ARG_TO_CFG.items():
        if arg_name in explicitly_set:
            val = getattr(args, arg_name)
            overrides.append(f"{dotpath}={val}")

    if overrides:
        cli_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, cli_cfg)

    # Store meta fields directly on cfg
    for key in ("phase", "ckpt_path"):
        if key in explicitly_set or getattr(args, key, None) is not None:
            OmegaConf.update(cfg, key, getattr(args, key), merge=True)

    return cfg
