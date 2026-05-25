"""
Layer 1 - BaseTrainer: Training paradigm base class

Template Method pattern: defines the fixed skeleton of the training loop, subclasses
implement different training paradigms (BC, Co-Training, RL, Auxiliary Learning) by overriding hook methods.

Provides general training infrastructure:
  - Checkpoint save/load/resume
  - EMA (Exponential Moving Average)
  - Loss spike protection + NaN gradient cleanup
  - W&B / local metrics logging
  - Progress bar (tqdm)
  - Distributed training awareness
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import copy
import gc
import json
import logging
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class BaseTrainer(ABC):
    """
    Training paradigm abstract base class — Template Method pattern.

    train() method defines the fixed training loop skeleton:
        prepare_training → on_train_begin → [fetch → train_step → on_step_end → maybe_save]* → on_train_end

    Subclasses must implement:
        - setup_dataloaders(): Configure data loaders
        - train_step(): Single step training logic (forward + backward + optimizer step)

    Subclasses may optionally override:
        - on_train_begin(): Freeze modules, initialize EMA
        - on_step_end(): EMA update, STDP update, logging
        - on_epoch_end(): Replay buffer refresh
        - on_train_end(): Final save
    """

    def __init__(self, cfg, model, accelerator, optimizer, lr_scheduler):
        """
        Args:
            cfg: OmegaConf training configuration
            model: ModelFramework or BaseArchitecture instance
            accelerator: HuggingFace Accelerator instance
            optimizer: torch optimizer
            lr_scheduler: learning rate scheduler
        """
        self.cfg = cfg
        self.model = model
        self.accelerator = accelerator
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.completed_steps = 0
        self.current_epoch = 0
        self.max_steps = cfg.trainer.max_train_steps
        self.gradient_clipping = cfg.trainer.get("gradient_clipping", 1.0)
        self.loss_spike_threshold = cfg.trainer.get("loss_spike_threshold", 100.0)

        # Logging
        self.logging_frequency = cfg.trainer.get("logging_frequency", 10)

        # EMA
        ema_cfg = getattr(cfg.trainer, "ema", None)
        self.use_ema = ema_cfg is not None and getattr(ema_cfg, "enabled", False)
        self.ema_decay = getattr(ema_cfg, "decay", 0.99) if ema_cfg else 0.99
        self.ema_model = None

        # LoRA
        from training.trainer_utils.peft import is_lora_enabled
        self.use_lora = is_lora_enabled(cfg)

        # Checkpoint
        self.checkpoint_dir = None

        # Dataloaders
        self._dataloaders: Dict[str, DataLoader] = {}
        self._data_iterators: Dict[str, Any] = {}

    # ═══════════════════════════════════════════════════════════════
    # Training preparation — distributed initialization, checkpoint loading, EMA initialization
    # ═══════════════════════════════════════════════════════════════

    def prepare_training(self):
        """
        Pre-training preparation (called before train()):
          1. Set random seed
          2. Load pretrained checkpoint / resume
          3. Adjust lr_scheduler (resume scenario)
          4. Freeze modules
          5. Print trainable parameters
          6. Distributed prepare (DeepSpeed/DDP wrapping)
          7. Initialize W&B
          8. Initialize EMA
        """
        from accelerate.utils import set_seed
        from training.trainer_utils.trainer_tools import TrainerUtils

        rank = dist.get_rank() if dist.is_initialized() else 0
        seed = getattr(self.cfg, "seed", 3047) + rank
        set_seed(seed)

        # Setup checkpoint directory
        output_dir = getattr(self.cfg, "output_dir", "outputs/default")
        self.checkpoint_dir = os.path.join(output_dir, "checkpoints")
        if self.accelerator.is_main_process:
            os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Load pretrained / resume
        self._init_checkpointing()

        # Adjust lr_scheduler for resume
        self._adjust_lr_scheduler_for_resume()

        # Freeze modules
        freeze_modules = self.cfg.trainer.get("freeze_modules", "")
        if freeze_modules:
            self.model = TrainerUtils.freeze_backbones(self.model, freeze_modules=freeze_modules)

        # Print trainable parameters
        TrainerUtils.print_trainable_parameters(self.model)

        # init dataloaders
        self._dataloaders = self.setup_dataloaders()
        self._create_data_iterators()

        # Distributed prepare
        self.model, self.optimizer, *prepared_dls = TrainerUtils.setup_distributed_training(
            self.accelerator,
            self.model,
            self.optimizer,
            *self._dataloaders.values(),
        )
        # Reassign prepared dataloaders
        for key, dl in zip(self._dataloaders.keys(), prepared_dls):
            self._dataloaders[key] = dl

        # W&B
        self._init_wandb()

        # EMA (after distributed setup)
        if self.use_ema and self.accelerator.is_main_process:
            logger.info(f"Initializing EMA with decay={self.ema_decay}")
            unwrapped = self.accelerator.unwrap_model(self.model)
            self.ema_model = copy.deepcopy(unwrapped).cpu()
            self.ema_model.eval()
            for p in self.ema_model.parameters():
                p.requires_grad_(False)

    # ═══════════════════════════════════════════════════════════════
    # Template Method — Training loop skeleton (not overridable)
    # ═══════════════════════════════════════════════════════════════

    def train(self):
        """
        Fixed training loop skeleton. Subclasses customize behavior by overriding hook methods.
        """
        self.on_train_begin()

        self._log_training_config()

        # Progress bar
        try:
            from tqdm import tqdm
            progress_bar = tqdm(
                range(self.max_steps),
                initial=self.completed_steps,
                disable=not self.accelerator.is_local_main_process,
            )
        except ImportError:
            progress_bar = None

        logger.info(f"Starting training for {self.max_steps} steps")

        # Main loop
        while self.completed_steps < self.max_steps:
            t_data_start = time.perf_counter()
            batches = self.fetch_batches()
            t_data_end = time.perf_counter()

            t_model_start = time.perf_counter()
            metrics = self.train_step(batches)
            t_model_end = time.perf_counter()

            self.completed_steps += 1

            # Timing metrics
            metrics["data_time"] = t_data_end - t_data_start
            metrics["model_time"] = t_model_end - t_model_start

            self.on_step_end(metrics)
            self._log_metrics(metrics)
            self._maybe_eval_and_save()

            # Progress bar
            if progress_bar is not None:
                progress_bar.update(1)
                postfix = {
                    "loss": f"{metrics.get('total_loss', metrics.get('action_loss', 0)):.4f}",
                    "lr": f"{metrics.get('lr', 0):.2e}",
                }
                progress_bar.set_postfix(postfix)

        # End
        self.on_train_end()
        self._finalize_training()
        logger.info("Training completed.")

    # ═══════════════════════════════════════════════════════════════
    # Abstract methods — Subclasses must implement
    # ═══════════════════════════════════════════════════════════════

    @abstractmethod
    def setup_dataloaders(self) -> Dict[str, DataLoader]:
        """Create and return a named DataLoader dictionary."""
        ...

    @abstractmethod
    def train_step(self, batches: Dict[str, Any]) -> Dict[str, float]:
        """Single training step: forward -> loss -> backward -> optimizer step."""
        ...

    # ═══════════════════════════════════════════════════════════════
    # Optional hooks — Subclasses override as needed
    # ═══════════════════════════════════════════════════════════════

    def on_train_begin(self):
        """Initialization before training starts (freeze modules, print param counts, etc.)."""
        pass

    def on_step_end(self, metrics: Dict[str, float]):
        """Called after each training step (EMA update, STDP update, metric logging)."""
        self._ema_update()

    def on_epoch_end(self, epoch: int):
        """Called at epoch end (replay buffer refresh, sampler update)."""
        pass

    def on_train_end(self):
        """Called at training end (final checkpoint save)."""
        pass

    # ═══════════════════════════════════════════════════════════════
    # Batch fetching
    # ═══════════════════════════════════════════════════════════════

    def fetch_batches(self) -> Dict[str, Any]:
        """Fetch next batch from each dataloader."""
        batches = {}
        for name, iterator in self._data_iterators.items():
            try:
                batches[name] = next(iterator)
            except StopIteration:
                self.current_epoch += 1
                self.on_epoch_end(self.current_epoch)
                # Reset with epoch-aware sampler
                if hasattr(self._dataloaders[name], "sampler") and callable(
                    getattr(self._dataloaders[name].sampler, "set_epoch", None)
                ):
                    self._dataloaders[name].sampler.set_epoch(self.current_epoch)
                self._data_iterators[name] = iter(self._dataloaders[name])
                batches[name] = next(self._data_iterators[name])
        return batches

    def _create_data_iterators(self):
        """Initialize iterators for all dataloaders."""
        for name, dl in self._dataloaders.items():
            if dl is not None:
                self._data_iterators[name] = iter(dl)

    # ═══════════════════════════════════════════════════════════════
    # Loss spike protection + NaN gradient cleanup
    # ═══════════════════════════════════════════════════════════════

    def _check_loss_spike(self, loss: torch.Tensor) -> torch.Tensor:
        """
        Loss spike protection.

        In distributed training, backward cannot be skipped (otherwise NCCL all_reduce will deadlock),
        so the loss is zeroed to produce zero gradients from backward.
        """
        loss_val = loss.detach().item()
        if torch.isnan(loss) or torch.isinf(loss) or loss_val > self.loss_spike_threshold:
            logger.warning(
                f"[step {self.completed_steps}] Loss spike: {loss_val:.4f}, zeroing loss"
            )
            return loss * 0.0  # keeps graph alive for backward
        return loss

    def _clean_nan_gradients(self):
        """Replace all NaN/Inf gradients with 0."""
        for param in self.model.parameters():
            if param.grad is not None:
                torch.nan_to_num(
                    param.grad, nan=0.0, posinf=0.0, neginf=0.0, out=param.grad
                )

    # ═══════════════════════════════════════════════════════════════
    # EMA (Exponential Moving Average)
    # ═══════════════════════════════════════════════════════════════

    def _ema_update(self):
        """EMA model parameter update."""
        if not self.use_ema or self.ema_model is None:
            return
        if not self.accelerator.is_main_process:
            return

        unwrapped = self.accelerator.unwrap_model(self.model)
        with torch.no_grad():
            for ema_p, model_p in zip(
                self.ema_model.parameters(), unwrapped.parameters()
            ):
                ema_p.data.mul_(self.ema_decay).add_(
                    model_p.data.cpu(), alpha=1 - self.ema_decay
                )

    # ═══════════════════════════════════════════════════════════════
    # Checkpoint save / load / resume
    # ═══════════════════════════════════════════════════════════════

    def _init_checkpointing(self):
        """Initialize checkpoint directory, handle checkpoint loading."""
        from training.trainer_utils.trainer_tools import TrainerUtils

        pretrained_checkpoint = self.cfg.trainer.get("pretrained_checkpoint", None)
        is_resume = self.cfg.trainer.get("is_resume", False)

        if is_resume:
            resume_path, resumed_steps = TrainerUtils.get_latest_checkpoint(
                self.checkpoint_dir
            )
            if resume_path:
                self._load_resume_checkpoint(resume_path, resumed_steps)
                return
            else:
                logger.warning(
                    f"is_resume=True but no resumable checkpoint in {self.checkpoint_dir}"
                )
                self.completed_steps = 0

        # Load pretrained weights (not resume)
        if pretrained_checkpoint:
            reload_modules = self.cfg.trainer.get("reload_modules", None)
            self.model = TrainerUtils.load_pretrained_checkpoint(
                self.model, pretrained_checkpoint, reload_modules=reload_modules
            )
            self.completed_steps = 0
            logger.info(f"Loaded pretrained checkpoint: {pretrained_checkpoint}")
        else:
            logger.info("No pretrained checkpoint. Starting from scratch.")
            self.completed_steps = 0

    def _load_resume_checkpoint(self, resume_path: str, completed_steps: int):
        """Resume training state from resume checkpoint."""
        # Validate GPU count
        meta_path = os.path.join(resume_path, "resume_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            saved_gpus = meta.get("num_gpus", -1)
            current_gpus = self.accelerator.num_processes
            if saved_gpus > 0 and saved_gpus != current_gpus:
                raise RuntimeError(
                    f"GPU count mismatch! Checkpoint: {saved_gpus}, current: {current_gpus}. "
                    f"Resume requires exact GPU count match."
                )

        self.completed_steps = completed_steps

        # Try to load full training state (optimizer + scheduler + RNG)
        training_state_dir = os.path.join(resume_path, "training_state")
        has_state = (
            os.path.isdir(training_state_dir)
            and len(os.listdir(training_state_dir)) > 0
        )
        has_model_state = has_state and any(
            f.endswith((".pt", ".bin", ".safetensors", ".pkl"))
            for root, dirs, files in os.walk(training_state_dir)
            for f in files
        )

        if has_model_state:
            self.accelerator.load_state(training_state_dir)
            logger.info(
                f"Resume OK: step={self.completed_steps}, path={resume_path}"
            )
        else:
            # Fallback: warm restart (model weights only)
            from training.trainer_utils.trainer_tools import TrainerUtils

            logger.warning(
                f"Checkpoint {resume_path} has no valid training_state. "
                f"Loading model weights only."
            )
            self.model = TrainerUtils.load_pretrained_checkpoint(
                self.model, resume_path
            )
            logger.info(f"Weights-only resume from step {self.completed_steps}")

    def _adjust_lr_scheduler_for_resume(self):
        """Adjust learning rate scheduler state based on completed steps."""
        if self.completed_steps > 0:
            logger.info(
                f"Adjusting LR scheduler for resume from step {self.completed_steps}"
            )
            for _ in range(self.completed_steps):
                self.lr_scheduler.step()
            logger.info(
                f"LR scheduler adjusted, current LR: {self.lr_scheduler.get_last_lr()}"
            )

    def _save_checkpoint(self):
        """Save self-contained checkpoint directory."""
        from omegaconf import OmegaConf

        if not self.checkpoint_dir:
            return

        # LoRA checkpoint
        if self.use_lora:
            if self.accelerator.is_main_process:
                checkpoint_path = os.path.join(
                    self.checkpoint_dir, f"steps_{self.completed_steps}"
                )
                from training.trainer_utils.peft import save_lora_checkpoint
                save_lora_checkpoint(
                    accelerator=self.accelerator,
                    model=self.model,
                    base_path=checkpoint_path,
                    cfg=self.cfg,
                )
            self.accelerator.wait_for_everyone()
            return

        if self.accelerator.is_main_process:
            save_format = self.cfg.trainer.get("save_format", "safetensors")
            checkpoint_dir_path = os.path.join(
                self.checkpoint_dir, f"steps_{self.completed_steps}"
            )
            os.makedirs(checkpoint_dir_path, exist_ok=True)

            # Save model weights
            state_dict = self.accelerator.get_state_dict(self.model)
            if save_format == "safetensors":
                from safetensors.torch import save_file

                torch.cuda.empty_cache()
                gc.collect()
                state_dict = {k: v.cpu().clone() for k, v in state_dict.items()}
                save_file(
                    state_dict,
                    os.path.join(checkpoint_dir_path, "model.safetensors"),
                )
                del state_dict
                gc.collect()
                torch.cuda.empty_cache()
            elif save_format == "pt":
                torch.save(
                    state_dict,
                    os.path.join(checkpoint_dir_path, "pytorch_model.pt"),
                )
            else:
                raise ValueError(
                    f"Unsupported save_format '{save_format}'. Expected 'pt' or 'safetensors'."
                )

            # Save config
            OmegaConf.save(
                self.cfg,
                os.path.join(checkpoint_dir_path, "framework_config.yaml"),
            )

            # Save dataset statistics if available
            output_dir = getattr(self.cfg, "output_dir", "")
            stats_src = os.path.join(output_dir, "dataset_statistics.json")
            if os.path.exists(stats_src):
                import shutil
                shutil.copy2(
                    stats_src,
                    os.path.join(checkpoint_dir_path, "dataset_statistics.json"),
                )

            # Save resume metadata
            resume_meta = {
                "completed_steps": self.completed_steps,
                "num_gpus": self.accelerator.num_processes,
                "gradient_accumulation_steps": self.accelerator.gradient_accumulation_steps,
                "framework_name": str(
                    getattr(self.cfg.framework, "name", "unknown")
                ),
            }
            with open(
                os.path.join(checkpoint_dir_path, "resume_meta.json"), "w"
            ) as f:
                json.dump(resume_meta, f, indent=2)

            # Save summary
            output_dir = getattr(self.cfg, "output_dir", "")
            if output_dir:
                summary_file = os.path.join(output_dir, "summary.jsonl")
                with open(summary_file, "a") as f:
                    f.write(json.dumps({"steps": self.completed_steps}) + "\n")

            logger.info(f"Checkpoint saved at {checkpoint_dir_path}")

        # All ranks must participate
        self.accelerator.wait_for_everyone()

        # Optionally save full training state
        save_training_state = self.cfg.trainer.get("save_training_state", False)
        if save_training_state:
            checkpoint_dir_path = os.path.join(
                self.checkpoint_dir, f"steps_{self.completed_steps}"
            )
            training_state_dir = os.path.join(checkpoint_dir_path, "training_state")
            try:
                self.accelerator.save_state(training_state_dir)
                logger.info(
                    f"Training state saved ({self.accelerator.num_processes} GPUs)"
                )
            except Exception as e:
                logger.warning(f"save_state failed: {e}. Resume will use warm restart.")

        self.accelerator.wait_for_everyone()

    def _maybe_eval_and_save(self):
        """Periodically evaluate and save checkpoint."""
        save_interval = self.cfg.trainer.get("save_steps", 10000)
        if self.completed_steps > 0 and self.completed_steps % save_interval == 0:
            self._save_checkpoint()

    # ═══════════════════════════════════════════════════════════════
    # W&B + Metrics Logging
    # ═══════════════════════════════════════════════════════════════

    def _init_wandb(self):
        """Initialize Weights & Biases."""
        wandb_mode = self.cfg.get("wandb_mode", None) or os.environ.get(
            "WANDB_MODE", "online"
        )
        if wandb_mode == "disabled":
            os.environ["WANDB_MODE"] = "disabled"
            return

        if self.accelerator.is_main_process:
            try:
                import wandb

                wandb_project = self.cfg.get("wandb_project", "loongforge-vla")
                wandb_entity = self.cfg.get("wandb_entity", "") or None
                run_id = self.cfg.get(
                    "run_id", self.cfg.framework.get("name", "train")
                )
                output_dir = getattr(self.cfg, "output_dir", "outputs")

                wandb.init(
                    name=run_id,
                    dir=os.path.join(output_dir, "wandb"),
                    project=wandb_project,
                    entity=wandb_entity,
                    group="vla-train",
                )
            except ImportError:
                logger.warning("wandb not installed, skipping W&B init")
            except Exception as e:
                logger.warning(f"wandb init failed: {e}")

    def _log_metrics(self, metrics: Dict[str, float]):
        """Log training metrics (W&B + local JSONL)."""
        if self.completed_steps % self.logging_frequency != 0:
            return
        if not self.accelerator.is_main_process:
            return

        # Add lr and step
        metrics["step"] = self.completed_steps
        if self.lr_scheduler is not None:
            try:
                metrics["lr"] = self.lr_scheduler.get_last_lr()[0]
            except Exception:
                pass

        # W&B
        try:
            import wandb
            if wandb.run is not None:
                wandb.log(metrics, step=self.completed_steps)
        except (ImportError, Exception):
            pass

        # Local JSONL
        output_dir = getattr(self.cfg, "output_dir", "")
        if output_dir:
            metrics_file = os.path.join(output_dir, "metrics.jsonl")
            try:
                with open(metrics_file, "a") as f:
                    f.write(json.dumps(metrics) + "\n")
            except Exception:
                pass

        # Console
        loss_val = metrics.get("total_loss", metrics.get("action_loss", float("nan")))
        lr_val = metrics.get("lr", float("nan"))
        logger.info(
            f"step {self.completed_steps:>6d}  "
            f"loss={loss_val:.5f}  lr={lr_val:.2e}"
        )

    def _log_training_config(self):
        """Print training configuration summary."""
        if not self.accelerator.is_main_process:
            return
        sep = "=" * 56
        logger.info(sep)
        logger.info("  Training Configuration")
        logger.info(f"  Total steps:       {self.max_steps}")
        logger.info(f"  Gradient clipping: {self.gradient_clipping}")
        logger.info(f"  Num processes:     {self.accelerator.num_processes}")
        logger.info(f"  Use EMA:           {self.use_ema}")
        logger.info(f"  Use LoRA:          {self.use_lora}")
        logger.info(sep)

    # ═══════════════════════════════════════════════════════════════
    # Training end processing
    # ═══════════════════════════════════════════════════════════════

    def _finalize_training(self):
        """Training end: save final model, close W&B."""
        from omegaconf import OmegaConf

        output_dir = getattr(self.cfg, "output_dir", "")
        if not output_dir:
            return

        if self.accelerator.is_main_process:
            # LoRA final save
            if self.use_lora:
                final_path = os.path.join(output_dir, "final_model")
                os.makedirs(final_path, exist_ok=True)
                from training.trainer_utils.peft import save_lora_checkpoint
                save_lora_checkpoint(
                    accelerator=self.accelerator,
                    model=self.model,
                    base_path=os.path.join(final_path, "final"),
                    cfg=self.cfg,
                )
                logger.info(f"LoRA training complete. Final model saved at {final_path}")
            else:
                # Full model save (use EMA if available)
                save_format = self.cfg.trainer.get("save_format", "safetensors")
                final_checkpoint = os.path.join(output_dir, "final_model")
                os.makedirs(final_checkpoint, exist_ok=True)

                if self.use_ema and self.ema_model is not None:
                    logger.info("Saving EMA model weights as final model")
                    state_dict = {
                        k: v.clone() for k, v in self.ema_model.state_dict().items()
                    }
                else:
                    state_dict = self.accelerator.get_state_dict(self.model)

                if save_format == "safetensors":
                    from safetensors.torch import save_file

                    torch.cuda.empty_cache()
                    gc.collect()
                    state_dict = {k: v.cpu().clone() for k, v in state_dict.items()}
                    save_file(
                        state_dict,
                        os.path.join(final_checkpoint, "model.safetensors"),
                    )
                    del state_dict
                    gc.collect()
                    torch.cuda.empty_cache()
                elif save_format == "pt":
                    torch.save(
                        state_dict,
                        os.path.join(final_checkpoint, "pytorch_model.pt"),
                    )

                # Save config
                OmegaConf.save(
                    self.cfg,
                    os.path.join(final_checkpoint, "framework_config.yaml"),
                )

                logger.info(f"Final model saved at {final_checkpoint}")

            # Close W&B
            try:
                import wandb
                if wandb.run is not None:
                    wandb.finish()
            except (ImportError, Exception):
                pass

        self.accelerator.wait_for_everyone()
