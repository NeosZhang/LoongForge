# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""BaseTrainer — pure native PyTorch distributed training skeleton."""

import copy
import gc
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from loongforge.embodied.distributed import DistributedContext
from loongforge.embodied.distributed.parallel import unwrap_model, wrap_model
from loongforge.embodied.distributed.checkpoint import (
    get_latest_checkpoint,
    load_pretrained,
    resume_training_state,
    save_checkpoint,
)
from loongforge.embodied.distributed.utils import set_seed, setup_logging

logger = logging.getLogger(__name__)


class BaseTrainer(ABC):
    """
    Training skeleton — Template Method pattern.

    Lifecycle: __init__(args) → train() → [_setup → _training_loop → _finalize]

    Data flow:
      - args.model_cfg (OmegaConf from YAML): model structure → _build_model()
      - args.* (CLI flags): training control → optimizer, scheduler, data, etc.

    Subclass override points:
      - _build_model() → construct model from model_cfg
      - _build_dataloaders() → construct dataloaders from CLI args
      - _train_forward(batch) → single forward pass, return dict with 'action_loss'
      - _on_train_begin() → hook before training loop
      - _on_step_end(metrics) → hook after each step
    """

    def __init__(self, args):
        self.args = args
        self.model_cfg = args.model_cfg

        # Initialized in _setup()
        self.ctx: Optional[DistributedContext] = None
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.lr_scheduler = None
        self.dataloaders: Dict[str, DataLoader] = {}

        # Training state
        self.completed_steps: int = 0
        self.current_epoch: int = 0
        self.max_steps: int = args.max_train_steps

        # Data iterators (managed by _fetch_batch for epoch cycling)
        self._data_iters: Dict[str, Any] = {}

        # EMA
        self.ema_model: Optional[nn.Module] = None

    # ═══════════════════════════════════════════════
    # Public interface
    # ═══════════════════════════════════════════════

    def train(self):
        """Main entry point."""
        self._setup()
        self._training_loop()
        self._finalize()

    # ═══════════════════════════════════════════════
    # Setup
    # ═══════════════════════════════════════════════

    def _setup(self):
        """One-shot initialization of all training resources."""
        args = self.args

        # 1. Distributed context
        self.ctx = DistributedContext()
        self.ctx.init()

        # 2. Seed
        set_seed(args.seed + self.ctx.rank)

        # 3. Output directories + logging
        self.output_dir = args.output_dir
        self.checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        if self.ctx.is_main:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.ctx.barrier()
        setup_logging(self.output_dir, self.ctx.rank)

        # 4. Build model (from YAML model_cfg)
        self.model = self._build_model()

        # 5. Pretrained weights / Resume (before wrapping)
        if args.resume:
            self._handle_resume()
        elif args.pretrained_checkpoint:
            self._load_pretrained(args.pretrained_checkpoint)

        # 6. Freeze modules
        self._freeze_modules(args.freeze_modules)

        # 7. Parallel wrapping (DDP/FSDP + mixed precision via policy)
        self.model = wrap_model(self.model, args, self.ctx)

        # 8. Optimizer + Scheduler (after wrapping; FSDP use_orig_params=True)
        self.optimizer = self._build_optimizer()
        self.lr_scheduler = self._build_scheduler()

        # 9. Resume optimizer/scheduler state (after wrapping + optimizer creation)
        if args.resume and self.completed_steps > 0:
            latest_path, _ = get_latest_checkpoint(self.checkpoint_dir)
            if latest_path:
                resume_training_state(
                    self.model, self.optimizer, self.lr_scheduler, latest_path, self.ctx
                )

        # 10. Data
        self.dataloaders = self._build_dataloaders()

        # 11. EMA
        if args.ema:
            self._init_ema()

        # 12. W&B
        self._init_wandb()

        # 13. Print stats
        self._log_param_stats()

        # Hook
        self._on_train_begin()

    # ═══════════════════════════════════════════════
    # Training loop
    # ═══════════════════════════════════════════════

    def _training_loop(self):
        args = self.args
        grad_accum = args.gradient_accumulation_steps
        grad_clip = args.gradient_clipping
        log_interval = args.logging_frequency
        save_interval = args.save_steps
        loss_spike_threshold = args.loss_spike_threshold

        self._init_data_iterator("vla")
        pbar = self._make_pbar()
        self._log_training_config()

        while self.completed_steps < self.max_steps:
            self.optimizer.zero_grad()
            t0 = time.perf_counter()

            # ── Gradient Accumulation ──
            accum_loss = 0.0
            for micro in range(grad_accum):
                batch = self._fetch_batch("vla")

                output = self._train_forward(batch)
                loss = output["action_loss"] / grad_accum

                # Loss spike protection: zero out to prevent NaN propagation
                loss_val = loss.detach().item() * grad_accum
                if torch.isnan(loss) or torch.isinf(loss) or loss_val > loss_spike_threshold:
                    if self.ctx.is_main:
                        logger.warning(
                            f"[step {self.completed_steps}] Loss spike: {loss_val:.4f}, zeroing"
                        )
                    loss = loss * 0.0

                # Backward with no_sync for non-last micro steps
                if self.ctx.is_distributed and micro < grad_accum - 1:
                    if hasattr(self.model, 'no_sync'):
                        with self.model.no_sync():
                            loss.backward()
                    else:
                        # FSDP2 (fully_shard): use set_requires_gradient_sync
                        self.model.set_requires_gradient_sync(False)
                        loss.backward()
                        self.model.set_requires_gradient_sync(True)
                else:
                    loss.backward()

                accum_loss += loss.detach().item()

            # ── NaN gradient cleanup ──
            self._clean_nan_gradients()

            # ── Gradient clipping ──
            if grad_clip > 0:
                self._clip_gradients(grad_clip)

            # ── Optimizer step ──
            self.optimizer.step()
            self.lr_scheduler.step()
            self.completed_steps += 1

            # ── Metrics ──
            metrics = self._collect_metrics(output, accum_loss, time.perf_counter() - t0)
            self._on_step_end(metrics)

            # ── Logging ──
            if self.completed_steps % log_interval == 0:
                self._log_metrics(metrics)

            # ── Checkpoint ──
            if self.completed_steps % save_interval == 0:
                self._save_checkpoint()

            # ── Progress bar ──
            if pbar:
                pbar.update(1)
                pbar.set_postfix(
                    loss=f"{metrics.get('action_loss', 0):.4f}",
                    lr=f"{metrics.get('lr', 0):.2e}",
                )

        if pbar:
            pbar.close()

    # ═══════════════════════════════════════════════
    # Abstract methods — subclass must implement
    # ═══════════════════════════════════════════════

    @abstractmethod
    def _build_model(self) -> nn.Module:
        """Build model from self.model_cfg. Return unwrapped model."""
        ...

    @abstractmethod
    def _build_dataloaders(self) -> Dict[str, DataLoader]:
        """Build dataloaders from self.args. Must include 'vla' key."""
        ...

    @abstractmethod
    def _train_forward(self, batch) -> Dict[str, torch.Tensor]:
        """Single forward pass. Must return dict with 'action_loss' key."""
        ...

    # ═══════════════════════════════════════════════
    # Optional hooks
    # ═══════════════════════════════════════════════

    def _on_train_begin(self):
        """Hook before training loop starts."""
        pass

    def _on_step_end(self, metrics: Dict[str, float]):
        """Hook after each training step. Default: EMA update."""
        self._ema_update()

    # ═══════════════════════════════════════════════
    # Internal methods
    # ═══════════════════════════════════════════════

    def _load_pretrained(self, path: str):
        """Load pretrained weights, preferring architecture.load_pretrained if available."""
        arch = getattr(self.model, "architecture", None)
        if arch and hasattr(arch, "load_pretrained"):
            arch.load_pretrained(path, device=self.ctx.device)
            if self.ctx.is_main:
                logger.info(f"Pretrained loaded via architecture: {path}")
        else:
            load_pretrained(self.model, path, self.ctx)

    def _handle_resume(self):
        """Resume from latest checkpoint."""
        path, step = get_latest_checkpoint(self.checkpoint_dir)
        if path:
            load_pretrained(self.model, path, self.ctx)
            self.completed_steps = step
            if self.ctx.is_main:
                logger.info(f"Resumed model weights from step {step}")
        else:
            if self.ctx.is_main:
                logger.warning("--resume set but no checkpoint found, starting from scratch")

    def _freeze_modules(self, freeze_str: str):
        """Freeze specified modules by dot-path."""
        if not freeze_str:
            return
        for path in [p.strip() for p in freeze_str.split(",") if p.strip()]:
            module = self.model
            try:
                for attr in path.split("."):
                    module = getattr(module, attr)
                for p in module.parameters():
                    p.requires_grad = False
                if self.ctx.is_main:
                    logger.info(f"Frozen: {path}")
            except AttributeError:
                if self.ctx.is_main:
                    logger.warning(f"Freeze target not found: {path}")

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Build AdamW with per-module LR groups."""
        from embodied.optimizer import build_optimizer
        return build_optimizer(self.model, self.args)

    def _build_scheduler(self):
        """Build LR scheduler."""
        from embodied.optimizer import build_scheduler
        return build_scheduler(self.optimizer, self.args)

    def _clip_gradients(self, max_norm: float):
        """Gradient clipping."""
        from embodied.optimizer import clip_gradients
        clip_gradients(self.model, max_norm)

    def _clean_nan_gradients(self):
        """Replace NaN/Inf gradients with 0."""
        from embodied.optimizer import clean_nan_gradients
        clean_nan_gradients(self.model)

    def _init_data_iterator(self, name: str):
        """Initialize iterator for named dataloader and store in self._data_iters."""
        self._data_iters[name] = iter(self.dataloaders[name])

    def _fetch_batch(self, dl_name: str):
        """Fetch next batch, handle epoch boundary by cycling the iterator."""
        try:
            return next(self._data_iters[dl_name])
        except StopIteration:
            self.current_epoch += 1
            dl = self.dataloaders[dl_name]
            if hasattr(dl, "sampler") and hasattr(dl.sampler, "set_epoch"):
                dl.sampler.set_epoch(self.current_epoch)
            self._data_iters[dl_name] = iter(dl)
            return next(self._data_iters[dl_name])

    def _save_checkpoint(self):
        save_checkpoint(
            self.model, self.optimizer, self.lr_scheduler,
            self.completed_steps, self.checkpoint_dir, self.ctx, self.args,
        )

    # ── EMA ──

    def _init_ema(self):
        if not self.ctx.is_main:
            return
        # FSDP shards parameters — EMA needs full params.
        # Skip EMA when FSDP is active (proper FSDP EMA requires summon_full_params).
        if self.args.distributed_strategy == "fsdp":
            logger.warning("EMA disabled under FSDP (requires summon_full_params). Skipping.")
            self.ema_model = None
            return
        raw = unwrap_model(self.model)
        self.ema_model = copy.deepcopy(raw).cpu().eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)
        logger.info(f"EMA initialized, decay={self.args.ema_decay}")

    def _ema_update(self):
        if not self.ema_model or not self.ctx.is_main:
            return
        raw = unwrap_model(self.model)
        with torch.no_grad():
            for ep, mp in zip(self.ema_model.parameters(), raw.parameters()):
                ep.data.mul_(self.args.ema_decay).add_(mp.data.cpu(), alpha=1 - self.args.ema_decay)

    # ── Logging ──

    def _collect_metrics(self, output: Dict, accum_loss: float, step_time: float) -> Dict[str, float]:
        metrics = {"action_loss": accum_loss, "step_time": step_time, "step": self.completed_steps}
        for k, v in output.items():
            if k == "action_loss":
                continue
            if isinstance(v, torch.Tensor) and v.numel() == 1:
                metrics[k] = v.item()
            elif isinstance(v, (int, float)):
                metrics[k] = v
        metrics["lr"] = self.lr_scheduler.get_last_lr()[0]
        return metrics

    def _log_metrics(self, metrics: Dict[str, float]):
        if not self.ctx.is_main:
            return
        loss = metrics.get("action_loss", float("nan"))
        lr = metrics.get("lr", 0)
        logger.info(f"step {self.completed_steps:>6d}  loss={loss:.5f}  lr={lr:.2e}")

        # W&B
        try:
            import wandb

            if wandb.run:
                wandb.log(metrics, step=self.completed_steps)
        except Exception:
            pass

        # JSONL
        metrics_file = os.path.join(self.output_dir, "metrics.jsonl")
        try:
            with open(metrics_file, "a") as f:
                f.write(json.dumps(metrics) + "\n")
        except Exception:
            pass

    def _init_wandb(self):
        if self.args.wandb_mode == "disabled" or not self.ctx.is_main:
            return
        try:
            import wandb

            wandb.init(
                project=self.args.wandb_project,
                name=os.path.basename(self.output_dir),
                mode=self.args.wandb_mode,
            )
        except Exception as e:
            logger.warning(f"W&B init failed: {e}")

    def _log_training_config(self):
        if not self.ctx.is_main:
            return
        args = self.args
        logger.info("=" * 60)
        logger.info(f"  Model type:      {self.model_cfg.model_type}")
        logger.info(f"  Architecture:    {self.model_cfg.architecture}")
        logger.info(f"  Phase:           {args.training_phase}")
        logger.info(f"  Strategy:        {args.distributed_strategy}")
        logger.info(f"  Dtype:           {args.dtype}")
        logger.info(f"  World size:      {self.ctx.world_size}")
        logger.info(f"  Max steps:       {args.max_train_steps}")
        logger.info(f"  Batch/GPU:       {args.per_device_batch_size}")
        logger.info(f"  Grad accum:      {args.gradient_accumulation_steps}")
        logger.info(f"  LR:              {args.lr}")
        logger.info(f"  Freeze:          {args.freeze_modules or '(none)'}")
        logger.info("=" * 60)

    def _log_param_stats(self):
        if not self.ctx.is_main:
            return
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(
            f"Parameters: {total / 1e6:.1f}M total, {trainable / 1e6:.1f}M trainable "
            f"({100 * trainable / max(total, 1):.1f}%)"
        )

    def _make_pbar(self):
        if not self.ctx.is_main:
            return None
        try:
            from tqdm import tqdm

            return tqdm(range(self.max_steps), initial=self.completed_steps, desc="Training")
        except ImportError:
            return None

    # ── Finalize ──

    def _finalize(self):
        """End of training: save final model, close W&B."""
        # Final checkpoint
        self._save_checkpoint()

        # Save EMA as final model
        if self.ctx.is_main and self.ema_model:
            final_path = os.path.join(self.output_dir, "final_model")
            os.makedirs(final_path, exist_ok=True)
            from safetensors.torch import save_model

            torch.cuda.empty_cache()
            gc.collect()
            ema_cpu = copy.deepcopy(self.ema_model).cpu()
            save_model(ema_cpu, os.path.join(final_path, "model.safetensors"))
            logger.info(f"EMA final model saved: {final_path}")

        # Close W&B
        if self.ctx.is_main:
            try:
                import wandb

                if wandb.run:
                    wandb.finish()
            except Exception:
                pass

        self.ctx.barrier()
        self.ctx.destroy()
