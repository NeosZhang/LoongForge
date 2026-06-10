# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""BaseTrainer — pure native PyTorch distributed training skeleton."""

import copy
import gc
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
    restore_rank_rng_state,
    resume_training_state,
    save_checkpoint,
)


from loongforge.embodied.train.utils.utils import log_stage, set_seed, setup_logging, set_deterministic
from loongforge.embodied.train.utils.logging import TrainingLogger, log_effective_config

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
        self.logger: Optional[TrainingLogger] = None

        # Training state
        self.completed_steps: int = 0
        self.current_epoch: int = 0
        self.train_iters: int = args.train_iters

        # Data iterators (managed by _fetch_batch for epoch cycling)
        self._data_iters: Dict[str, Any] = {}
        self._resume_dataloader_state: Dict[str, Dict[str, Any]] = {}
        self._resume_rng_per_rank = None

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

        # 2. Seed — use the same seed on all ranks (align with lerobot/accelerate baseline).
        # DistributedSampler handles per-rank data partitioning internally via its own seed+rank offset.
        set_seed(args.seed)
        if getattr(args, "deterministic_mode", False):
            set_deterministic()

        # 3. Output directories + logging
        self.output_dir = args.output_dir
        self.checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        if self.ctx.is_main:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.ctx.barrier()
        setup_logging(self.output_dir, self.ctx.rank)

        # Dump fully-resolved CLI args + model config now that the file
        # handler is attached, so the effective config also lands in the log.
        log_effective_config(args)

        # 4. TrainingLogger (initialize early for logging during setup)
        self.logger = TrainingLogger(
            output_dir=self.output_dir,
            wandb_project=args.wandb_project,
            wandb_mode=args.wandb_mode,
            is_main=self.ctx.is_main,
            model_cfg=self.model_cfg,
        )
        self.logger.init_wandb(os.path.basename(self.output_dir))

        # 5. Build model (from YAML model_cfg)
        mt = getattr(self.model_cfg, "model_type", "?")
        arch = getattr(self.model_cfg, "architecture", "?")
        with log_stage(
            "model",
            start_msg=f"building: model_type={mt}  architecture={arch}",
            end_msg="built in {elapsed}",
        ):
            self.model = self._build_model()

        # 5. Pretrained weights / Resume (before wrapping)
        latest_path = None
        if args.resume:
            latest_path, latest_step, latest_epoch = get_latest_checkpoint(self.checkpoint_dir)
            assert latest_path, (
                f"--resume set but no checkpoint was found in {self.checkpoint_dir}. "
                f"Point --output-dir to a run whose checkpoints/ directory contains "
                f"a checkpoint, or drop --resume to start from scratch."
            )
            with log_stage(
                "ckpt",
                start_msg=f"resume requested: dir=={latest_path}",
                end_msg="resume done in {elapsed}",
            ):
                self._handle_resume(latest_path, latest_step, latest_epoch)
        elif args.pretrained_checkpoint:
            with log_stage(
                "ckpt",
                start_msg=f"loading pretrained: {args.pretrained_checkpoint}",
                end_msg="pretrained loaded in {elapsed}",
            ):
                self._load_pretrained(args.pretrained_checkpoint)

        # 6. Freeze modules
        self._freeze_modules(args.freeze_modules)

        with log_stage(
            "wrap_model",
            start_msg=f"wrap_model: strategy={args.distributed_strategy}, dtype={args.dtype}",
            end_msg="done in {elapsed}",
        ):
            # 7. Parallel wrapping (DDP/FSDP + mixed precision via policy)
            self.model = wrap_model(self.model, args, self.ctx)

        with log_stage(
            "optimizer", 
            start_msg="building optimizer", end_msg="optimizer built in {elapsed}"
        ):
            # 8. Optimizer + Scheduler (after wrapping; FSDP use_orig_params=True)
            self.optimizer = self._build_optimizer()
            self.lr_scheduler = self._build_scheduler()

        # 9. Resume optimizer/scheduler/RNG state (after wrapping + optimizer creation).
        # Gate on `args.resume` only — step==0 is a valid resume point and must
        # still restore optimizer/scheduler/RNG/dataloader, otherwise we'd run
        # with resumed weights but a fresh optimizer (half-resume).
        if args.resume and latest_path:
            with log_stage(
                "ckpt",
                start_msg=f"restoring optimizer/scheduler/RNG state from {latest_path}",
                end_msg="optimizer/scheduler/RNG state restored in {elapsed}",
            ):
                saved_epoch, dataloader_state, rng_per_rank = resume_training_state(
                    self.model, self.optimizer, self.lr_scheduler, latest_path, self.ctx,
                    args=self.args,
                    restore_rng=False,
                )
                # Trust the epoch from training_state.pt over resume_meta.json
                # when present (training_state is the freshest source). Use
                # `is not None` so a legitimate saved_epoch=0 still overrides.
                if saved_epoch is not None:
                    self.current_epoch = saved_epoch
                self._resume_dataloader_state = dataloader_state or {}
                self._resume_rng_per_rank = rng_per_rank

        # 10. Data
        with log_stage("data", start_msg="building dataloaders"):
            self.dataloaders = self._build_dataloaders()
            self._restore_dataloader_states()
        # 11. EMA
        if args.ema:
            self._init_ema()

        # 12. Print stats
        self.logger.log_param_stats(self.model)

        # Hook
        self._on_train_begin()

    # ═══════════════════════════════════════════════
    # Training loop
    # ═══════════════════════════════════════════════

    def _training_loop(self):
        args = self.args
        grad_accum = args.gradient_accumulation_steps
        grad_clip = args.clip_grad
        log_interval = args.log_interval
        save_interval = args.save_interval
        loss_spike_threshold = args.loss_spike_threshold

        self._init_data_iterator("vla")

        while self.completed_steps < self.train_iters:
            self.optimizer.zero_grad()
            t0 = time.perf_counter()

            # ── Gradient Accumulation ──
            accum_loss = 0.0
            for micro in range(grad_accum):
                batch = self._fetch_batch("vla")

                output = self._train_forward(batch)
                loss = output["action_loss"] / grad_accum

                # Loss spike protection: zero out to prevent NaN propagation
                loss_val = output["action_loss"].detach().item()
                if torch.isnan(loss) or torch.isinf(loss) or loss_val > loss_spike_threshold:
                    self.logger.log_loss_spike(self.completed_steps, loss_val)
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

                accum_loss += loss_val

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
            step_time = time.perf_counter() - t0
            batch_size = grad_accum * args.per_device_batch_size
            metrics = self.logger.collect_metrics(
                output, accum_loss, step_time,
                self.completed_steps, self.lr_scheduler,
                self.completed_steps * batch_size,
                self.model, batch_size,
            )
            self._on_step_end(metrics)

            # ── Logging ──
            if self.completed_steps % log_interval == 0:
                self.logger.log_metrics(
                    metrics, self.completed_steps, self.train_iters,
                    args.per_device_batch_size, self.ctx.world_size, self.ctx.is_distributed,
                )

            # ── Checkpoint ──
            if self.completed_steps % save_interval == 0:
                self._save_checkpoint()

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
            self.logger.log_pretrained_loaded(path, via_architecture=True)
        else:
            load_pretrained(self.model, path, self.ctx)
            self.logger.log_pretrained_loaded(path, via_architecture=False)

    def _handle_resume(self, path: str, step: int, epoch: int):
        """Resume model weights from a discovered checkpoint."""
        load_pretrained(self.model, path, self.ctx)
        self.completed_steps = step
        self.current_epoch = epoch
        self.logger.log_resume(step)

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
                self.logger.log_frozen_module(path)
            except AttributeError:
                self.logger.log_freeze_not_found(path)

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Build AdamW with per-module LR groups."""
        from loongforge.embodied.optimizer import build_optimizer
        return build_optimizer(self.model, self.args)

    def _build_scheduler(self):
        """Build LR scheduler."""
        from loongforge.embodied.optimizer import build_scheduler
        return build_scheduler(self.optimizer, self.args)

    def _clip_gradients(self, max_norm: float):
        """Gradient clipping."""
        from loongforge.embodied.optimizer import clip_gradients
        clip_gradients(self.model, max_norm)

    def _clean_nan_gradients(self):
        """Replace NaN/Inf gradients with 0."""
        from loongforge.embodied.optimizer import clean_nan_gradients
        clean_nan_gradients(self.model)

    def _restore_dataloader_states(self):
        """Restore full dataloader states when checkpoints provide them."""
        if not self._resume_dataloader_state:
            return
        for name, state in self._resume_dataloader_state.items():
            dl = self.dataloaders.get(name)
            if dl is None:
                if self.ctx.is_main:
                    logger.warning(f"Checkpoint has dataloader state for unknown loader: {name}")
                continue
            if hasattr(dl, "load_state_dict"):
                dl.load_state_dict(state)
                if self.ctx.is_main:
                    logger.info(f"Restored dataloader state: {name}")
            elif self.ctx.is_main:
                logger.warning(
                    f"Dataloader '{name}' does not support load_state_dict(); "
                    "dataloader state in checkpoint will be ignored."
                )

    def _init_data_iterator(self, name: str):
        """Initialize iterator for named dataloader and store in self._data_iters.

        The trainer (`self.current_epoch`) is the single source of truth — it
        is set by checkpoint resume (meta.json then training_state.pt) before
        this runs. We push that value INTO the sampler via `set_epoch()` so
        the sampler's shuffle stream is aligned, instead of reading sampler.epoch
        back into self.current_epoch (which would clobber a resumed value to 0
        when the sampler's load_state_dict didn't restore it).

        SKIP `set_epoch` when this loader's state was just restored via
        `dl.load_state_dict()` — `StatefulDistributedSampler.set_epoch` clears
        the `_yielded` progress counter, which would re-emit the epoch from
        sample 0 and silently undo the in-epoch resume position.
        """
        dl = self.dataloaders[name]
        sampler = getattr(dl, "sampler", None)
        restored_from_state = name in self._resume_dataloader_state
        if (
            sampler is not None
            and hasattr(sampler, "set_epoch")
            and not restored_from_state
        ):
            sampler.set_epoch(self.current_epoch)
        self._data_iters[name] = iter(dl)
        if self._resume_rng_per_rank is not None:
            restore_rank_rng_state(self._resume_rng_per_rank, self.ctx)
            if self.ctx.is_main:
                logger.info("RNG state resumed successfully after dataloader iterator init")
            self._resume_rng_per_rank = None
        if self.ctx.is_main:
            logger.info(f"Dataloader '{name}' positioned at epoch={self.current_epoch}")

    def _advance_epoch(self, name: str):
        """Move dataloader to the next epoch."""
        self.current_epoch += 1
        dl = self.dataloaders[name]
        if hasattr(dl, "sampler") and hasattr(dl.sampler, "set_epoch"):
            dl.sampler.set_epoch(self.current_epoch)
        self._data_iters[name] = iter(dl)

    def _fetch_batch(self, dl_name: str):
        """Fetch next batch, handle epoch boundary by cycling the iterator."""
        try:
            batch = next(self._data_iters[dl_name])
        except StopIteration:
            self._advance_epoch(dl_name)
            batch = next(self._data_iters[dl_name])

        device = next(self.model.parameters()).device
        if hasattr(batch, "to"):
            batch = batch.to(device)
        return batch

    def _get_dataloader_state(self) -> Dict[str, Dict[str, Any]]:
        """Return full dataloader states for exact checkpoint resume when supported."""
        states = {}
        for name, dl in self.dataloaders.items():
            if name in self._data_iters and hasattr(dl, "state_dict"):
                states[name] = dl.state_dict()
            elif self.ctx.is_main:
                logger.warning(
                    f"Dataloader '{name}' has not been iterated or does not support state_dict(); "
                    "dataloader state will not be saved in this checkpoint."
                )
        return states

    def _save_checkpoint(self):
        save_checkpoint(
            self.model, self.optimizer, self.lr_scheduler,
            self.completed_steps, self.checkpoint_dir, self.ctx, self.args,
            epoch=self.current_epoch,
            dataloader_state=self._get_dataloader_state(),
        )

    # ── EMA ──

    def _init_ema(self):
        if not self.ctx.is_main:
            return
        # FSDP shards parameters — EMA needs full params.
        # Skip EMA when FSDP is active (proper FSDP EMA requires summon_full_params).
        if self.args.distributed_strategy == "fsdp":
            self.logger.log_ema_disabled_fsdp()
            self.ema_model = None
            return
        raw = unwrap_model(self.model)
        self.ema_model = copy.deepcopy(raw).cpu().eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)
        self.logger.log_ema_initialized(self.args.ema_decay)

    def _ema_update(self):
        if not self.ema_model or not self.ctx.is_main:
            return
        raw = unwrap_model(self.model)
        with torch.no_grad():
            for ep, mp in zip(self.ema_model.parameters(), raw.parameters()):
                ep.data.mul_(self.args.ema_decay).add_(mp.data.cpu(), alpha=1 - self.args.ema_decay)

    # ── Finalize ──

    def _finalize(self):
        """End of training: save final model, close W&B."""
        # Final checkpoint — skip when the last loop iteration already saved
        # at this exact step (train_iters % save_interval == 0), otherwise
        # we'd overwrite the same steps_{N} dir twice and waste I/O.
        if self.completed_steps % self.args.save_interval != 0:
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
            self.logger.log_final_model_saved(final_path)

        # Close W&B
        self.logger.finish_wandb()

        self.ctx.barrier()
        self.ctx.destroy()
