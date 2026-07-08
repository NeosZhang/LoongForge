# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""VLA training config validation.

Called by parse_train_args() on the DictConfig stage (before to_object()), so it
benefits from OmegaConf missing-key / interpolation checks. Operates on the three
typed configs: training_args (TrainingArgs), model_cfg (ModelConfig),
data_cfg (DataConfig).
"""

import logging
import os

logger = logging.getLogger(__name__)


def validate(training_args, model_cfg, data_cfg):
    """Validate the combination of TrainingArgs + ModelConfig + DataConfig.

    Raises ValueError on hard errors, logs warnings for soft issues.
    """
    # ── Learning rate ──
    if training_args.lr_base <= 0:
        raise ValueError(f"--lr-base must be positive, got {training_args.lr_base}")
    if training_args.min_lr < 0:
        raise ValueError(f"--min-lr must be >= 0, got {training_args.min_lr}")
    if training_args.min_lr >= training_args.lr_base:
        logger.warning(
            f"--min-lr ({training_args.min_lr}) >= --lr-base ({training_args.lr_base}); "
            f"cosine decay will have no effect."
        )

    # ── Steps ──
    if training_args.train_iters <= 0:
        raise ValueError(f"--train-iters must be positive, got {training_args.train_iters}")
    if training_args.lr_warmup_iters >= training_args.train_iters:
        logger.warning(
            f"--lr-warmup-iters ({training_args.lr_warmup_iters}) >= --train-iters ({training_args.train_iters})"
        )
    if training_args.cuda_graph_warmup_steps <= 0:
        raise ValueError(
            f"--cuda-graph-warmup-steps must be positive, got {training_args.cuda_graph_warmup_steps}"
        )

    # ── CUDA graph ──
    if training_args.cuda_graph_impl == "local":
        logger.warning(
            "Host-side loss/grad NaN checks are disabled because CUDA graph mode "
            "is enabled."
        )

        if training_args.cuda_graph_pad_length is None:
            raise ValueError(
                "--cuda-graph-pad-length must be set when --cuda-graph-impl=local."
            )
        if training_args.cuda_graph_pad_length < 0:
            raise ValueError(
                f"--cuda-graph-pad-length must be non-negative, got {training_args.cuda_graph_pad_length}"
            )
        if training_args.cuda_graph_scope not in {"full_iteration", "per_microbatch"}:
            raise ValueError(
                f"Unsupported --cuda-graph-scope={training_args.cuda_graph_scope!r} in embodied trainer."
            )

    # ── Tokenizer ──
    if training_args.tokenizer_path is None and not os.environ.get("TOKENIZER_PATH"):
        logger.warning(
            "Neither --tokenizer-path nor TOKENIZER_PATH env var is set. "
            "Model initialization may fail if a tokenizer is required."
        )

    # ── ZeRO optimizer options ──
    if training_args.zero_parameters_as_bucket_view and not training_args.zero_optimizer:
        logger.warning(
            "--zero-parameters-as-bucket-view has no effect without --zero-optimizer."
        )

    # ── Profiler mutual exclusion ──
    if training_args.use_pytorch_profiler and training_args.use_nsys_profiler:
        raise ValueError(
            "--use-pytorch-profiler and --use-nsys-profiler are mutually exclusive."
        )
    if training_args.use_pytorch_profiler or training_args.use_nsys_profiler:
        if training_args.profile_step_end < training_args.profile_step_start:
            raise ValueError(
                f"--profile-step-end ({training_args.profile_step_end}) must be greater than "
                f"--profile-step-start ({training_args.profile_step_start})."
            )

    # ── Model config sanity ──
    if not model_cfg.model_type:
        raise ValueError("ModelConfig.model_type must be set (from YAML model.model_type).")
