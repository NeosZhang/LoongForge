"""
overwatch.py - Centralized distributed-aware logging

Provides two modes:
  - DistributedOverwatch: Multi-GPU training, only rank=0 outputs INFO
  - PureOverwatch: Single-GPU / non-distributed scenario

Usage:
    from training.trainer_utils.overwatch import initialize_overwatch
    overwatch = initialize_overwatch(__name__)
    overwatch.info("message")
"""

import logging
import os
from contextlib import nullcontext
from logging import LoggerAdapter
from typing import Any, Callable, ClassVar, Dict, MutableMapping, Tuple, Union


# ═══════════════════════════════════════════════════════════════
# Context-aware LoggerAdapter
# ═══════════════════════════════════════════════════════════════

class ContextAdapter(LoggerAdapter):
    """Prepend log messages with context prefix for better readability."""
    CTX_PREFIXES: ClassVar[Dict[int, str]] = {
        **{0: "[*] "},
        **{idx: "|=> ".rjust(4 + (idx * 4)) for idx in [1, 2, 3]},
    }

    def process(
        self, msg: str, kwargs: MutableMapping[str, Any]
    ) -> Tuple[str, MutableMapping[str, Any]]:
        ctx_level = kwargs.pop("ctx_level", 0)
        return f"{self.CTX_PREFIXES[ctx_level]}{msg}", kwargs


# ═══════════════════════════════════════════════════════════════
# Distributed Overwatch (multi-GPU)
# ═══════════════════════════════════════════════════════════════

class DistributedOverwatch:
    """Centralized logging under distributed training."""
    def __init__(self, name: str) -> None:
        from accelerate import PartialState

        self.logger = ContextAdapter(logging.getLogger(name), extra={})
        self.distributed_state = PartialState()

        # Logger delegation
        self.debug = self.logger.debug
        self.info = self.logger.info
        self.warning = self.logger.warning
        self.error = self.logger.error
        self.critical = self.logger.critical

        # Only log INFO on main process, ERROR on others
        self.logger.setLevel(
            logging.INFO if self.distributed_state.is_main_process else logging.ERROR
        )

    @property
    def rank_zero_only(self) -> Callable[..., Any]:
        """rank zero logging wrappers."""
        return self.distributed_state.on_main_process

    @property
    def local_zero_only(self) -> Callable[..., Any]:
        """Local rank zero logging wrappers."""
        return self.distributed_state.on_local_main_process

    @property
    def rank_zero_first(self) -> Callable[..., Any]:
        """Rank zero logging wrappers."""
        return self.distributed_state.main_process_first

    @property
    def local_zero_first(self) -> Callable[..., Any]:
        """Local rank zero logging wrappers."""
        return self.distributed_state.local_main_process_first

    def is_rank_zero(self) -> bool:
        """check if is rank zero process."""
        return self.distributed_state.is_main_process

    def rank(self) -> int:
        """Get current process index."""
        return self.distributed_state.process_index

    def local_rank(self) -> int:
        """Get current local process index."""
        return self.distributed_state.local_process_index

    def world_size(self) -> int:
        """Get total number of processes."""
        return self.distributed_state.num_processes


# ═══════════════════════════════════════════════════════════════
# Pure Overwatch (single-GPU / non-distributed)
# ═══════════════════════════════════════════════════════════════

class PureOverwatch:
    """Centralized logging in non-distributed scenario."""
    def __init__(self, name: str) -> None:
        self.logger = ContextAdapter(logging.getLogger(name), extra={})

        self.debug = self.logger.debug
        self.info = self.logger.info
        self.warning = self.logger.warning
        self.error = self.logger.error
        self.critical = self.logger.critical

        self.logger.setLevel(logging.INFO)

    @staticmethod
    def get_identity_ctx() -> Callable[..., Any]:
        """Return identity context manager."""
        def identity(fn: Callable[..., Any]) -> Callable[..., Any]:
            return fn
        return identity

    @property
    def rank_zero_only(self) -> Callable[..., Any]:
        """Rank zero logging wrappers."""
        return self.get_identity_ctx()

    @property
    def local_zero_only(self) -> Callable[..., Any]:
        """Local rank zero logging wrappers."""
        return self.get_identity_ctx()

    @property
    def rank_zero_first(self) -> Callable[..., Any]:
        """Rank zero logging wrappers."""
        return nullcontext

    @property
    def local_zero_first(self) -> Callable[..., Any]:
        """Local rank zero logging wrappers."""
        return nullcontext

    @staticmethod
    def is_rank_zero() -> bool:
        """Check if current process is rank zero."""
        return True

    @staticmethod
    def rank() -> int:
        """Get current process index."""
        return 0

    @staticmethod
    def local_rank() -> int:
        """Get current local process index."""
        return 0

    @staticmethod
    def world_size() -> int:
        """Get total number of processes."""
        return 1


def initialize_overwatch(name: str) -> Union[DistributedOverwatch, PureOverwatch]:
    """Automatically select Overwatch type based on environment."""
    return (
        DistributedOverwatch(name)
        if int(os.environ.get("WORLD_SIZE", -1)) != -1
        else PureOverwatch(name)
    )
