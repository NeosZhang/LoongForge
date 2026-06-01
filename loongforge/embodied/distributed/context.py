# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Distributed environment context, replacing HuggingFace Accelerator."""

import os

import torch
import torch.distributed as dist


class DistributedContext:
    """Global distributed context — single instance throughout training lifetime."""

    def __init__(self, backend: str = "nccl"):
        self.backend = backend
        self.rank: int = 0
        self.local_rank: int = 0
        self.world_size: int = 1
        self.device: torch.device = torch.device("cpu")
        self._initialized: bool = False

    def init(self):
        """Initialize from torchrun environment variables."""
        if "RANK" not in os.environ:
            # Single-card mode
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
                torch.cuda.set_device(self.device)
            return

        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.device = torch.device(f"cuda:{self.local_rank}")
        torch.cuda.set_device(self.device)
        dist.init_process_group(backend=self.backend, device_id=self.device)
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self._initialized = True

    @property
    def is_main(self) -> bool:
        """Whether current process is rank 0."""
        return self.rank == 0

    @property
    def is_distributed(self) -> bool:
        """Whether distributed training is active."""
        return self._initialized

    def barrier(self):
        """Synchronize all processes."""
        if self._initialized:
            dist.barrier()

    def destroy(self):
        """Clean up distributed process group."""
        if self._initialized:
            dist.destroy_process_group()
            self._initialized = False
