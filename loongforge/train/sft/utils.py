# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""utils for sft"""

import logging
from collections import deque

from typing import TYPE_CHECKING, List, Optional, Union, Any, Type, Dict
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader
from transformers.utils import PaddingStrategy

from datasets.distributed import split_dataset_by_node

from megatron.core import mpu, tensor_parallel, parallel_state
from megatron.core.packed_seq_params import PackedSeqParams

from megatron.legacy.data.data_samplers import MegatronPretrainingRandomSampler

from loongforge.utils import get_args, get_tokenizer, constants
from loongforge.data import DataCollatorForSupervisedDataset
from loongforge.tokenizer import AutoTokenizerFromHF


if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset
logger = logging.getLogger(__name__)


######## utils for build dataset ########
def get_dataset_blend_from_list(
    dataset_names: Optional[List[str]],
) -> Optional[List[str]]:
    """get dataset from list"""
    if dataset_names is None:
        return None

    return [_dataset_name.strip() for _dataset_name in dataset_names]


def _cyclic_iter(iter):
    """cyclic iteration"""
    while True:
        for x in iter:
            yield x


def build_sft_data_collator(
    cls: Type[DataCollatorForSupervisedDataset], **kwargs
) -> DataCollatorForSupervisedDataset:
    """build data collator for sft"""
    args = get_args()
    tokenizer = get_tokenizer()

    assert isinstance(
        tokenizer, AutoTokenizerFromHF
    ), f"Only support HFTokenizer for sft, but got {args.tokenizer_type}."

    pad_to_multiple_of = 1
    # When using sequence parallel, sequence will further be split by TP size
    # When using context parallel, sequence is split by CP size as well
    pad_to_multiple_of *= (
        args.tensor_model_parallel_size if args.sequence_parallel else 1
    )
    pad_to_multiple_of *= (
        (2 * args.context_parallel_size) if args.context_parallel_size > 1 else 1
    )

    # https://github.com/NVIDIA/TransformerEngine/blob/v2.4/transformer_engine/pytorch/utils.py#L425
    # https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/common/gemm/cublaslt_gemm.cu#L151
    pad_to_multiple_of *= 128 if args.fp8 else 1

    padding = (
        PaddingStrategy.LONGEST
        if args.variable_seq_lengths
        else PaddingStrategy.MAX_LENGTH
    )

    # When chunkpipe is enabled, all chunks are already padded to chunksize,
    # so max_length should be chunksize instead of seq_length.
    max_length = args.chunksize if args.enable_chunkpipe else args.seq_length

    data_collator = cls(
        tokenizer=tokenizer.hf_tokenizer(),
        label_pad_token_id=constants.IGNORE_INDEX,
        pad_to_multiple_of=pad_to_multiple_of,
        padding=padding,
        max_length=max_length,
        **kwargs,
    )
    return data_collator


class _IterableWithState:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.step = 0
        self._iterator = iter(self.dataloader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self._iterator)
            self.step += 1
            return batch
        except StopIteration:
            self._iterator = iter(self.dataloader)
            # self.step = 0
            batch = next(self._iterator)
            self.step += 1
            return batch

    def save_state(self):
        """dataloader save state"""
        return {"step": self.step}

    def load_state(self, state):
        """dataloader load state"""
        target = state.get("step", 0)
        if target <= self.step:
            return
        for _ in range(target - self.step):
            next(self._iterator)
        self.step = target


class SavableCyclicIterator:
    """
    Cyclic iterator that:
      - exposes `.iterable` with save_state/load_state (via _IterableWithState)
      - yields batches infinitely
    Compatible with Megatron's maybe_save_dataloader_state().
    """

    def __init__(self, dataloader):
        self.iterable = _IterableWithState(dataloader)
        self._iterator = self._cyclic_iter(self.iterable)

    def _cyclic_iter(self, iterable_with_state):
        while True:
            for batch in iterable_with_state:
                yield batch

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._iterator)

    def save_state(self):
        """dataloader save state"""
        return self.iterable.save_state()

    def load_state(self, state):
        """dataloader load state"""
        return self.iterable.load_state(state)


class ChunkPipeGroupBatchSampler:
    """Batch sampler that shuffles chunk groups while preserving intra-group order
    and aligning chunk groups to training step boundaries.

    In chunkpipe SFT, a long sequence is split into multiple consecutive chunks
    that must be yielded in order. Each chunk carries a `chunk_group_size` field
    indicating how many consecutive chunks belong to the same source sequence
    (1 for binpacked short-sequence chunks).

    All chunks of a long sequence must fall within the same training step
    (same gradient-accumulation window), because KV cache is carried between
    chunks and would be invalidated by a gradient update.

    This sampler:
      1. Scans the dataset to identify groups (consecutive chunks with the same group size).
      2. Shuffles groups for training randomness.
      3. Shards groups across data-parallel ranks via LPT multiway partition
         (whole groups, never split), balancing total chunk count per rank so
         that all ranks produce nearly the same number of complete steps.
      4. Schedules groups into fixed-capacity step windows (capacity = num_microbatches
         x micro_batch_size chunks) using FFD (first-fit decreasing) to maximize
         packing density, ensuring no group is split across step boundaries.
      5. Aligns the per-rank step count to the cross-rank minimum, so every rank
         yields the same number of micro-batches per epoch (required for
         collective-sync correctness under DDP).
      6. Yields one micro-batch at a time, keeping group members consecutive.
    """

    def __init__(
        self,
        dataset,
        total_samples,
        consumed_samples,
        micro_batch_size,
        data_parallel_rank,
        data_parallel_size,
        num_microbatches,
        seed=0,
    ):
        self.total_samples = total_samples
        self.consumed_samples = consumed_samples
        self.micro_batch_size = micro_batch_size
        self.data_parallel_rank = data_parallel_rank
        self.data_parallel_size = data_parallel_size
        self.num_microbatches = num_microbatches
        self.seed = seed

        # Step capacity in chunks: each step has num_microbatches micro-batches,
        # each micro-batch holds micro_batch_size chunks.
        self.step_capacity = num_microbatches * micro_batch_size

        # Build groups by scanning chunk_group_size
        self.groups = []
        idx = 0
        group_sizes = dataset["chunk_group_size"]
        max_group_size = 0
        while idx < total_samples:
            size = group_sizes[idx]
            self.groups.append(list(range(idx, idx + size)))
            max_group_size = max(max_group_size, size)
            idx += size

        assert max_group_size <= self.step_capacity, (
            f"Max chunk_group_size ({max_group_size}) exceeds step capacity "
            f"({self.step_capacity} = num_microbatches {num_microbatches} "
            f"x micro_batch_size {micro_batch_size}). "
            f"Increase global_batch_size or decrease seq_length/chunksize ratio."
        )

        # Pre-compute usable samples per epoch for epoch/resume calculation.
        # Use an unshuffled LPT partition + cross-rank min alignment as the
        # stable baseline. The shuffled per-epoch schedule in __iter__ may
        # yield slightly different step counts, but epoch/resume accounting
        # needs a deterministic, shuffle-independent estimate shared by all
        # ranks so that `_epoch = consumed_samples // usable_total` and the
        # shuffle seed `seed + _epoch` agree across ranks.
        dummy_buckets = self._lpt_partition(range(len(self.groups)))
        _, _, dummy_aligned = self._schedule_buckets(dummy_buckets)
        self.usable_per_rank = dummy_aligned * self.step_capacity
        self.usable_total = self.usable_per_rank * self.data_parallel_size

        # Determine initial epoch and within-epoch offset for checkpoint resume
        if self.usable_total > 0:
            self._epoch = consumed_samples // self.usable_total
            self._resume_offset = (consumed_samples % self.usable_total) // self.data_parallel_size
            self._resume_offset = (self._resume_offset // self.step_capacity) * self.step_capacity
        else:
            self._epoch = 0
            self._resume_offset = 0

        # Per-step G (number of source sequences in the step) queue. Populated
        # yield-time in __iter__, consumed by get_batch via TP rank-0 popleft +
        # broadcast. The deque object itself is created once and never cleared,
        # so downstream references (saved in args.chunkpipe_step_g_queue) stay
        # valid across epochs. FIFO alignment with the DataLoader's actual
        # batch production is guaranteed because sampler yield and get_batch
        # consumption both run in the main process in strict order.
        self._step_g_queue = deque()

    def __len__(self):
        return self.total_samples

    def _lpt_partition(self, group_indices):
        """LPT (Longest Processing Time) multiway partition into DP buckets.

        Distributes the given group indices across `data_parallel_size` buckets
        such that the total chunk count per bucket is balanced. Largest groups
        are assigned first to the bucket with the smallest current load; ties
        are broken by bucket id for determinism. LPT guarantees
        `max_load - min_load <= max(group_size) <= step_capacity`, i.e. the
        cross-rank imbalance is bounded by one step window.

        All ranks run this locally and obtain the same assignment because the
        inputs (`self.groups` and `group_indices`) and the algorithm are
        deterministic — no collective communication is required.

        Args:
            group_indices: iterable of indices into self.groups.

        Returns:
            List[List[int]]: length `data_parallel_size`; `buckets[r]` is the
            list of group indices assigned to rank r.
        """
        buckets = [[] for _ in range(self.data_parallel_size)]
        loads = [0] * self.data_parallel_size
        sorted_indices = sorted(
            group_indices,
            key=lambda gi: (-len(self.groups[gi]), gi),
        )
        for gi in sorted_indices:
            r = min(range(self.data_parallel_size), key=lambda r: (loads[r], r))
            buckets[r].append(gi)
            loads[r] += len(self.groups[gi])
        return buckets

    def _schedule_buckets(self, buckets):
        """Schedule all buckets and cross-rank-align their step counts.

        Runs `_schedule_step_aligned` independently on every bucket (locally,
        all ranks compute the same results) and truncates the current rank's
        schedule to the minimum step count across all buckets. This guarantees
        every rank yields the same number of complete steps per epoch, which
        is required for DDP collectives to stay in lock-step.

        Args:
            buckets: output of `_lpt_partition`.

        Returns:
            Tuple (my_steps, my_step_gs, aligned_count):
              my_steps: List[List[int]] of complete steps for the current rank,
                        already truncated to aligned_count.
              my_step_gs: List[int], number of source groups placed in each
                          step of my_steps (aligned with my_steps).
              aligned_count: int, the common step count across all ranks.
        """
        step_counts = []
        my_steps = None
        my_step_gs = None
        for r, bucket in enumerate(buckets):
            steps_r, step_gs_r = self._schedule_step_aligned(
                [self.groups[gi] for gi in bucket]
            )
            step_counts.append(len(steps_r))
            if r == self.data_parallel_rank:
                my_steps = steps_r
                my_step_gs = step_gs_r
        aligned_count = min(step_counts) if step_counts else 0
        assert aligned_count > 0, (
            f"ChunkPipe sampler: 0 complete steps per epoch. "
            f"total_chunks={len(self.groups)}, step_capacity={self.step_capacity}, DP={self.data_parallel_size}.\n"
            f"Possible causes:\n"
            f"  1. Dataset too small (need at least {self.step_capacity * self.data_parallel_size} chunks total)\n"
            f"  2. --global-batch-size too large (reduces step_capacity={self.step_capacity})\n"
            f"Suggestion: add more data into dataset or reduce --global-batch-size."
        )
        return my_steps[:aligned_count], my_step_gs[:aligned_count], aligned_count

    def _schedule_step_aligned(self, rank_groups):
        """Arrange groups into a list of complete step windows.

        For each step window of `step_capacity` chunks:
          1. Greedily place multi-chunk groups (long sequences) that fit, in
             size-descending order (FFD).
          2. Fill remaining slots with single-chunk groups (binpacked short sequences).

        Under-filled steps (when no single-groups remain and no deferred
        multi-group fits the current vacancy) are dropped but scheduling
        continues — deferred multi-groups may still combine into complete
        windows in subsequent iterations. This preserves the invariant that
        every emitted step contains only whole groups and is exactly
        `step_capacity` chunks long, so downstream flattening + fixed-size
        micro-batching never splits a chunk group across a step boundary.

        Args:
            rank_groups: list of groups (each group is a list of dataset indices)
                         assigned to this DP rank.

        Returns:
            Tuple (steps, step_gs):
              steps: List[List[int]] — each inner list is exactly
                `step_capacity` dataset indices, representing one complete
                training step.
              step_gs: List[int] —               step_gs[i] is the number of source groups
                placed in steps[i] (used as G for the per-sample loss path).
        """
        # FFD (first-fit decreasing): sort multi-chunk groups by size descending
        # so that large groups are placed first and small groups act as "glue"
        # to fill remaining capacity. This significantly improves packing
        # density compared to shuffle-order placement, reducing under-filled
        # steps and the amount of data dropped per epoch. Within-step order of
        # groups does not affect training correctness (gradients are accumulated
        # across all chunks in the step).
        multi_groups = deque(sorted(
            (g for g in rank_groups if len(g) > 1),
            key=len, reverse=True,
        ))
        single_groups = deque(g for g in rank_groups if len(g) == 1)

        steps = []
        step_gs = []

        while multi_groups or single_groups:
            remaining = self.step_capacity
            step_indices = []
            step_group_count = 0

            # Phase 1: greedily place multi-chunk groups
            deferred = deque()
            while multi_groups:
                group = multi_groups.popleft()
                if len(group) <= remaining:
                    step_indices.extend(group)
                    remaining -= len(group)
                    step_group_count += 1
                else:
                    deferred.append(group)
            # Put back groups that didn't fit for future steps
            multi_groups = deferred

            # Phase 2: fill remaining slots with single-chunk groups
            while single_groups and remaining > 0:
                step_indices.extend(single_groups.popleft())
                remaining -= 1
                step_group_count += 1

            if len(step_indices) < self.step_capacity:
                # Under-filled step: no single-groups left to fill gaps and the
                # groups placed here plus the deferred ones could not combine
                # into a full window. Drop this step but keep scheduling —
                # remaining deferred multi-groups may still combine into
                # complete windows in subsequent iterations (e.g. two deferred
                # groups of size 5+3 can fit a fresh 8-wide step together even
                # though neither fit alongside a previously-placed 7).
                #
                # Termination: the init-time assertion max_group_size <=
                # step_capacity guarantees Phase 1 always places at least one
                # group when multi_groups is non-empty, so every outer
                # iteration consumes at least one group and the loop is
                # bounded by len(rank_groups).
                continue

            steps.append(step_indices)
            step_gs.append(step_group_count)

        return steps, step_gs

    def __iter__(self):
        total_groups = len(self.groups)

        # Shuffle groups deterministically — different seed per epoch
        g = torch.Generator()
        g.manual_seed(self.seed + self._epoch)
        group_order = torch.randperm(total_groups, generator=g).tolist()

        # Shard groups across DP ranks via LPT multiway partition (balanced by
        # chunk count, not group count). All ranks run the same deterministic
        # partition locally, so each rank knows exactly which groups it owns
        # without any collective communication.
        buckets = self._lpt_partition(group_order)

        # Schedule every bucket and truncate the current rank's schedule to
        # the cross-rank minimum step count. This guarantees every rank yields
        # the same number of micro-batches per epoch, preventing DDP collective
        # desync when one rank's iterator exhausts before others.
        my_steps, my_step_gs, _ = self._schedule_buckets(buckets)

        # Resume skip in micro-batch granularity. _resume_offset was already
        # rounded down to step_capacity boundary in __init__, and
        # step_capacity % micro_batch_size == 0 holds by construction, so
        # skip_mbs lands on a step boundary.
        skip_mbs = self._resume_offset // self.micro_batch_size
        self._resume_offset = 0  # only skip once

        # Yield one micro-batch at a time. Every micro-batch within a step
        # shares the same G = number of source groups in that step. We append
        # G to _step_g_queue immediately before yield (never clear it) so that
        # get_batch's popleft order matches yield order strictly — robust to
        # DataLoader prefetching and epoch boundaries. Skipped micro-batches
        # (via resume) do NOT enter the queue, preserving FIFO alignment.
        mb_idx = 0
        for step, G in zip(my_steps, my_step_gs):
            for mb_start in range(0, len(step), self.micro_batch_size):
                if mb_idx < skip_mbs:
                    mb_idx += 1
                    continue
                mb_idx += 1
                batch = step[mb_start:mb_start + self.micro_batch_size]
                self._step_g_queue.append(G)
                self.consumed_samples += self.micro_batch_size * self.data_parallel_size
                yield batch

        # Epoch complete — next __iter__ call will use a different shuffle
        self._epoch += 1


def _build_cylic_iterator(
    dataset: Union["Dataset", "IterableDataset"],
    consumed_samples: int,
    data_collator: DataCollatorForSupervisedDataset,
):
    """build data iterator for sft"""
    if dataset is None:
        return None

    args = get_args()

    _dataloader_kwargs = {}
    if args.sft_data_streaming:
        # split distributed dataset for streaming
        dataset = split_dataset_by_node(
            dataset=dataset,
            rank=mpu.get_data_parallel_rank(),
            world_size=mpu.get_data_parallel_world_size(),
        )

        dataset = dataset.shuffle(
            buffer_size=args.streaming_buffer_size,
            seed=args.seed,
        )

        _dataloader_kwargs = dict(
            batch_size=args.micro_batch_size,
        )
    else:
        # build distribued sampler for non-streaming dataset
        if args.enable_chunkpipe:
            num_microbatches = args.global_batch_size // (
                args.micro_batch_size * mpu.get_data_parallel_world_size()
            )
            _batch_sampler = ChunkPipeGroupBatchSampler(
                dataset,
                total_samples=len(dataset),
                consumed_samples=consumed_samples,
                micro_batch_size=args.micro_batch_size,
                data_parallel_rank=mpu.get_data_parallel_rank(),
                data_parallel_size=mpu.get_data_parallel_world_size(),
                num_microbatches=num_microbatches,
                seed=args.seed,
            )
            # Expose the per-step G FIFO queue to get_batch. Only the deque
            # reference is shared; the sampler object itself stays private.
            # The deque is created once in __init__ and never replaced, so
            # this reference remains valid across epochs.

            # However, when VPP is enabled, the chunkpipe_step_g_queue of the following vp_stage will overwrite the
            # previous one, and the final args.chunkpipe_step_g_queue is a reference to the last vp_stage dataset,
            # because all vp_stage share the same args
            args.chunkpipe_step_g_queue = _batch_sampler._step_g_queue
        else:
            _batch_sampler = MegatronPretrainingRandomSampler(
            dataset,
            total_samples=len(dataset),
            consumed_samples=consumed_samples,  # not support for streaming now!
            micro_batch_size=args.micro_batch_size,
            data_parallel_rank=mpu.get_data_parallel_rank(),
            data_parallel_size=mpu.get_data_parallel_world_size(),
            data_sharding=args.data_sharding,
        )

        _dataloader_kwargs = dict(
            batch_sampler=_batch_sampler,
            persistent_workers=True if args.num_workers > 0 else False,
        )

    dataloader = DataLoader(
        dataset,
        collate_fn=data_collator,
        num_workers=args.num_workers,
        pin_memory=True,
        **_dataloader_kwargs,
    )

    if args.dataloader_save is not None:
        return SavableCyclicIterator(dataloader)
    else:
        return iter(_cyclic_iter(dataloader))


def build_sft_cyclic_iterators(
    train_ds: Optional[Union["Dataset", "IterableDataset"]],
    valid_ds: Optional[Union["Dataset", "IterableDataset"]],
    test_ds: Optional[Union["Dataset", "IterableDataset"]],
    data_collator: Optional[DataCollatorForSupervisedDataset],
):
    """build data iterators for sft"""
    args = get_args()
    train_iter = _build_cylic_iterator(
        train_ds, args.consumed_train_samples, data_collator
    )
    valid_iter = _build_cylic_iterator(
        valid_ds, 0 if args.skip_train else args.consumed_valid_samples, data_collator
    )
    test_iter = _build_cylic_iterator(test_ds, 0, data_collator)
    return train_iter, valid_iter, test_iter


######## utils for get_batch ########
def _get_position_ids(data: torch.Tensor):
    """create position ids"""
    current_device = data.device
    _, seq_length = data.shape

    position_ids = torch.arange(seq_length, dtype=torch.long, device=current_device)
    position_ids = position_ids.unsqueeze(0).expand_as(data)
    return position_ids


def _get_attention_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """create attention mask"""
    args = get_args()
    current_device = attention_mask.device
    batch_size, seq_length = attention_mask.shape

    # Only used attn_mask when attn_mask_type in [padding, padding_causal, arbitrary] in TE
    # TODO: for multi-acceleator, maybe we should update attn_mask_type and attention_mask shape

    if args.context_parallel_size > 1:
        # Firstly, context parallel only support causal mask in TE now.
        # Secondly, when context-parallel is enabled, the input data is of a relatively long length,
        # and micro-batch-size does not need to be increased, nor padding occurs
        # create causal mask here, shape [B, 1, S, S].
        attention_mask = torch.tril(
            torch.ones(
                (batch_size, seq_length, seq_length),
                dtype=torch.long,
                device=current_device,
            )
        )
        attention_mask.unsqueeze_(1)
        attention_mask = (attention_mask < 0.5).bool()
    else:
        # create mask for te, shape [B, 1, 1, S]. attn_mask_type is padding_causal or causal.
        attention_mask.unsqueeze_(1).unsqueeze_(1)
        attention_mask = (attention_mask < 0.5).bool()

    return attention_mask


def _get_packed_sequence_params(attention_mask: torch.Tensor) -> PackedSeqParams:
    """create packed sequence params"""
    # assume micro_batch_size == 1
    assert attention_mask.shape[0] == 1, "attention_mask should be of shape [1, S]"

    packed_seq_params = PackedSeqParams()
    packed_seq_params.qkv_format = "thd"

    # calculate cu_seqlens_q
    # example: mask = [[1, 1, 2, 2, 2, 3, 3, 4, 5, 5, 5, 0, 0]]
    # expacted cu_seqlens_q = [0, 2, 5, 7, 8, 11, 13]
    max_num = attention_mask.max().item()
    reduced_mask = torch.bincount(attention_mask.view(-1), minlength=max_num + 1)
    reduced_mask = reduced_mask[1:].to(dtype=torch.int32, device=attention_mask.device)

    cu_seqlens = reduced_mask.cumsum(dim=0).to(torch.int32)
    zero = torch.zeros(1, dtype=torch.int32, device=attention_mask.device)
    # The lengths of padding tokens must also be taken into account in cu_seqlens;
    # otherwise, the attention calculation will be incorrect.
    cu_seqlens[-1] = attention_mask.shape[1]
    cu_seqlens = torch.cat((zero, cu_seqlens))

    packed_seq_params.cu_seqlens_q = cu_seqlens
    packed_seq_params.cu_seqlens_kv = cu_seqlens  # just for self-attention
    packed_seq_params.max_seqlen_q = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
    packed_seq_params.max_seqlen_kv = packed_seq_params.max_seqlen_q

    return packed_seq_params


def get_batch_on_this_tp_rank(data_iterator):
    """get batch on this tp rank"""
    args = get_args()
    tokenizer = get_tokenizer()

    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None

    # broadcast required keys across tp
    required_keys = ["attention_mask"]
    if args.enable_chunkpipe:
        required_keys.append("chunk_group_size")
        required_keys.append("group_total_tokens")

    if args.pipeline_model_parallel_size == 1:
        required_keys += ["input_ids", "labels"] + (
            ["loss_mask"] if not args.eod_mask_loss else []
        )

    elif mpu.is_pipeline_first_stage():
        required_keys.append("input_ids")

    elif mpu.is_pipeline_last_stage():
        required_keys += ["input_ids", "labels"] + (
            ["loss_mask"] if not args.eod_mask_loss else []
        )

    data_b = tensor_parallel.broadcast_data(required_keys, data, torch.int64)

    # tokens & position ids
    tokens = data_b["input_ids"].long() if "input_ids" in data_b else None
    position_ids = None
    if tokens is not None:
        position_ids = _get_position_ids(tokens)

    # labels & loss mask
    labels = data_b["labels"].long() if "labels" in data_b else None
    if labels is not None:
        if not args.enable_chunkpipe:
            # Shift labels for next-token prediction; chunkpipe data is already pre-shifted
            labels = torch.roll(labels, shifts=-1, dims=1)
            labels[:, -1] = constants.IGNORE_INDEX
        # labels[labels == tokenizer.pad] == constants.IGNORE_INDEX
        # labels[labels == tokenizer.eos] == constants.IGNORE_INDEX

    # create loss mask
    loss_mask = data_b["loss_mask"].long() if "loss_mask" in data_b else None
    if loss_mask is not None:
        if not args.enable_chunkpipe:
            # pp last && not eod_mask_loss; chunkpipe data is already pre-shifted
            loss_mask = torch.roll(loss_mask, shifts=-1, dims=1)
            loss_mask[:, -1] = 0

    elif labels is not None:
        # pp last && eod_mask_loss
        assert args.eod_mask_loss, "eod_mask_loss should be true here!"
        loss_mask = torch.ones(labels.size(), dtype=torch.float, device=labels.device)
        loss_mask[labels == constants.IGNORE_INDEX] = 0.0
        loss_mask[labels == tokenizer.pad] = 0.0
        loss_mask[labels == tokenizer.eos] = 0.0

    # attention mask
    attention_mask = None
    packed_seq_params = None

    if not args.packing_sft_data:
        attention_mask = _get_attention_mask(
            data_b["attention_mask"].long()
        )
    else:
        # attention_mask will be ignored in te
        packed_seq_params = _get_packed_sequence_params(
            data_b["attention_mask"].long()
        )

    batch = {
        "tokens": tokens,
        "labels": labels,
        "loss_mask": loss_mask,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "packed_seq_params": packed_seq_params,
    }
    if args.enable_chunkpipe and "chunk_group_size" in data_b:
        batch["chunk_group_size"] = data_b["chunk_group_size"]
        batch["group_total_tokens"] = data_b["group_total_tokens"]

        # Per-step G (source sequence count). Unlike per-chunk fields, G is not
        # carried through the dataset/collator path; it's produced by the
        # sampler yield-time and delivered via a FIFO deque on args. TP rank 0
        # pops the queue, other TP ranks receive the value via broadcast.
        #
        # VPP mode: get_batch is called multiple times per step (once per VP stage).
        # To avoid popping the queue multiple times, only pop on the first VP stage.
        vp_size = mpu.get_virtual_pipeline_model_parallel_world_size()
        vp_stage = mpu.get_virtual_pipeline_model_parallel_rank()
        tp_rank = mpu.get_tensor_model_parallel_rank()
        is_vpp_enabled = vp_size is not None and vp_size > 1

        # Determine if this rank should pop the queue:
        # - Non-VPP: only TP rank 0 pops
        # - VPP: only TP rank 0 AND VP last stage pops，because args.chunkpipe_step_g_queue is last vp_stage,
        #           and only last vp_stage will have loss_func calculations
        should_pop = (tp_rank == 0) and ((not is_vpp_enabled) or (vp_stage == (vp_size - 1)))

        if should_pop:
            step_num_groups = args.chunkpipe_step_g_queue.popleft()
        else:
            step_num_groups = 0
        g_tensor = torch.tensor(
            [step_num_groups],
            dtype=torch.long,
            device=torch.cuda.current_device(),
        )
        torch.distributed.broadcast(
            g_tensor,
            mpu.get_tensor_model_parallel_src_rank(),
            group=mpu.get_tensor_model_parallel_group(),
        )
        batch["step_num_groups"] = g_tensor

    return batch


def get_batch_on_this_cp_rank(batch: Dict[str, Any]):
    """Slice batch input along sequence dimension into multiple chunks,
    which are parallelized across GPUs in a context parallel group.
    """
    cp_size = parallel_state.get_context_parallel_world_size()
    if cp_size > 1:
        packed_seq_params = batch.get('packed_seq_params', None)
        cp_rank = parallel_state.get_context_parallel_rank()
        for key, val in batch.items():
            if val is not None:
                if key == 'packed_seq_params':
                    batch[key] = val
                    continue
          
                seq_dim = 1 if key != 'attention_mask' else 2
                if packed_seq_params is not None and packed_seq_params.qkv_format == 'thd':
                    #assert get_accelerator_backend() == "NvidiaGpu", "Only NvidiaGPU supports packed_seq_params."
                    import transformer_engine_torch as tex
                    # assume cu_seqlens_q == cu_seqlens_kv
                    cu_seqlens_q = packed_seq_params.cu_seqlens_q
                    seq_idx_val = tex.thd_get_partitioned_indices(
                        cu_seqlens_q, val.shape[seq_dim], cp_size, cp_rank
                    )
                    batch[key] = val.index_select(seq_dim, seq_idx_val)
                else:
                    val = val.view(
                        *val.shape[0:seq_dim],
                        2 * cp_size,
                        val.shape[seq_dim] // (2 * cp_size),
                        *val.shape[(seq_dim + 1) :],
                    )
                    index = torch.tensor(
                        [cp_rank, (2 * cp_size - cp_rank - 1)], device="cpu", pin_memory=True
                    ).cuda(non_blocking=True)
                    val = val.index_select(seq_dim, index)
                    val = val.view(*val.shape[0:seq_dim], -1, *val.shape[(seq_dim + 2) :])
                    batch[key] = val

    return batch