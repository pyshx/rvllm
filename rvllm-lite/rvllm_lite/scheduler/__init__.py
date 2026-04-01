"""Continuous batching scheduler -- mirrors rvllm-scheduler crate.

Manages three queues (waiting, running, swapped) and decides which
sequence groups to run each step. Supports chunked prefill and
preemption via recompute or swap.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import logging

from rvllm_lite.config import SchedulerConfig
from rvllm_lite.sequence import SequenceGroup, SequenceStatus

log = logging.getLogger("rvllm-lite.scheduler")


@dataclass
class SchedulerOutputs:
    scheduled_seq_groups: list[SequenceGroup]
    num_batched_tokens: int
    preempted: list[SequenceGroup] = field(default_factory=list)
    blocks_to_swap_in: dict[int, int] = field(default_factory=dict)
    blocks_to_swap_out: dict[int, int] = field(default_factory=dict)
    blocks_to_copy: dict[int, int] = field(default_factory=dict)


class Scheduler:
    def __init__(self, config: SchedulerConfig, block_manager=None):
        self.config = config
        self.block_manager = block_manager
        self.waiting: list[SequenceGroup] = []
        self.running: list[SequenceGroup] = []
        self.swapped: list[SequenceGroup] = []

    def add_seq_group(self, seq_group: SequenceGroup):
        self.waiting.append(seq_group)

    def abort_seq_group(self, request_id: str):
        for queue in (self.waiting, self.running, self.swapped):
            for sg in queue:
                if sg.request_id == request_id:
                    for seq in sg.seqs:
                        seq.status = SequenceStatus.FINISHED_ABORTED
                    queue.remove(sg)
                    return

    def has_unfinished_seqs(self) -> bool:
        return bool(self.waiting or self.running or self.swapped)

    def schedule(self) -> SchedulerOutputs:
        scheduled: list[SequenceGroup] = []
        num_batched_tokens = 0
        max_tokens = self.config.max_num_batched_tokens or 8192
        max_seqs = self.config.max_num_seqs

        # 1. Continue running sequences (decode, T=1 each)
        still_running = []
        for sg in self.running:
            if sg.is_finished:
                continue
            if len(scheduled) >= max_seqs:
                break
            scheduled.append(sg)
            num_batched_tokens += 1  # decode = 1 token per seq
            still_running.append(sg)
        self.running = still_running

        # 2. Admit waiting sequences (prefill)
        still_waiting = []
        for sg in self.waiting:
            if len(scheduled) >= max_seqs:
                still_waiting.append(sg)
                continue

            prompt_len = len(sg.seqs[0].prompt_token_ids) - sg.prompt_tokens_processed
            if self.config.max_prefill_chunk:
                prompt_len = min(prompt_len, self.config.max_prefill_chunk)

            if num_batched_tokens + prompt_len > max_tokens:
                still_waiting.append(sg)
                continue

            # Admit this sequence group
            for seq in sg.seqs:
                seq.status = SequenceStatus.RUNNING
            sg.prompt_tokens_processed += prompt_len
            scheduled.append(sg)
            num_batched_tokens += prompt_len
            self.running.append(sg)

        self.waiting = still_waiting

        return SchedulerOutputs(
            scheduled_seq_groups=scheduled,
            num_batched_tokens=num_batched_tokens,
        )
