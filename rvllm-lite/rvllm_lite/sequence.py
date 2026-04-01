"""Sequence and request types -- mirrors rvllm-sequence crate."""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
import time


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    SWAPPED = auto()
    FINISHED_STOPPED = auto()
    FINISHED_LENGTH = auto()
    FINISHED_ABORTED = auto()

    @property
    def is_finished(self) -> bool:
        return self in (
            SequenceStatus.FINISHED_STOPPED,
            SequenceStatus.FINISHED_LENGTH,
            SequenceStatus.FINISHED_ABORTED,
        )


@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    max_tokens: int = 256
    stop: list[str] = field(default_factory=list)
    repetition_penalty: float = 1.0


@dataclass
class Sequence:
    seq_id: int
    prompt_token_ids: list[int]
    output_token_ids: list[int] = field(default_factory=list)
    status: SequenceStatus = SequenceStatus.WAITING
    block_table: list[int] = field(default_factory=list)

    @property
    def num_tokens(self) -> int:
        return len(self.prompt_token_ids) + len(self.output_token_ids)

    @property
    def last_token_id(self) -> int:
        if self.output_token_ids:
            return self.output_token_ids[-1]
        return self.prompt_token_ids[-1]

    def append_token(self, token_id: int):
        self.output_token_ids.append(token_id)


@dataclass
class SequenceGroup:
    request_id: str
    seqs: list[Sequence]
    sampling_params: SamplingParams
    arrival_time: float = field(default_factory=time.monotonic)
    prompt_tokens_processed: int = 0

    @property
    def is_prefill(self) -> bool:
        return self.prompt_tokens_processed < len(self.seqs[0].prompt_token_ids)

    @property
    def is_finished(self) -> bool:
        return all(s.status.is_finished for s in self.seqs)


@dataclass
class RequestOutput:
    request_id: str
    token_ids: list[int]
    text: str
    finished: bool
    logprobs: list[float] | None = None
