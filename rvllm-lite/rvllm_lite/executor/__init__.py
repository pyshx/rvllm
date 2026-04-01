"""Executor -- mirrors rvllm-executor crate.

Dispatches batch generation to workers.
"""

from __future__ import annotations
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rvllm_lite.config import EngineConfig
    from rvllm_lite.sequence import SamplingParams

from rvllm_lite.worker import Worker

log = logging.getLogger("rvllm-lite.executor")


class Executor:
    """Single or multi-GPU executor dispatching to workers."""

    def __init__(self, config: "EngineConfig"):
        self.config = config
        self.workers: list[Worker] = []

    def init_workers(self):
        worker = Worker(self.config, rank=0, device_id=0)
        worker.init_model()
        self.workers = [worker]
        log.info(f"Executor: {len(self.workers)} worker(s) initialized")

    def generate(self, prompts: list[str], sampling_params: "SamplingParams") -> list:
        """Dispatch batch to driver worker."""
        return self.workers[0].generate(prompts, sampling_params)

    @property
    def tokenizer(self):
        return self.workers[0].tokenizer if self.workers else None
