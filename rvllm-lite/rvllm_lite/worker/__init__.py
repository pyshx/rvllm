"""Per-GPU worker -- mirrors rvllm-worker crate.

Uses vLLM's LLM as the execution backend (torch.compile, FA3, CUDA graphs).
Our serving stack controls request lifecycle, batching, and streaming.
"""

from __future__ import annotations
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rvllm_lite.config import EngineConfig

log = logging.getLogger("rvllm-lite.worker")


class Worker:
    """Single-GPU worker wrapping vLLM's LLM execution engine."""

    def __init__(self, config: "EngineConfig", rank: int = 0, device_id: int = 0):
        self.config = config
        self.rank = rank
        self.device_id = device_id
        self._llm = None

    def init_model(self):
        """Load model using vLLM (torch.compile + FA3 + CUDA graphs)."""
        from vllm import LLM

        log.info(f"Worker {self.rank}: loading {self.config.model.model} on GPU {self.device_id}")
        self._llm = LLM(
            model=self.config.model.model,
            dtype=self.config.model.dtype,
            max_model_len=self.config.model.max_model_len,
            tensor_parallel_size=self.config.parallel.tensor_parallel_size,
            gpu_memory_utilization=self.config.cache.gpu_memory_utilization,
            enforce_eager=self.config.model.enforce_eager,
            disable_log_stats=True,
        )
        log.info(f"Worker {self.rank}: model ready")

    def generate(self, prompts: list[str], sampling_params) -> list:
        """Run batch generation via vLLM's compiled forward pass."""
        from vllm import SamplingParams as VllmSamplingParams

        vllm_params = VllmSamplingParams(
            temperature=sampling_params.temperature,
            top_p=sampling_params.top_p,
            top_k=sampling_params.top_k if sampling_params.top_k > 0 else -1,
            max_tokens=sampling_params.max_tokens,
            stop=sampling_params.stop if sampling_params.stop else None,
            repetition_penalty=sampling_params.repetition_penalty,
        )
        return self._llm.generate(prompts, vllm_params)

    @property
    def tokenizer(self):
        if self._llm:
            return self._llm.get_tokenizer()
        return None
