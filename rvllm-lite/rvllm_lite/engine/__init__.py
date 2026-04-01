"""Async GPU engine -- mirrors rvllm-engine crate.

Key design from rvLLM: GPU work runs on a dedicated OS thread,
not an asyncio task. This keeps the HTTP event loop responsive
while the GPU thread processes batches.
"""

from __future__ import annotations
import asyncio
import logging
import threading
import time
import uuid
from dataclasses import dataclass
from typing import AsyncIterator

from rvllm_lite.config import EngineConfig
from rvllm_lite.executor import Executor
from rvllm_lite.sequence import RequestOutput, SamplingParams

log = logging.getLogger("rvllm-lite.engine")


@dataclass
class _PendingRequest:
    request_id: str
    prompt: str
    sampling_params: SamplingParams


class AsyncGpuLLMEngine:
    """rvLLM's dedicated-GPU-thread engine pattern.

    - Async interface for the API layer
    - Dedicated OS thread for GPU batch execution
    - Requests accumulate, get batched, results fanned out
    """

    def __init__(self, config: EngineConfig):
        self.config = config
        self.executor = Executor(config)

        # Request queue: async API -> GPU thread
        self._pending: list[_PendingRequest] = []
        self._pending_lock = threading.Lock()

        # Output channels: GPU thread -> async API
        self._output_queues: dict[str, asyncio.Queue[RequestOutput]] = {}
        self._loop: asyncio.AbstractEventLoop | None = None

        # GPU thread control
        self._gpu_thread: threading.Thread | None = None
        self._running = False

        # Batching config
        self._batch_timeout_ms = 5  # collect requests for up to 5ms before batching
        self._max_batch_size = config.scheduler.max_num_seqs

    async def start(self):
        """Initialize model and start GPU thread."""
        log.info("Engine: initializing executor...")
        self.executor.init_workers()
        self._loop = asyncio.get_running_loop()
        self._running = True
        self._gpu_thread = threading.Thread(target=self._gpu_loop, daemon=True, name="rvllm-gpu")
        self._gpu_thread.start()
        log.info("Engine: GPU thread started")

    def _gpu_loop(self):
        """Dedicated GPU thread -- collects and batches requests."""
        while self._running:
            # Drain pending requests
            with self._pending_lock:
                batch = list(self._pending)
                self._pending.clear()

            if not batch:
                time.sleep(0.001)
                continue

            # Cap batch size
            if len(batch) > self._max_batch_size:
                overflow = batch[self._max_batch_size:]
                batch = batch[:self._max_batch_size]
                with self._pending_lock:
                    self._pending.extend(overflow)

            self._execute_batch(batch)

    def _execute_batch(self, batch: list[_PendingRequest]):
        """Execute a batch of requests via vLLM."""
        # Group by sampling params (vLLM needs uniform params per batch)
        # For simplicity, process each request individually if params differ
        # TODO: group by compatible params for true batching
        for req in batch:
            try:
                results = self.executor.generate([req.prompt], req.sampling_params)
                if results:
                    result = results[0]
                    output_text = result.outputs[0].text
                    token_ids = list(result.outputs[0].token_ids)
                    output = RequestOutput(
                        request_id=req.request_id,
                        token_ids=token_ids,
                        text=output_text,
                        finished=True,
                    )
                    self._emit_output(req.request_id, output)
            except Exception as e:
                log.error(f"Generation failed for {req.request_id}: {e}")
                output = RequestOutput(
                    request_id=req.request_id,
                    token_ids=[],
                    text="",
                    finished=True,
                )
                self._emit_output(req.request_id, output)

    def _emit_output(self, request_id: str, output: RequestOutput):
        if request_id in self._output_queues and self._loop:
            self._loop.call_soon_threadsafe(
                self._output_queues[request_id].put_nowait, output
            )

    async def generate(
        self, prompt: str, sampling_params: SamplingParams
    ) -> AsyncIterator[RequestOutput]:
        """Add a request and stream outputs."""
        request_id = str(uuid.uuid4())

        queue: asyncio.Queue[RequestOutput] = asyncio.Queue()
        self._output_queues[request_id] = queue

        req = _PendingRequest(
            request_id=request_id,
            prompt=prompt,
            sampling_params=sampling_params,
        )
        with self._pending_lock:
            self._pending.append(req)

        try:
            while True:
                output = await queue.get()
                yield output
                if output.finished:
                    break
        finally:
            self._output_queues.pop(request_id, None)

    async def shutdown(self):
        self._running = False
        if self._gpu_thread:
            self._gpu_thread.join(timeout=5.0)
        log.info("Engine: shut down")
