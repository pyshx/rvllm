"""OpenAI-compatible API server -- mirrors rvllm-api crate."""

from __future__ import annotations
import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from rvllm_lite.config import EngineConfig
from rvllm_lite.engine import AsyncGpuLLMEngine
from rvllm_lite.sequence import SamplingParams

log = logging.getLogger("rvllm-lite.api")


def create_app(args) -> FastAPI:
    config = EngineConfig.from_args(args)
    engine = AsyncGpuLLMEngine(config)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await engine.start()
        yield
        await engine.shutdown()

    app = FastAPI(title="rvLLM-lite", lifespan=lifespan)
    app.state.engine = engine
    app.state.model_name = args.model

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    async def list_models():
        return {
            "object": "list",
            "data": [
                {
                    "id": app.state.model_name,
                    "object": "model",
                    "owned_by": "rvllm-lite",
                }
            ],
        }

    @app.post("/v1/completions")
    async def completions(request: Request):
        body = await request.json()
        prompt = body.get("prompt", "")
        stream = body.get("stream", False)

        params = SamplingParams(
            temperature=body.get("temperature", 1.0),
            top_p=body.get("top_p", 1.0),
            top_k=body.get("top_k", -1),
            max_tokens=body.get("max_tokens", 256),
            stop=body.get("stop", []),
        )

        if stream:
            return StreamingResponse(
                _stream_completions(engine, prompt, params, app.state.model_name),
                media_type="text/event-stream",
            )

        # Non-streaming: collect all tokens
        text = ""
        token_ids = []
        async for output in engine.generate(prompt, params):
            text = output.text
            token_ids = output.token_ids

        return {
            "id": f"cmpl-{uuid.uuid4().hex[:8]}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": app.state.model_name,
            "choices": [{"text": text, "index": 0, "finish_reason": "stop"}],
            "usage": {"completion_tokens": len(token_ids)},
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = await request.json()
        messages = body.get("messages", [])
        stream = body.get("stream", False)

        # Build prompt from messages (simple concatenation for now)
        prompt = _messages_to_prompt(messages)

        params = SamplingParams(
            temperature=body.get("temperature", 1.0),
            top_p=body.get("top_p", 1.0),
            max_tokens=body.get("max_tokens", 256),
            stop=body.get("stop", []),
        )

        if stream:
            return StreamingResponse(
                _stream_chat(engine, prompt, params, app.state.model_name),
                media_type="text/event-stream",
            )

        text = ""
        async for output in engine.generate(prompt, params):
            text = output.text

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": app.state.model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
        }

    return app


def _messages_to_prompt(messages: list[dict]) -> str:
    """Simple chat template. Real impl should use tokenizer's chat template."""
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        parts.append(f"<|{role}|>\n{content}")
    parts.append("<|assistant|>\n")
    return "\n".join(parts)


async def _stream_completions(engine, prompt, params, model_name):
    import json

    async for output in engine.generate(prompt, params):
        chunk = {
            "id": f"cmpl-{uuid.uuid4().hex[:8]}",
            "object": "text_completion",
            "model": model_name,
            "choices": [
                {
                    "text": output.text,
                    "index": 0,
                    "finish_reason": "stop" if output.finished else None,
                }
            ],
        }
        yield f"data: {json.dumps(chunk)}\n\n"
    yield "data: [DONE]\n\n"


async def _stream_chat(engine, prompt, params, model_name):
    import json

    async for output in engine.generate(prompt, params):
        chunk = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion.chunk",
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": output.text},
                    "finish_reason": "stop" if output.finished else None,
                }
            ],
        }
        yield f"data: {json.dumps(chunk)}\n\n"
    yield "data: [DONE]\n\n"
