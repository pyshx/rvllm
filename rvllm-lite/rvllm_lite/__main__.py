"""Entry point: python -m rvllm_lite"""

import argparse
import logging
import uvicorn

logging.basicConfig(level=logging.INFO, format="%(name)s | %(levelname)s | %(message)s")
log = logging.getLogger("rvllm-lite")


def main():
    parser = argparse.ArgumentParser(description="rvLLM-lite inference server")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model name or path")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--enforce-eager", action="store_true", help="Disable torch.compile")
    args = parser.parse_args()

    from rvllm_lite.api.app import create_app

    app = create_app(args)
    log.info(f"rvLLM-lite serving {args.model} on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
