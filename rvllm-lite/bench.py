#!/usr/bin/env python3
"""rvLLM-lite direct engine benchmark.

Same format as rvLLM's vllm_direct_bench.py for apples-to-apples comparison.

Usage:
    python3 rvllm-lite/bench.py --model /root/models/Qwen2.5-7B
"""

import time
import json
import argparse

PROMPTS = [
    "Explain the theory of relativity in simple terms.",
    "Write a Python function to sort a list of integers.",
    "What are the main differences between TCP and UDP?",
    "Describe the process of photosynthesis step by step.",
    "Write a short story about a robot learning to paint.",
    "Explain how a transformer neural network works.",
    "What are the advantages of Rust over C++?",
    "Describe the water cycle in detail.",
    "Write a haiku about machine learning.",
    "Explain the concept of recursion with an example.",
    "What is the difference between a stack and a queue?",
    "Describe how HTTPS encryption works.",
    "Write a SQL query to find duplicate records in a table.",
    "Explain the CAP theorem in distributed systems.",
    "What are the main principles of object-oriented programming?",
    "Describe the architecture of a modern CPU.",
    "Write a regular expression to validate email addresses.",
    "Explain how garbage collection works in Java.",
    "What is the difference between concurrency and parallelism?",
    "Describe the MapReduce programming model.",
    "Explain how a B-tree index works in databases.",
    "What are the trade-offs between microservices and monoliths?",
    "Describe the process of DNS resolution.",
    "Write pseudocode for the A* pathfinding algorithm.",
]


def main():
    parser = argparse.ArgumentParser(description="rvLLM-lite direct engine benchmark")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--output", default="/root/results_rvllm_lite.json")
    parser.add_argument("--concurrency", type=str, default="16,24,32,48,64,96,128")
    args = parser.parse_args()

    from vllm import LLM, SamplingParams

    print(f"Loading model: {args.model}")
    t0 = time.perf_counter()
    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=False,
        max_model_len=4096,
        dtype="float16",
        disable_log_stats=True,
    )
    load_time = time.perf_counter() - t0
    print(f"Model loaded in {load_time:.1f}s")

    sampling = SamplingParams(temperature=0.8, max_tokens=args.max_tokens)
    batch_sizes = [int(x) for x in args.concurrency.split(",")]

    print()
    print("rvLLM-lite direct engine benchmark (H100 SXM)")
    print(f"Model: {args.model}, max_tokens={args.max_tokens}")
    print(f"{'N':>6} | {'tok/s':>10} | {'tokens':>8} | {'elapsed':>8}")
    print("-" * 45)

    results = []
    for i, n in enumerate(batch_sizes):
        prompts = [PROMPTS[j % len(PROMPTS)] for j in range(n)]
        if i == 0:
            llm.generate(prompts[:4], sampling)  # warmup
        t0 = time.perf_counter()
        out = llm.generate(prompts, sampling)
        elapsed = time.perf_counter() - t0
        total = sum(len(o.outputs[0].token_ids) for o in out)
        tps = total / elapsed
        r = {
            "n": n,
            "total_tokens": total,
            "elapsed_sec": round(elapsed, 3),
            "tok_per_sec": round(tps, 1),
            "avg_tokens_per_req": round(total / n, 1),
        }
        results.append(r)
        print(f"{n:>6} | {tps:>10,.1f} | {total:>8,} | {elapsed:>7.2f}s")

    output = {
        "model": args.model,
        "max_tokens": args.max_tokens,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "load_time_sec": round(load_time, 1),
        "results": results,
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
