#!/usr/bin/env python3
"""
rvLLM production load test.
Tests continuous batching under concurrent HTTP load.

Usage:
    python3 bench/loadtest.py --url http://localhost:8000 --concurrency 32 --duration 60 --max-tokens 128
    python3 bench/loadtest.py --url http://localhost:8000 --concurrency 32 --duration 60 --json results.json
"""

import asyncio
import aiohttp
import json
import random
import time
import statistics
import argparse
import sys

PROMPTS = [
    "Explain quantum computing in simple terms",
    "Write a Python function to sort a list",
    "What are the benefits of renewable energy",
    "Describe how a transformer neural network works",
    "What are the main differences between TCP and UDP",
    "Explain the theory of relativity in simple terms",
    "Describe the process of photosynthesis step by step",
    "Write a short story about a robot learning to paint",
    "What are the advantages of Rust over C++",
    "Describe the water cycle in detail",
    "Explain the concept of recursion with an example",
    "What is the difference between a stack and a queue",
    "Describe how HTTPS encryption works",
    "Explain the CAP theorem in distributed systems",
    "What are the main principles of object-oriented programming",
    "Describe the architecture of a modern CPU",
    "Explain how garbage collection works in Java",
    "What is the difference between concurrency and parallelism",
    "Describe the MapReduce programming model",
    "Explain how a B-tree index works in databases",
    "What are the trade-offs between microservices and monoliths",
    "Describe the process of DNS resolution",
    "Write pseudocode for the A* pathfinding algorithm",
    "Explain the concept of eventual consistency",
    "What is a bloom filter and when would you use one",
    "Describe how consistent hashing works",
    "Explain the differences between REST and GraphQL",
    "What is the actor model in concurrent programming",
    "Describe how a neural network learns through backpropagation",
    "Explain the concept of zero-knowledge proofs",
]

JSON_HEADERS = {
    "Content-Type": "application/json",
    "Connection": "keep-alive",
}


class LoadTester:
    def __init__(self, url, concurrency, duration, max_tokens, temperature, model):
        self.url = url.rstrip("/")
        self.concurrency = concurrency
        self.duration = duration
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.model = model
        self.results = []
        self.errors = 0
        self.error_details = []
        self.total_tokens = 0
        self.total_prompt_tokens = 0
        self.start_time = 0
        self.end_time = 0
        self._lock = asyncio.Lock()

    async def send_request(self, session, payload_bytes):
        start = time.perf_counter()
        try:
            async with session.post(
                f"{self.url}/v1/completions",
                data=payload_bytes,
                headers=JSON_HEADERS,
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    async with self._lock:
                        self.errors += 1
                        self.error_details.append(f"HTTP {resp.status}: {body[:200]}")
                    return

                data = json.loads(await resp.read())
                elapsed = time.perf_counter() - start
                usage = data.get("usage", {})
                completion_tokens = usage.get("completion_tokens", 0)
                prompt_tokens = usage.get("prompt_tokens", 0)

                if completion_tokens == 0:
                    async with self._lock:
                        self.errors += 1
                        self.error_details.append("0 completion tokens")
                    return

                async with self._lock:
                    self.results.append({
                        "latency": elapsed,
                        "completion_tokens": completion_tokens,
                        "prompt_tokens": prompt_tokens,
                        "tok_per_sec": completion_tokens / elapsed,
                    })
                    self.total_tokens += completion_tokens
                    self.total_prompt_tokens += prompt_tokens
        except asyncio.CancelledError:
            raise
        except Exception as e:
            async with self._lock:
                self.errors += 1
                self.error_details.append(str(e)[:200])

    async def worker(self, session, deadline):
        while time.perf_counter() < deadline:
            prompt = random.choice(PROMPTS)
            payload = json.dumps(
                {
                    "model": self.model,
                    "prompt": prompt,
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "stream": False,
                },
                separators=(",", ":"),
            ).encode("utf-8")
            await self.send_request(session, payload)

    async def run(self):
        timeout = aiohttp.ClientTimeout(total=max(300, self.duration * 2))
        connector = aiohttp.TCPConnector(limit=0, limit_per_host=0, ttl_dns_cache=3600)
        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            # Warmup: single request to prime model/caches
            print(f"Warming up...", end=" ", flush=True)
            warmup_payload = json.dumps(
                {
                    "model": self.model,
                    "prompt": PROMPTS[0],
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "stream": False,
                },
                separators=(",", ":"),
            ).encode("utf-8")
            await self.send_request(session, warmup_payload)
            if self.results:
                print(f"done ({self.results[0]['latency']*1000:.0f}ms)")
            else:
                print("failed")
                if self.error_details:
                    print(f"  Error: {self.error_details[-1]}")
                    return

            # Reset stats after warmup
            self.results.clear()
            self.errors = 0
            self.error_details.clear()
            self.total_tokens = 0
            self.total_prompt_tokens = 0

            # Run concurrent workers for the specified duration
            self.start_time = time.perf_counter()
            deadline = self.start_time + self.duration
            print(f"Running {self.concurrency} concurrent workers for {self.duration}s...")

            workers = [self.worker(session, deadline) for _ in range(self.concurrency)]
            await asyncio.gather(*workers)

            self.end_time = time.perf_counter()

    def get_results(self):
        if not self.results:
            return None

        actual_duration = self.end_time - self.start_time
        latencies = sorted([r["latency"] for r in self.results])
        n = len(latencies)

        def pct(p):
            idx = min(int(n * p / 100), n - 1)
            return latencies[idx]

        return {
            "config": {
                "url": self.url,
                "concurrency": self.concurrency,
                "duration_target": self.duration,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "model": self.model,
            },
            "summary": {
                "actual_duration_s": actual_duration,
                "total_requests": n,
                "errors": self.errors,
                "total_completion_tokens": self.total_tokens,
                "total_prompt_tokens": self.total_prompt_tokens,
                "throughput_tok_s": self.total_tokens / actual_duration,
                "requests_per_s": n / actual_duration,
                "avg_tokens_per_request": self.total_tokens / n,
            },
            "latency_ms": {
                "avg": statistics.mean(latencies) * 1000,
                "p50": pct(50) * 1000,
                "p75": pct(75) * 1000,
                "p90": pct(90) * 1000,
                "p95": pct(95) * 1000,
                "p99": pct(99) * 1000,
                "min": latencies[0] * 1000,
                "max": latencies[-1] * 1000,
                "stdev": statistics.stdev(latencies) * 1000 if n > 1 else 0,
            },
            "per_request_tok_s": {
                "avg": statistics.mean([r["tok_per_sec"] for r in self.results]),
                "p50": sorted([r["tok_per_sec"] for r in self.results])[n // 2],
            },
        }

    def report(self):
        results = self.get_results()
        if results is None:
            print("\nNo successful requests.")
            if self.error_details:
                print(f"Errors ({self.errors}):")
                for e in self.error_details[:5]:
                    print(f"  - {e}")
            return

        s = results["summary"]
        l = results["latency_ms"]
        t = results["per_request_tok_s"]

        print(f"\n{'=' * 50}")
        print(f"  Load Test Results")
        print(f"{'=' * 50}")
        print(f"  URL:             {self.url}")
        print(f"  Duration:        {s['actual_duration_s']:.1f}s (target {self.duration}s)")
        print(f"  Concurrency:     {self.concurrency}")
        print(f"  Max tokens:      {self.max_tokens}")
        print(f"{'=' * 50}")
        print(f"  Requests:        {s['total_requests']} ({s['errors']} errors)")
        print(f"  Total tokens:    {s['total_completion_tokens']}")
        print(f"  Throughput:      {s['throughput_tok_s']:.0f} tok/s")
        print(f"  Requests/s:      {s['requests_per_s']:.1f}")
        print(f"{'=' * 50}")
        print(f"  Latency (ms):")
        print(f"    avg:           {l['avg']:.0f}")
        print(f"    p50:           {l['p50']:.0f}")
        print(f"    p75:           {l['p75']:.0f}")
        print(f"    p90:           {l['p90']:.0f}")
        print(f"    p95:           {l['p95']:.0f}")
        print(f"    p99:           {l['p99']:.0f}")
        print(f"    min:           {l['min']:.0f}")
        print(f"    max:           {l['max']:.0f}")
        print(f"    stdev:         {l['stdev']:.0f}")
        print(f"{'=' * 50}")
        print(f"  Per-request tok/s:")
        print(f"    avg:           {t['avg']:.1f}")
        print(f"    p50:           {t['p50']:.1f}")
        print(f"{'=' * 50}")

        if self.error_details:
            print(f"\n  First errors:")
            for e in self.error_details[:3]:
                print(f"    - {e}")


def main():
    parser = argparse.ArgumentParser(
        description="rvLLM production load test -- continuous batching under concurrent HTTP load"
    )
    parser.add_argument("--url", default="http://localhost:8000", help="Server base URL")
    parser.add_argument("--concurrency", type=int, default=16, help="Number of concurrent workers")
    parser.add_argument("--duration", type=int, default=30, help="Test duration in seconds")
    parser.add_argument("--max-tokens", type=int, default=128, help="Max tokens per request")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--model", default="default", help="Model name for request payload")
    parser.add_argument("--json", default=None, help="Save results to JSON file")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for prompt selection")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    tester = LoadTester(
        url=args.url,
        concurrency=args.concurrency,
        duration=args.duration,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        model=args.model,
    )

    asyncio.run(tester.run())
    tester.report()

    if args.json:
        results = tester.get_results()
        if results:
            with open(args.json, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {args.json}")


if __name__ == "__main__":
    main()
