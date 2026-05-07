"""
scripts/load_test.py
Sends 500 prompts through the autopilot API and compares routed vs baseline cost.
Run: python scripts/load_test.py [--url http://localhost:8000] [--count 500]
"""

import sys
import time
import json
import random
import argparse
import csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx

API_URL = "http://localhost:8000"
BASELINE_COST_PER_1K = 0.002  # GPT-4 equivalent

TEST_PROMPTS = [
    # Simple
    "What is 7 times 8?",
    "What is the capital of Germany?",
    "Who wrote Hamlet?",
    "What does CPU stand for?",
    "Convert 50 Celsius to Fahrenheit.",
    "What is the square root of 256?",
    "Define the word 'serendipity'.",
    "What year did World War 1 start?",
    "What is the boiling point of water in Kelvin?",
    "How many bytes are in a megabyte?",
    # Medium
    "Explain how Python's garbage collector works.",
    "Write a function to check if a number is prime.",
    "What are the differences between REST and GraphQL?",
    "Explain the CAP theorem with an example.",
    "How does JWT authentication work?",
    "Write a Python decorator for timing function calls.",
    "What is a context manager and when should you use one?",
    "Explain Big O notation with three examples.",
    "How does Redis handle cache eviction?",
    "Describe the publisher-subscriber design pattern.",
    # Complex
    "Design a distributed rate limiter that works across 50 servers with Redis.",
    "How would you architect a real-time bidding system for 1M concurrent users?",
    "Design a globally distributed database with multi-region active-active replication.",
    "Implement a fraud detection system using machine learning at bank scale.",
    "How would you build a recommendation engine for 100M daily active users?",
    "Design a streaming data pipeline processing 100TB of logs per day.",
    "Build a zero-downtime migration strategy from monolith to microservices.",
    "Design a multi-tenant SaaS with strict data isolation and SOC2 compliance.",
    "How would you implement exactly-once semantics in a distributed message queue?",
    "Architect a vector database for semantic search over 1 billion embeddings.",
]


def run_load_test(api_url: str, count: int, output_file: str):
    print(f"{'='*60}")
    print(f"  LLM Cost Autopilot — Load Test")
    print(f"  Target: {api_url}")
    print(f"  Requests: {count}")
    print(f"{'='*60}\n")

    results = []
    total_routed_cost = 0.0
    total_tokens = 0
    success = 0
    errors = 0
    t_start = time.time()

    for i in range(count):
        prompt = random.choice(TEST_PROMPTS)
        try:
            resp = httpx.post(
                f"{api_url}/v1/completions",
                json={"prompt": prompt, "max_tokens": 200},
                timeout=120,
            )
            if resp.status_code == 200:
                data = resp.json()
                total_routed_cost += data.get("cost", 0)
                total_tokens += data.get("total_tokens", 0)
                success += 1
                results.append({
                    "prompt": prompt[:60],
                    "model": data.get("model_used"),
                    "tier": data.get("complexity_tier"),
                    "tokens": data.get("total_tokens"),
                    "cost": data.get("cost"),
                    "savings": data.get("estimated_savings"),
                    "latency_ms": data.get("latency_ms"),
                    "status": "ok",
                })
                if (i + 1) % 50 == 0:
                    elapsed = time.time() - t_start
                    rps = (i + 1) / elapsed
                    print(f"  [{i+1:4d}/{count}] ✅ {rps:.1f} req/s | tokens={total_tokens:,} | cost=${total_routed_cost:.4f}")
            else:
                errors += 1
                results.append({"status": "error", "code": resp.status_code, "prompt": prompt[:60]})
        except Exception as e:
            errors += 1
            results.append({"status": "exception", "error": str(e), "prompt": prompt[:60]})

    elapsed = time.time() - t_start
    baseline_cost = (total_tokens / 1000) * BASELINE_COST_PER_1K
    savings = baseline_cost - total_routed_cost
    savings_pct = (savings / baseline_cost * 100) if baseline_cost > 0 else 0

    # Save results
    Path(output_file).parent.mkdir(exist_ok=True)
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys() if results else [])
        writer.writeheader()
        writer.writerows(results)

    print(f"\n{'='*60}")
    print(f"  LOAD TEST RESULTS")
    print(f"{'='*60}")
    print(f"  Duration          : {elapsed:.1f}s")
    print(f"  Requests          : {count}")
    print(f"  Successful        : {success}")
    print(f"  Errors            : {errors}")
    print(f"  Throughput        : {success/elapsed:.2f} req/s")
    print(f"  Total Tokens      : {total_tokens:,}")
    print(f"  Routed Cost       : ${total_routed_cost:.4f}")
    print(f"  Baseline (GPT-4)  : ${baseline_cost:.4f}")
    print(f"  Cost Savings      : ${savings:.4f} ({savings_pct:.1f}%)")
    print(f"  Results saved     : {output_file}")
    print(f"{'='*60}")

    return savings_pct


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=API_URL)
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--output", default="data/load_test_results.csv")
    args = parser.parse_args()
    run_load_test(args.url, args.count, args.output)
