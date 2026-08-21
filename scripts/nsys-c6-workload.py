#!/usr/bin/env python3
"""Issue one synchronized C6 OpenAI-compatible workload for Nsight capture."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8888/v1/chat/completions")
    parser.add_argument("--model", required=True)
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--label", default="nsys-c6")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.concurrency < 1 or args.concurrency > 6:
        raise SystemExit("concurrency must be in [1, 6]")

    barrier = __import__("threading").Barrier(args.concurrency)

    def request_one(index: int) -> dict[str, object]:
        prompt = (
            "Explain how tensor parallel inference coordinates attention, "
            "mixture-of-experts routing, and collective communication. "
            f"This is deterministic profiling request {index}."
        )
        body = json.dumps(
            {
                "model": args.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": args.max_tokens,
                "temperature": 0,
            }
        ).encode()
        request = urllib.request.Request(
            args.url, body, {"Content-Type": "application/json"}
        )
        barrier.wait()
        started = time.perf_counter()
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            result = json.load(response)
        elapsed = time.perf_counter() - started
        usage = result.get("usage", {})
        choice = result.get("choices", [{}])[0]
        return {
            "request": index,
            "latency_seconds": elapsed,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "finish_reason": choice.get("finish_reason"),
        }

    started = time.perf_counter()
    errors: list[dict[str, object]] = []
    results: list[dict[str, object]] = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.concurrency
    ) as executor:
        futures = {
            executor.submit(request_one, index): index
            for index in range(args.concurrency)
        }
        for future in concurrent.futures.as_completed(futures):
            try:
                results.append(future.result())
            except Exception as error:  # noqa: BLE001 - preserve workload errors
                errors.append(
                    {"request": futures[future], "error": f"{type(error).__name__}: {error}"}
                )

    elapsed = time.perf_counter() - started
    results.sort(key=lambda item: int(item["request"]))
    completion_tokens = sum(
        int(item["completion_tokens"] or 0) for item in results
    )
    latencies = [float(item["latency_seconds"]) for item in results]
    summary = {
        "label": args.label,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "model": args.model,
        "concurrency": args.concurrency,
        "requested_max_tokens": args.max_tokens,
        "wall_seconds": elapsed,
        "completion_tokens": completion_tokens,
        "output_tokens_per_second": completion_tokens / elapsed if elapsed else None,
        "latency_seconds_mean": statistics.fmean(latencies) if latencies else None,
        "latency_seconds_max": max(latencies) if latencies else None,
        "results": results,
        "errors": errors,
    }
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
