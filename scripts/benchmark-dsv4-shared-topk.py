#!/usr/bin/env python3
"""Compare DeepSeek-V4 top-k shared-expert append paths."""

from __future__ import annotations

import argparse
import json
import time

import torch
from vllm.model_executor.layers.fused_moe.router.dsv4_topk import dsv4_topk


def _cuda_time(fn, warmup: int, repetitions: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repetitions):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repetitions


def _wall_time(fn, warmup: int, repetitions: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(repetitions):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - started) * 1e3 / repetitions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=36)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--repetitions", type=int, default=2000)
    args = parser.parse_args()

    torch.manual_seed(0)
    num_experts = 256
    gating_output = torch.randn(
        (args.tokens, num_experts), dtype=torch.float32, device="cuda"
    )
    correction_bias = torch.randn(num_experts, dtype=torch.float32, device="cuda")

    def cat_path():
        weights, ids = dsv4_topk(gating_output, correction_bias, torch.int32, 1.0)
        shared_ids = torch.arange(
            num_experts, num_experts + 1, dtype=ids.dtype, device=ids.device
        ).expand(args.tokens, 1)
        shared_weights = torch.ones(
            (args.tokens, 1), dtype=weights.dtype, device=weights.device
        )
        return (
            torch.cat((weights, shared_weights), dim=-1),
            torch.cat((ids, shared_ids), dim=-1),
        )

    def fused_path():
        return dsv4_topk(
            gating_output,
            correction_bias,
            torch.int32,
            1.0,
            num_fused_shared_experts=1,
            shared_expert_weight=1.0,
        )

    cat_weights, cat_ids = cat_path()
    fused_weights, fused_ids = fused_path()
    torch.testing.assert_close(cat_weights, fused_weights, atol=0, rtol=0)
    torch.testing.assert_close(cat_ids, fused_ids, atol=0, rtol=0)

    print(
        json.dumps(
            {
                "tokens": args.tokens,
                "cat_cuda_ms": _cuda_time(cat_path, args.warmup, args.repetitions),
                "fused_cuda_ms": _cuda_time(fused_path, args.warmup, args.repetitions),
                "cat_wall_ms": _wall_time(cat_path, args.warmup, args.repetitions),
                "fused_wall_ms": _wall_time(fused_path, args.warmup, args.repetitions),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
