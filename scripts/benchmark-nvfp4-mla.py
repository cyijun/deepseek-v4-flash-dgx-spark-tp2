#!/usr/bin/env python3
"""Bounded-memory benchmark for the true-layout NVFP4 sparse MLA kernel."""

import argparse
import statistics

import torch
from vllm.models.deepseek_v4.nvidia.ops.nvfp4_mla import (
    nvfp4_mla_sparse_attention,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=6)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--swa-width", type=int, default=128)
    parser.add_argument("--swa-len", type=int, default=128)
    parser.add_argument("--extra-width", type=int, default=128)
    parser.add_argument("--extra-len", type=int, default=2)
    parser.add_argument("--warmup-iters", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--capture", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(0)
    device = torch.device("cuda")
    query = torch.randn(
        args.tokens, args.heads, 512, dtype=torch.bfloat16, device=device
    )
    output = torch.empty_like(query)
    sinks = torch.zeros(args.heads, dtype=torch.float32, device=device)

    cache_block_size = 256
    swa_cache = torch.zeros(2, cache_block_size, 288, dtype=torch.uint8, device=device)
    extra_cache = torch.zeros(
        2, cache_block_size, 288, dtype=torch.uint8, device=device
    )
    swa_indices = torch.full(
        (args.tokens, args.swa_width), -1, dtype=torch.int32, device=device
    )
    extra_indices = torch.full(
        (args.tokens, args.extra_width), -1, dtype=torch.int32, device=device
    )
    swa_indices[:, : args.swa_len] = torch.arange(
        args.swa_len, dtype=torch.int32, device=device
    )
    extra_indices[:, : args.extra_len] = torch.arange(
        args.extra_len, dtype=torch.int32, device=device
    )
    swa_lens = torch.full(
        (args.tokens,), args.swa_len, dtype=torch.int32, device=device
    )
    extra_lens = torch.full(
        (args.tokens,), args.extra_len, dtype=torch.int32, device=device
    )

    def run() -> None:
        nvfp4_mla_sparse_attention(
            query=query,
            swa_cache=swa_cache,
            swa_indices=swa_indices,
            swa_lens=swa_lens,
            output=output,
            sm_scale=512**-0.5,
            sinks=sinks,
            extra_cache=extra_cache,
            extra_indices=extra_indices,
            extra_lens=extra_lens,
        )

    run()
    torch.cuda.synchronize()
    if args.capture:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        run = graph.replay

    for _ in range(args.warmup_iters):
        run()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(args.iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(args.iters)]
    for start, end in zip(starts, ends):
        start.record()
        run()
        end.record()
    torch.cuda.synchronize()
    samples_ms = [start.elapsed_time(end) for start, end in zip(starts, ends)]
    print(
        f"tokens={args.tokens} heads={args.heads} "
        f"swa={args.swa_len}/{args.swa_width} "
        f"extra={args.extra_len}/{args.extra_width} capture={args.capture}: "
        f"median={statistics.median(samples_ms):.3f} ms "
        f"mean={statistics.fmean(samples_ms):.3f} ms",
        flush=True,
    )


if __name__ == "__main__":
    main()
