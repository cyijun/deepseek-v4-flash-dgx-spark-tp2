# Nsight Systems report: two-node DeepSeek-V4-Flash

[English] | [简体中文](nsight-profile.zh-CN.md)

## Result first

The complete TP=2 path was captured on both DGX Spark nodes for the official
checkpoint and the packed-NVFP4 checkpoint. The main result is not a RoCE
bandwidth problem:

- B12X accounts for about 51–53% of summed GPU-kernel duration in both traces.
- The NVFP4 run launches 1.93x as many GPU kernels as the official run and
  1.31x as many B12X kernels. Its lower speculative acceptance causes more
  target verification work.
- During the kernel-aligned window, SM active is about 87% for the official
  run and 81% for NVFP4, but SM issue is only 15% and 11%; tensor active is
  around 6%. The GPUs stay busy with many short, dependency-heavy kernels
  without saturating tensor throughput.
- RoCE carries only 2.46–3.08 Gbit/s per direction on a 200 Gbit/s link. No
  error, discard, ECN, CNP, retransmit, timeout, `port_xmit_wait`, or
  out-of-buffer counter increased.

The trace-specific output rates are not an end-to-end A/B because the two
checkpoints produced different MTP acceptance. They are used here to explain
where each captured batch spent time.

## Environment and capture contract

| Item | Value |
| --- | --- |
| Hardware | 2x DGX Spark, GB10/SM121, TP=2, PP=1 |
| Workload | one synchronized C6 batch, 30 prompt tokens and 64 requested output tokens per request |
| Safety | 6 GiB KV/node, 108 GiB no-swap cgroup, 10 GiB runtime reserve, 512 MiB swap-growth stop |
| Image | `sha256:cdeb2cea590c15ee49e70e69d733ba06ac4ac4c0eda09ad0c6fef61b25a76242` on both nodes |
| vLLM / FlashInfer source | `2db20513...` / `6398edbbc...` |
| Nsight Systems | 2025.3.2.474, delayed CUDA/NVTX/OSRT/CUBLAS/CUDNN collection |
| Host GPU metrics | 1 kHz, aligned to the first and last application CUDA kernel |
| Network evidence | RDMA port, mlx5 hardware, and netdev sysfs counters before/after the batch |

The application profiler is mounted read-only from the host and starts only
after model load and CUDA-graph capture. Separate host sessions collect GPU
metrics. Raw `.nsys-rep` and SQLite files stay under ignored `profiles/`
directories; the normalized results are committed in
[`data/nsight-summary.csv`](../data/nsight-summary.csv).

## End-to-end trace windows

| Profile | KV layout | Target / draft | Wall time | Output rate | Trace acceptance |
| --- | --- | --- | ---: | ---: | --- |
| Official | `fp8_ds_mla`, 584 B/token | B12X W4A16 / B12X W4A16 | 2.423 s | 158.48 tok/s | 65.4% draft acceptance in the reporting window |
| NVFP4 | `nvfp4_ds_mla`, 288 B/token | B12X W4A16 / Marlin | 4.189 s | 91.66 tok/s | about 35% across the two reporting windows that cover the batch |

The official trace happened to have unusually favorable acceptance. Longer
unprofiled C6x256 runs are therefore the performance reference:

| Runtime | Configuration | Steady output tok/s |
| --- | --- | ---: |
| Adapted, official checkpoint | B12X target+draft, MTP-5 | 127.54, 118.80, 128.25; mean **124.86** |
| Adapted, NVFP4 checkpoint | B12X target, Marlin draft, MTP-5 | 105.69, 106.14, 107.42; mean **106.42** |
| Stock nightly, official checkpoint | auto-selected DeepGEMM MXFP4, no MTP | 85.89, 84.34, 87.28; mean **85.84** |

The adapted official lane is 45.5% faster than the only stock lane that
starts. This is a system result dominated by working MTP-5; it is not a claim
that B12X alone is 45.5% faster than DeepGEMM.

## GPU kernel evidence

The percentages below use summed kernel duration, which can exceed wall time
when streams overlap.

| Profile / rank | Kernel span | Kernel calls | B12X | NCCL | Sparse attention |
| --- | ---: | ---: | ---: | ---: | ---: |
| Official / 0 | 2411.8 ms | 44,930 | 920 calls, 1381.0 ms, 1501.1 us/call | 229.2 ms | DSv4 decode: 963 calls, 32.5 ms |
| Official / 1 | 2411.4 ms | 44,930 | 920 calls, 1390.9 ms, 1511.9 us/call | 201.9 ms | DSv4 decode: 963 calls, 34.4 ms |
| NVFP4 / 0 | 4176.6 ms | 86,617 | 1204 calls, 2056.6 ms, 1708.1 us/call | 305.3 ms | FP4 MLA: 1374 calls, 129.1 ms |
| NVFP4 / 1 | 4176.2 ms | 86,617 | 1204 calls, 2060.2 ms, 1711.2 us/call | 352.1 ms | FP4 MLA: 1374 calls, 129.1 ms |

NVFP4 also spends 98–99 ms in 168 Marlin draft kernels. The 13.8% difference
between the two B12X per-call averages is not a pure kernel A/B: the
checkpoints, routed rows, B12X runtime family, and target/draft mix differ.
It is still a useful optimization target after acceptance is controlled.

| Profile / rank | GR active | SM active | SM issue | Tensor active | Compute warps |
| --- | ---: | ---: | ---: | ---: | ---: |
| Official / 0 | 99.03% | 86.64% | 14.96% | 6.05% | 21.28% |
| Official / 1 | 98.97% | 86.86% | 14.96% | 6.05% | 21.33% |
| NVFP4 / 0 | 92.66% | 80.85% | 10.56% | 5.63% | 16.79% |
| NVFP4 / 1 | 93.55% | 80.91% | 10.51% | 5.58% | 16.88% |

The matched ranks make a hardware or one-sided scheduling fault unlikely.
The NCCL-duration asymmetry changes direction between profiles, so those
durations include rank waiting rather than demonstrating a consistently slow
NIC.

## RoCE evidence

`port_xmit_data` and `port_rcv_data` use four-octet units. Using the
application kernel span as the active interval:

| Profile | Bytes per direction | Packets per direction | Effective rate |
| --- | ---: | ---: | ---: |
| Official | about 0.928 GB | about 1.143 million | 3.079 Gbit/s |
| NVFP4 | about 1.283 GB | about 1.579 million | 2.458 Gbit/s |

Every checked congestion and reliability delta was zero. The data therefore
does not justify OFED replacement or link tuning as the first performance
action. NCCL time is more likely synchronization/iteration structure at this
small message regime.

## Stock vLLM nightly comparison

The cached stock image is the same immutable ID on both nodes and reports
vLLM `0.27.2rc1.dev110+gacb0f1dcd`, Torch 2.13.0+cu130, and FlashInfer
0.6.16.post3.

1. Forced `flashinfer_b12x` fails before loading weights because stock vLLM's
   MXFP4 oracle does not map that backend for the official checkpoint.
2. `auto` selects `DEEPGEMM_MXFP4`, loads the model, then MTP-5 startup fails
   in DSpark sparse MLA. FlashInfer routes `num_tokens=5` to a paged kernel
   that asserts `num_tokens > 64`.
3. With MTP disabled, stock starts and serves correctly. After excluding one
   cold 63.38 tok/s run, three C6x256 runs average 85.84 tok/s.

This means stock nightly has a usable target-only baseline, but no comparable
MTP-5/B12X deployment for this checkpoint today. See
[`data/stock-nightly-summary.csv`](../data/stock-nightly-summary.csv) for the
exact attempts.

## Nsight update and OFED decision

Both systems currently have Nsight Systems 2025.3.2. NVIDIA's current download
page offers a newer Linux Arm Server / Grace build, so a standalone Nsight
upgrade is possible. It was intentionally not changed during this baseline.

DGX OS release notes state that Arm64 DGX Spark uses inbox OFED 50.0-2 and that
DOCA-OFED is not supported. The hosts match that contract (`rdma-core`
50.0-2ubuntu0.2, inbox `mlx5_core`, Secure Boot enabled). Nsight's native NIC
collector reports “NVIDIA OFED driver is not installed” because it expects a
different NVIDIA/MLNX OFED integration; that is not evidence that Spark lacks
working RDMA.

Do not install DOCA/MLNX OFED merely to enable one profiler panel. Keep the
supported inbox stack and use the captured sysfs counters. Relevant NVIDIA
documents:

- [DGX Spark Nsight tools](https://docs.nvidia.com/dgx/dgx-spark/nsight.html)
- [Nsight Systems downloads](https://developer.nvidia.com/nsight-systems/get-started)
- [DGX OS 7 release notes](https://docs.nvidia.com/dgx/dgx-os-7-user-guide/release_notes.html)
- [DGX Spark OS and component update](https://docs.nvidia.com/dgx/dgx-spark/os-and-component-update.html)

## Recommended optimization order

1. Improve NVFP4 checkpoint draft acceptance or draft integration first; the
   extra verification iterations multiply every target, attention, routing,
   and collective cost.
2. Then optimize the NVFP4 target's small-M B12X path, using a shape-aligned
   microbenchmark before changing the full system.
3. Reduce launch and routing overhead only after separating target and draft
   grids; the NVFP4 trace contains nearly twice as many kernels.
4. Treat RoCE tuning as a guardrail/latency task, not a bandwidth-throughput
   task, unless a later long-context or larger-batch trace changes the data.
