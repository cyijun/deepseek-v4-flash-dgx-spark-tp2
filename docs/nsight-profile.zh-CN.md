# Nsight Systems 双机观测报告

[English](nsight-profile.md) | [简体中文]

## 结论

已经对官方 checkpoint 和 packed-NVFP4 checkpoint 的完整 TP=2 链路做了双 rank
Nsight Systems 采集。首要瓶颈不是 RoCE 带宽：

- 两套 trace 中，B12X 都占累计 GPU kernel 时间的约 51–53%。
- NVFP4 trace 的 GPU kernel 调用数是官方版的 1.93 倍，B12X 调用数为 1.31 倍；较低的
  speculative acceptance 导致更多 target verification。
- 与 kernel 时间窗对齐后，官方版 SM active 约 87%，NVFP4 约 81%；但 SM issue 只有
  15%/11%，Tensor active 约 6%。GPU 一直有工作，但大量短 kernel 和依赖链没有吃满
  tensor throughput。
- 200 Gbit/s RoCE 链路每方向只承载约 2.46–3.08 Gbit/s，而且 error、discard、ECN、
  CNP、retransmit、timeout、`port_xmit_wait` 和 out-of-buffer 均没有新增计数。

两种 checkpoint 的 MTP 接受率不同，所以单次 trace 的 output tok/s 不能当作严格 A/B；
trace 用于定位时间消耗，稳态性能使用额外的 C6x256 无 profiler 结果。

## 采集口径

| 项目 | 配置 |
| --- | --- |
| 硬件 | 2x DGX Spark，GB10/SM121，TP=2，PP=1 |
| workload | 同步 C6；每请求 30 prompt token、最多 64 output token |
| 安全限制 | 每节点 6 GiB KV、108 GiB no-swap cgroup、10 GiB reserve、512 MiB swap-growth 熔断 |
| 镜像 | 两端均为 `sha256:cdeb2cea590c15ee49e70e69d733ba06ac4ac4c0eda09ad0c6fef61b25a76242` |
| Nsight | 2025.3.2.474；延迟开启 CUDA/NVTX/OSRT/CUBLAS/CUDNN 采集 |
| GPU metrics | host 侧 1 kHz，并按应用首末 CUDA kernel 对齐 |
| 网络 | workload 前后读取 RDMA port、mlx5 hardware 和 netdev sysfs counter |

原始 `.nsys-rep`/SQLite 保存在 git 忽略的 `profiles/` 中；归一化数据见
[`data/nsight-summary.csv`](../data/nsight-summary.csv)。

## 结果

| Profile | KV | target / draft | trace wall | trace output rate |
| --- | --- | --- | ---: | ---: |
| 官方版 | `fp8_ds_mla`，584 B/token | B12X W4A16 / B12X W4A16 | 2.423 s | 158.48 tok/s |
| NVFP4 | `nvfp4_ds_mla`，288 B/token | B12X W4A16 / Marlin | 4.189 s | 91.66 tok/s |

官方 trace 对应窗口的 draft acceptance 为 65.4%；NVFP4 跨两个日志窗口合计约 35%。因此
更可比的 C6x256 稳态结果是：

| Runtime | 三次稳态 output tok/s | 均值 |
| --- | --- | ---: |
| 适配版 + 官方 checkpoint + MTP-5 | 127.54 / 118.80 / 128.25 | **124.86** |
| 适配版 + NVFP4 + MTP-5 | 105.69 / 106.14 / 107.42 | **106.42** |
| 原版 nightly + 官方 checkpoint + 无 MTP | 85.89 / 84.34 / 87.28 | **85.84** |

适配版官方链路比唯一能启动的 stock target-only 链路高 45.5%。这是“可用 MTP-5”的系统
收益，不能解释成 B12X 单 kernel 比 DeepGEMM 快 45.5%。

| Profile / rank | kernel span | kernel calls | B12X | NCCL | sparse attention |
| --- | ---: | ---: | ---: | ---: | ---: |
| 官方 / 0 | 2411.8 ms | 44,930 | 920 次 / 1381.0 ms / 1501.1 us | 229.2 ms | 963 次 / 32.5 ms |
| 官方 / 1 | 2411.4 ms | 44,930 | 920 次 / 1390.9 ms / 1511.9 us | 201.9 ms | 963 次 / 34.4 ms |
| NVFP4 / 0 | 4176.6 ms | 86,617 | 1204 次 / 2056.6 ms / 1708.1 us | 305.3 ms | 1374 次 / 129.1 ms |
| NVFP4 / 1 | 4176.2 ms | 86,617 | 1204 次 / 2060.2 ms / 1711.2 us | 352.1 ms | 1374 次 / 129.1 ms |

NVFP4 还在 168 个 Marlin draft kernel 中花费约 98–99 ms。NVFP4 B12X 单次平均比官方版
高 13.8%，但两者 checkpoint、routed rows、B12X runtime family 和 target/draft 混合均不同，
不能当成纯 kernel 因果 A/B。

| Profile / rank | GR active | SM active | SM issue | Tensor active | Compute warps |
| --- | ---: | ---: | ---: | ---: | ---: |
| 官方 / 0 | 99.03% | 86.64% | 14.96% | 6.05% | 21.28% |
| 官方 / 1 | 98.97% | 86.86% | 14.96% | 6.05% | 21.33% |
| NVFP4 / 0 | 92.66% | 80.85% | 10.56% | 5.63% | 16.79% |
| NVFP4 / 1 | 93.55% | 80.91% | 10.51% | 5.58% | 16.88% |

两个 rank 的 GPU 指标高度一致。NCCL 时间差在两种 profile 中方向相反，更像 rank wait，
而不是固定某一端 NIC 变慢。

## RoCE

按 `port_xmit_data`/`port_rcv_data` 的 4-octet 单位和应用 kernel span 计算：

| Profile | 每方向数据 | 每方向 packet | 有效速率 |
| --- | ---: | ---: | ---: |
| 官方版 | 约 0.928 GB | 约 114.3 万 | 3.079 Gbit/s |
| NVFP4 | 约 1.283 GB | 约 157.9 万 | 2.458 Gbit/s |

所有拥塞和可靠性计数 delta 都为零。因此当前证据不支持优先换 OFED 或调大带宽；小消息下的
NCCL 时间更可能包含同步等待和 iteration 结构成本。

## 原版 vLLM nightly

两端缓存的 stock image ID 相同，为 `177a...`，版本为 vLLM
`0.27.2rc1.dev110+gacb0f1dcd`、Torch 2.13.0+cu130、FlashInfer 0.6.16.post3。

1. 强制 B12X 时，stock MXFP4 oracle 在加载权重前就拒绝 `flashinfer_b12x`。
2. `auto` 会选 `DEEPGEMM_MXFP4`，但 MTP-5 在 DSpark sparse-MLA warmup 失败：
   `num_tokens=5` 被送入要求 `num_tokens > 64` 的 paged kernel。
3. 关闭 MTP 后可以稳定服务；排除一次 63.38 tok/s 冷运行后，三次均值为 85.84 tok/s。

精确尝试记录见
[`data/stock-nightly-summary.csv`](../data/stock-nightly-summary.csv)。这说明 stock 已有可用的
target-only 基线，但目前还没有与适配版等价的 B12X + MTP-5 链路。

## Nsight 更新与 OFED

两台机器当前是 Nsight Systems 2025.3.2。NVIDIA 下载页已提供更新的 Linux Arm
Server / Grace 安装包，因此 Nsight 可以独立升级；为保证本轮基线不变，本次没有升级。

DGX OS release notes 明确写明 Arm64 DGX Spark 使用 inbox OFED 50.0-2，并且不支持
DOCA-OFED。当前机器也正是 `rdma-core` 50.0-2ubuntu0.2、inbox `mlx5_core`、Secure
Boot 开启。`nsys status --network` 所说的 “NVIDIA OFED driver is not installed” 表示
原生 NIC collector 期望另一套 NVIDIA/MLNX OFED 集成，不代表 Spark 的 RDMA 不工作。

不要为了打开一个 Nsight NIC panel 强装 DOCA/MLNX OFED。保留官方支持的 inbox stack，
继续用本报告中的 sysfs counter。官方资料：

- [DGX Spark Nsight tools](https://docs.nvidia.com/dgx/dgx-spark/nsight.html)
- [Nsight Systems downloads](https://developer.nvidia.com/nsight-systems/get-started)
- [DGX OS 7 release notes](https://docs.nvidia.com/dgx/dgx-os-7-user-guide/release_notes.html)
- [DGX Spark OS/component update](https://docs.nvidia.com/dgx/dgx-spark/os-and-component-update.html)

## 优化顺序

1. 先改善 NVFP4 checkpoint 的 draft acceptance / draft integration；额外 verification 会放大
   target、attention、routing 和 collective 的全部成本。
2. 再用 shape-aligned microbenchmark 优化 NVFP4 target 的小 M B12X 路径。
3. 区分 target/draft grid 后再优化 launch/routing；NVFP4 trace 的 kernel 数接近两倍。
4. 除非长上下文或更大 batch 的新 trace 改变结论，否则 RoCE 只做可靠性和延迟保护，不作为
   吞吐优化第一优先级。
