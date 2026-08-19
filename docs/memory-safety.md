# DGX Spark 统一内存安全策略

## 风险模型

GB10 的 CUDA allocation 与 CPU process 共享系统 DRAM。模型 load、JIT compile、CUDA graph capture、KV allocation 和 page cache 会同时挤压 `MemAvailable`。一旦 CUDA-backed page 参与 swap，节点可能长时间失去响应；传统独立显存 OOM 估算不足以保护系统。

此前一次实验把两台机器 RAM 塞满并导致人工重启。此后所有部署必须遵守以下停止条件。

## 固定预算

| 项目 | 每节点限制 | 原因 |
| --- | ---: | --- |
| 容器 memory | 108 GiB | 在约 121 GiB 可见统一内存内保留 host recovery margin |
| 容器 memory+swap | 108 GiB | 禁止容器把超额内存转移到 swap |
| 启动前 `MemAvailable` | ≥110 GiB | 确保没有并行 workload 污染容量估算 |
| 运行中 `MemAvailable` | ≥10 GiB | 为 SSH、Docker、JIT 和停止操作保留恢复空间 |
| 相对启动 swap 增长 | ≤512 MiB | 捕获 CUDA/unified-memory swap thrash 的早期信号 |
| KV cache | 6 GiB | 满足 C6/4096 实验 envelope，同时低于失败的 10 GiB 试验 |
| 最大并发 | 6 | 用户要求；不尝试 C128 |

已测 runtime 在 KV 前约消耗 89.2 GiB/node。简单相加不是精确峰值模型，因为 graph capture、compiler workspace 和 page cache 在启动阶段有瞬态峰值，所以还需要运行时 guard。

## 三层保护

1. **Preflight**：在拉起任何 rank 前检查双端 `MemAvailable`、初始 `SwapFree`、模型 shard、image label、RoCE state 和残留容器。
2. **Cgroup**：两个 rank 都用 `--memory 108g --memory-swap 108g`，把故障限制在容器内。
3. **Guard loop**：startup loop 和独立的 `runtime-guard.sh` 每两秒读取双端 `/proc/meminfo`。越过任一阈值时收集 bounded log，删除两端容器并返回非零。

## 已观测边界

- 10 GiB KV trial 在 graph warmup 中越过 10 GiB host reserve，不再作为默认值。
- TileLang control 重试和当前 optimized image startup 均触发 swap-growth stop。
- optimized shared-routing image 在 head swap 增长 654 MiB 时被停止，未记录吞吐。
- 停止线触发不是“测试失败后继续等”；它是实验设计的一部分，不应通过提高阈值绕过。

## 每次实验前检查

```bash
awk '/MemAvailable|SwapTotal|SwapFree/ {print}' /proc/meminfo
ssh "$WORKER_HOST" "awk '/MemAvailable|SwapTotal|SwapFree/ {print}' /proc/meminfo"
docker ps
ssh "$WORKER_HOST" docker ps
```

只有两端 `MemAvailable >= 110 GiB` 且没有冲突容器时才启动。停止后应确认容器消失、内存回收，并保留任何 guard reason。
