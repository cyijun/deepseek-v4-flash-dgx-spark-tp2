# 端到端复现

## 1. 环境前提

- 两台 aarch64 DGX Spark / GB10 / SM121；
- Ubuntu 24.04、NVIDIA driver 可运行 CUDA 13-compatible container；
- 两端 Docker、NVIDIA Container Toolkit、SSH key 和 `/dev/infiniband`；
- 同一 RoCEv2 subnet，HCA port 为 `ACTIVE`；
- 两端相同 Hugging Face snapshot revision 和 48 个 safetensors shard；
- 启动前两端 `MemAvailable >= 110 GiB`。

不要在 Action runner、image build 或其他大模型仍占用内存时启动服务。

## 2. 取得部署仓库

```bash
git clone https://github.com/cyijun/deepseek-v4-flash-dgx-spark-tp2.git
cd deepseek-v4-flash-dgx-spark-tp2
cp config/deployment.env.example config/deployment.env
```

填写 worker SSH、RoCE IP/NIC/HCA、模型 cache root、snapshot revision 和 image。

## 3. 构建容器

### GitHub Action

在 Actions 中手工运行 **Package ARM64 runtime**。它先在标准 GitHub-hosted
`ubuntu-24.04-arm` 上准备 build context；标准 runner 的 14 GB 磁盘小于 vLLM base
image 的解压尺寸，不能承担最终 Docker build。

只想验证和下载源码上下文时，把 workflow input `prepare_only` 设为 `true`；此模式完全在
标准云 ARM runner 上完成并上传 `arm64-build-context` artifact，不等待 build runner。

建议在 `container-release` environment 配置 owner approval，并给自托管 runner 增加：

```text
self-hosted, linux, ARM64, dgx-spark
```

默认 build job 匹配 `dgx-spark` 标签。如已有 GitHub ARM larger runner（至少 150 GB
storage），在 repository Actions variable 中设置：

```text
ARM64_BUILD_RUNNER=<larger-runner-custom-label>
```

Action 从 fork checkout `config/versions.env` 中的固定 commit，发布两个 tag：手工 tag 和 `sha-<deployment-commit>`，同时生成 provenance。部署时优先把 GHCR digest 而不是可变 tag 写入 `IMAGE`。

### 本地构建

```bash
./scripts/build-image.sh
```

如已有 checkout：

```bash
VLLM_SOURCE=/path/to/vllm \
FLASHINFER_SOURCE=/path/to/flashinfer \
IMAGE=local/deepseek-v4-flash:vllm027-sm121 \
./scripts/build-image.sh
```

脚本拒绝 source commit 与版本契约不一致的 checkout。

## 4. 部署

官方 control：

```bash
MODEL_REPO=/same/path/on/both/nodes/official-cache \
./scripts/deploy-official.sh
```

NVFP4 + true FP4 MLA KV：

```bash
MODEL_REPO=/same/path/on/both/nodes/nvfp4-cache \
MODEL_REV=<immutable-snapshot> \
./scripts/deploy-nvfp4.sh
```

脚本先启动 rank 1，再启动 rank 0，并在 health wait 中执行 memory/swap guard。健康 endpoint：

```text
http://127.0.0.1:8888/health
```

持续保护应在独立终端运行：

```bash
./scripts/runtime-guard.sh
```

## 5. API smoke test

```bash
curl http://127.0.0.1:8888/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "deepseek-v4-flash",
    "messages": [{"role": "user", "content": "Explain tensor parallelism briefly."}],
    "max_tokens": 64,
    "temperature": 0
  }'
```

## 6. GuideLLM C6

复制 `benchmarks/guidellm-c6-template.json`，把 tokenizer snapshot 和 served model 替换为本次配置，然后运行与最新正式记录相同的 10-second warmup / 180-second measured window。

```bash
guidellm run \
  --config benchmarks/guidellm-c6.json \
  --disable-console-interactive
```

benchmark 期间必须保持 `runtime-guard.sh` 存活。冷启动后的第一次运行可能包含 TileLang/Triton/CuTeDSL JIT；只有日志证明 measured window 无新 compile 时，第二次相同 run 才可作为 hot result。

## 7. 停止与审计

```bash
./scripts/stop.sh
./scripts/status.sh
```

记录以下数据后再比较：checkpoint revision、image digest、两个 source label、KV bytes、prompt policy、target/draft backend、graph shapes、completed/in-flight/error request、output tok/s、output tok/target iteration、TPOT、TTFT 和 guard 最低内存。

## 8. 双 rank Nsight Systems 采集

只在模型已完成加载、JIT 和 CUDA graph warmup 后采集一个短 C6 batch。先在忽略提交的
`config/deployment.env` 中设置两台机器都存在的绝对路径：

```bash
NSYS_OUTPUT_DIR=/absolute/path/to/repo/profiles/nsys-official-c6
NSYS_HOST_ROOT=/opt/nvidia/nsight-systems/2025.3.2
NSYS_SESSION_PREFIX=officialc6
```

按 checkpoint 使用 `deploy-official.sh` 或 `deploy-nvfp4.sh`。此时 vLLM 位于 delayed
Nsight session 中，未调用 capture 脚本前不会记录应用 trace。先完成一次不采集的 API
warmup，再启动同步采集：

```bash
./scripts/nsys-capture.sh \
  python3 ./scripts/nsys-c6-workload.py \
  --model deepseek-v4-flash-0731-vllm027 \
  --concurrency 6 \
  --max-tokens 64 \
  --output profiles/nsys-official-c6/workload.json
```

脚本同时启动两端应用 CUDA/NVTX trace、1 kHz host GPU metrics，并在 workload 前后读取
RDMA port、mlx5 hardware 和 netdev sysfs counter。两端 `sudo -n nsys` 必须已配置；原始
`.nsys-rep`/SQLite 很大，只保存在 git 忽略的 `profiles/`。归一化结果见
[`data/nsight-summary.csv`](../data/nsight-summary.csv) 和
[双 rank Nsight 报告](nsight-profile.zh-CN.md)。

正常部署必须将 `NSYS_OUTPUT_DIR` 留空，以免引入 profiler 开销。

## 9. 原版 vLLM nightly 诊断基线

此脚本用于判断 stock nightly 的兼容边界，不是第三种受支持部署 profile。它仍会执行相同的
镜像一致性、110 GiB 启动门槛、108 GiB no-swap cgroup、6 GiB KV 和 C6 限制：

```bash
STOCK_NIGHTLY_IMAGE=vllm/vllm-openai:nightly \
STOCK_MTP_NUM_TOKENS=0 \
  ./scripts/deploy-stock-nightly.sh
```

当前缓存版本的 MTP-5 DSpark sparse-MLA warmup 会失败，所以可复现的 stock 性能基线必须
显式设为 `STOCK_MTP_NUM_TOKENS=0`。它会由 `auto` 选择 DeepGEMM MXFP4；不要把其
target-only 吞吐与适配版 MTP-5 的差值解释成单个 MoE backend 的因果收益。精确结果见
[`data/stock-nightly-summary.csv`](../data/stock-nightly-summary.csv)。
