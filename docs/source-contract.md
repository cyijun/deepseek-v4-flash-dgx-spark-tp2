# 源码版本契约

## 固定入口

构建使用以下 immutable refs：

| 组件 | 固定版本 | 链接 |
| --- | --- | --- |
| vLLM binary base | `vllm/vllm-openai:v0.27.1@sha256:0a51…967` | [Docker Hub manifest](https://hub.docker.com/r/vllm/vllm-openai) |
| vLLM source overlay | `2db20513ab9d73e61aecabb3ab83e8f60644718e` | [commit](https://github.com/cyijun/vllm/commit/2db20513ab9d73e61aecabb3ab83e8f60644718e) · [branch](https://github.com/cyijun/vllm/tree/feat/deepseek-v4-nvfp4-ds-mla) |
| FlashInfer source/JIT overlay | `6398edbbc6796d81781bd54827be860b65d8f38b` | [commit](https://github.com/cyijun/flashinfer/commit/6398edbbc6796d81781bd54827be860b65d8f38b) · [branch](https://github.com/cyijun/flashinfer/tree/agent/apply-swiglu-limit-to-silu-b12x) |
| B12X dependency stack | `1.2.4` | installed from PyPI during image build |
| B12X runtime package | `0.15.3` | copied from `ghcr.io/anemll/dspark-vllm-gx10:0.1.1` after installing the dependency stack |

Action 不使用 branch tip 作为默认输入。branch 链接用于浏览，完整 commit 才是构建身份。

## vLLM 适配链

基线是 grafted `v0.27.1` commit `6e448d0`。以下按历史顺序列出 fork 上的定制提交；其中明确标记的 revert 是试验过程的一部分，不是当前净功能。

| Commit | 作用 | 当前状态 |
| --- | --- | --- |
| [`a0ed013`](https://github.com/cyijun/vllm/commit/a0ed013b4f5b27f1d9da94fa30ca2f3dcde0f897) | SM12x DeepSeek-V4 NVFP4 MLA 基础支持 | 保留 |
| [`7f4d2fa`](https://github.com/cyijun/vllm/commit/7f4d2fa80ab931b9782039ed0d219fc6d2f736ea) | DeepGEMM 关闭时选择 SM12x fallback | 保留 |
| [`1aac180`](https://github.com/cyijun/vllm/commit/1aac1809f7d197012362b18d4abbccafa7f11674) → [`0d166be`](https://github.com/cyijun/vllm/commit/0d166bedb05a198d51ebcd643374381030472cc4) | 第一次共享 eager B12X workspace 尝试及回退 | 已回退 |
| [`0a3ff67`](https://github.com/cyijun/vllm/commit/0a3ff67b691a5d64fd629eb45be6f9dcc4268cd1) | 将 NVFP4 dtype 传播到 compressor metadata | 保留 |
| [`428121a`](https://github.com/cyijun/vllm/commit/428121a1f2da311c319420b21f42ff962294ac50) + [`1b5d460`](https://github.com/cyijun/vllm/commit/1b5d4600beacca361cd06453d96f50e42ddff95c) | 跨 MoE layer 共享并同步 workspace 的第二次尝试 | 后续回退 |
| [`ca49ebb`](https://github.com/cyijun/vllm/commit/ca49ebb57de9cdee3778804776bd01c9b1b4007a) + [`705931a`](https://github.com/cyijun/vllm/commit/705931a2b85f4a309517a04ed9f492d51f962e44) | 回退同步和跨 layer workspace | 回退提交保留在历史 |
| [`c261e99`](https://github.com/cyijun/vllm/commit/c261e99652e8aa4ab10c05f31ecf8514a1075922) | 清理 FlashInfer B12X padding route | 保留 |
| [`cb7bb81`](https://github.com/cyijun/vllm/commit/cb7bb81afdbaa45a35f935206946f2cedf56badd) | DSpark context cache 写入真实 packed FP4 layout | 保留 |
| [`3eaf5e9`](https://github.com/cyijun/vllm/commit/3eaf5e9611e55b8a03167a3bfa7c52db3e0179f9) | 保留 native MTP expert quantization | 保留 |
| [`a7ffe76`](https://github.com/cyijun/vllm/commit/a7ffe76beee4b06f543f833c89679d9b69450027) | 共享 eager B12X runtime buffer | 保留 |
| [`cc50448`](https://github.com/cyijun/vllm/commit/cc50448ee0e66bd378c34b73e8cacef937ee18cf) | 保留 source B12X scale view | 保留 |
| [`9e897fd`](https://github.com/cyijun/vllm/commit/9e897fd0d8e30c9864984db83d32425f09dbab48) | 融合 DeepSeek-V4 NVFP4 sparse MLA | 保留 |
| [`663671a`](https://github.com/cyijun/vllm/commit/663671a1eb0e7e6811933035aa0069c3ffa8a2a9) | packed KV page offset 使用 64-bit | 保留 |
| [`0b385de`](https://github.com/cyijun/vllm/commit/0b385de77226e0f23481287b4101a79dee56cac1) | B12X workspace graph-safe | 保留 |
| [`5729130`](https://github.com/cyijun/vllm/commit/57291305af582525c9a7143e0b54e596a89448c4) | SM12x MXFP4 experts 使用 B12X | 保留 |
| [`4204c66`](https://github.com/cyijun/vllm/commit/4204c6622b4e8eea197fd7d55c438a4e3c288bd1) | NVFP4 MLA CUDA graph 支持 | 保留 |
| [`b06ffc9`](https://github.com/cyijun/vllm/commit/b06ffc910cb651612d5dfc26a72074423341b902) | native MXFP4 draft experts 使用 B12X W4A16 | 保留 |
| [`2cdf91c`](https://github.com/cyijun/vllm/commit/2cdf91c357fedeef8dae4152b2d6b4597254f678) | 避开 B12X block-FP8 M12 graph fault | 保留 |
| [`c2141dd`](https://github.com/cyijun/vllm/commit/c2141dd49c8b88eaa4ff480f5d5afd30369d6775) | 支持指定 exact eager CUDA-graph token size | 保留 |
| [`3d548d4`](https://github.com/cyijun/vllm/commit/3d548d4128b44db0aedea205f01ea637faabb439) | NVFP4 checkpoint 的 packed B12X W4A16 execution mode | 保留 |
| [`8c0ad8a`](https://github.com/cyijun/vllm/commit/8c0ad8a7a0eb282364c33afcd4d95778d7037eba) | DSpark pipeline-parallel bring-up | 源码保留；部署不启用 PP |
| [`0ff333e`](https://github.com/cyijun/vllm/commit/0ff333eaa63beedd994432798a551906dbc4a758) | FlashMLA 支持 B12X `wo_a` | 保留 |
| [`d337c70`](https://github.com/cyijun/vllm/commit/d337c7046ea9670093e5e9ffe3179fc33b8e287c) | CUDA graph capture 前预热 B12X route packing | 保留 |
| [`7e64417`](https://github.com/cyijun/vllm/commit/7e64417e4af6d0079a0a9bfb999dd667f7263a58) | top-k kernel 直接写 shared-expert slots，移除两次 cat | 保留；GPU unit validated |
| [`e28d323`](https://github.com/cyijun/vllm/commit/e28d323) | 避免 W4A16 route masking 的重复工作 | 保留 |
| [`8c00e28`](https://github.com/cyijun/vllm/commit/8c00e28) → [`42db932`](https://github.com/cyijun/vllm/commit/42db932) | single-chunk speculative orchestration 尝试及回退 | 已回退 |
| [`f28e538`](https://github.com/cyijun/vllm/commit/f28e538) | 增加 planned B12X MXFP4 execution route | 保留 |
| [`5235aaf`](https://github.com/cyijun/vllm/commit/5235aaf) + [`3470fbc`](https://github.com/cyijun/vllm/commit/3470fbc) | 修正 B12X packed route count 绑定并分离 scratch | 保留 |
| [`cfdb952`](https://github.com/cyijun/vllm/commit/cfdb952) | 接受 B12X ultrawide decode tile | 保留 |
| [`8d87d34`](https://github.com/cyijun/vllm/commit/8d87d34) + [`416e114`](https://github.com/cyijun/vllm/commit/416e114) | packed standalone MXFP4 weight 与 legacy B12X runtime 支持 | 保留；正式对照使用 |
| [`023c7ab`](https://github.com/cyijun/vllm/commit/023c7ab) | legacy B12X prefill launch 按 1024 routed rows 分块 | 保留；修复 M=1052 性能 cliff |
| [`2db2051`](https://github.com/cyijun/vllm/commit/2db20513ab9d73e61aecabb3ab83e8f60644718e) | legacy B12X decode tile 的有界 selector override | 保留；C6 使用 128×64/128-thread |

## FlashInfer 适配链

| Commit | 作用 | 当前状态 |
| --- | --- | --- |
| [`43c966a`](https://github.com/cyijun/flashinfer/commit/43c966ab57947d9a74413ccdce682df2a9035115) | B12X SM12x 的 SiLU 应用 `swiglu_limit` clamp | 保留 |
| [`bb1dcfd`](https://github.com/cyijun/flashinfer/commit/bb1dcfd4a688935478ed94a72fc15e12348bd31a) | caller-owned B12X output | 保留 |
| [`a8b8b3e`](https://github.com/cyijun/flashinfer/commit/a8b8b3e747772af58f3cb070d84b151367087c1d) | W4A16 dispatch 保留 SiLU clamp | 保留 |
| [`e669583`](https://github.com/cyijun/flashinfer/commit/e6695831f237dc6bc022ddf29581ef272c862d7a) | W4A16 tensor-core decode tile 保护 | 保留 |
| [`6398edb`](https://github.com/cyijun/flashinfer/commit/6398edbbc6796d81781bd54827be860b65d8f38b) | 按 scale format 限制 W4A16 wide FC2 tile | 保留，镜像固定点 |

SiLU clamp 的数学语义是：

```text
gate = min(gate, limit)
up   = clamp(up, -limit, limit)
out  = silu(gate) * up
```

它是 generic gated activation 行为，不是 DeepSeek-specific API；`limit=None` 仍保留原 optimized SiLU fast path。

## 更新规则

更新任一 fork 时必须同时：

1. 修改 `config/versions.env` 和 Action input default；
2. 重新构建 image，并核对 OCI source labels；
3. 在两个节点拉取同一 digest；
4. 跑 source-contract、GPU correctness 和 bounded-memory startup；
5. 若有性能结论，重新生成 report artifact，而不是覆盖历史基准含义。

## 基准镜像说明

正式 180 秒 C6 对照运行在用户新缓存的 ARM64 nightly 依赖镜像上，其 OCI revision 为
`acb0f1dcdb668d90bbbf50e57552d2f6f0987c87`，再 overlay 上述基于 v0.27.1 的定制源码。
这个本地镜像没有可回溯的公开 RepoDigest，因此发布 Action 仍以可复现的 v0.27.1 manifest
作为默认 binary base。报告中的正式数值只对应 nightly 依赖镜像；用 Action 生成的新镜像在发布前
必须重新跑同一 C6 correctness/performance gate，不能把历史结果直接继承给新 digest。
