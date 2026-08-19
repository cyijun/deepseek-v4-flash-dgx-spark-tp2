# 完整实验过程

机器可读主记录为 [`data/attempts.json`](../data/attempts.json)。本页按决策顺序说明每条路线解决了什么、为何保留或放弃。

## 1. 从 mock 结构到完整权重

最初使用 DeepSeek-V4-shaped 小模型验证 loader、tensor naming、256 routed experts、shared expert、MTP layer 和 NVFP4 metadata。mock 适合快速触发缺少的 operator 或 dispatch，但无法模拟完整模型内存、跨层 graph capture、TP collective 或 speculative acceptance。

因此所有部署和性能结论最终回到两端 cache 中相同 revision 的 48-shard checkpoint。仓库不分发权重。

## 2. 真 NVFP4 MLA KV 跑通

vLLM fork 新增 288 B/token 的 packed row：512 个四位 KV value 占 256 B，外加 32 B block scale。context cache write、page offset、sparse attention/dequant 和 CUDA graph 支持逐步补齐后，完整 NVFP4 checkpoint 能在 SM121 双机 TP=2 服务。

这也澄清了 Anemll 的命名：其 `nvfp4_ds_mla` 分支实际使用 584 B/token 的 FP8/BF16/UE8M0 layout。两者不能用名称直接作 FP4 kernel A/B。

## 3. 图捕获修复了 C2，但不是所有并发都受益

完整 12-token C2 target-verification shape 在 SM121 上不适合被 padding 到 24-token graph。将该 exact shape 强制 eager 后：

| C2 路径 | Aggregate output tok/s |
| --- | ---: |
| Earlier graph | 27.95 |
| Hybrid exact-eager | 41.58 |

提升为 48.7%。C1/C4/C6 graph path 没有发生同样变化，因此运行间的其他差值仍受 acceptance 和固定时间窗口影响。

## 4. W4A4 更慢的定位

NVFP4 权重保留四位 packed storage 后，比较 target expert execution：

| Target | Draft | C6 tok/s | Output tok/iter | Target iter |
| --- | --- | ---: | ---: | ---: |
| B12X W4A16 | Marlin | 80.27 | 2.493 | 186.4 ms |
| B12X W4A16 | B12X W4A16 | 69.43 | 2.442 | 211.0 ms |
| CUTLASS W4A4 | B12X W4A16 | 68.40 | 2.498 | 219.1 ms |

这个比较不是单一 datatype A/B，因为 kernel family 和 draft integration 不完全相同。但 routed `M=36` layer microbenchmark也得到同方向：W4A4 5.282 ms，W4A16 4.505 ms。

小 decode M 下，packed weight traffic 仍是主要流量；W4A4 额外承担 dynamic activation reduce、quantize/pack、scale 和 intermediate-buffer 成本，无法达到理论 FP4 peak。最终 NVFP4 preset 默认 target B12X W4A16。

## 5. 对齐 Anemll 的官方 control

为避免把 abliterated checkpoint acceptance、真 FP4 KV 和 runtime 差异混在一起，后续 control 对齐：

- 官方 FP8/MXFP4 checkpoint；
- target/draft 都是 B12X W4A16；
- physical FP8 DS-MLA KV；
- probabilistic MTP-5；
- `thinking=true, reasoning_effort=low`；
- TP=2、C6、256 prompt → 256 output；
- 10 s warmup + 60 s measured window。

| Runtime | C6 tok/s | Output tok/target iter | Estimated target iter | TPOT | TTFT |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current vLLM 0.27.1 hot | 97.20 | 2.706 | 167.0 ms | 61.98 ms | 743.58 ms |
| Anemll recorded reference | 108.18 | 2.873 | 159.3 ms | 55.67 ms | 704.29 ms |

当前吞吐低 10.15%；接受 token/iteration 低 5.82%，estimated iteration time 高 4.82%。这仍是跨 runtime/compiler 的 system comparison，不是单 commit A/B。

第一次 official control 只有 67.99 tok/s 且 TTFT 约 9.91 s，因为 measured window 内出现 TileLang/Triton/CuTeDSL compile。它保留为 cold-start 证据，不作为 steady baseline。

## 6. Draft 和 decode kernel 不是已证明的主慢点

official checkpoint layer-0 draft output 跨容器 all-close：最大绝对差 0.0078125，cosine similarity 0.9999766。routed `M=36` 时，当前 FlashInfer W4A16 4.250 ms，Anemll 4.281 ms。

重新按相同 decode grid 比较 profile 后：

| Component | Current µs/call | Anemll µs/call |
| --- | ---: | ---: |
| B12X main MoE | 1453.31 | 2142.97 |
| Sparse MLA decode | 48.57 | 69.55 |
| MHC post | 8.40 | 17.70 |
| Route post-prefix | 12.65 | 10.21 |

最初 aggregate profile 把当前 trace 的 43 个大 prefill call 与 Anemll decode-only trace 相加比较，误导了归因。完整表见 [`data/decode-profile.csv`](../data/decode-profile.csv)。

## 7. TileLang 0.1.9 路线被否决

TileLang 0.1.9 在 isolated MHC-post benchmark 中一度更低，但 full run 的 80.2 tok/s 被 JIT 污染，不能作 steady result。导出的 MHC cubin SASS 与 0.1.12 相同，retry startup 又触发 swap guard。因此不 pin 0.1.9，也不声称它更快。

## 8. 已实施的 route 优化

DeepSeek-V4 top-k kernel 原本已有八个内部 lane，只输出 routed top-6；router 随后对 shared expert ID 和 weight 各执行一次 `torch.cat`。368 个 B12X call 对应 trace 中 736 个 cat kernel。

[`7e64417`](https://github.com/cyijun/vllm/commit/7e64417e4af6d0079a0a9bfb999dd667f7263a58) 直接从 top-k kernel 写第七/第八 lane，其他 router shape仍走 generic fallback。

| Tokens | Top-k + cats | Fused output | Reduction |
| ---: | ---: | ---: | ---: |
| 6 | 30.42 µs | 12.38 µs | 59.3% |
| 30 | 30.34 µs | 12.34 µs | 59.3% |
| 36 | 30.65 µs | 12.38 µs | 59.6% |

七个 CUDA test 通过，ID/weight 与旧路径一致。理论上 target+draft iteration 可少约 0.8 ms，但 optimized full image startup 在 head swap 增长 654 MiB 时被保护器停止，尚无 TP=2 end-to-end 吞吐数字。

## 9. 最终保留与停止的路线

保留：TP=2、B12X W4A16、真实 NVFP4 MLA、official FP8 control、exact eager shape、route-pack prewarm、fused shared route、C6/6 GiB/guardrails。

停止或隔离：PP=2 部署、C128、10 GiB KV、TileLang 0.1.9 pin、把 cold/JIT run 当 steady result、把 Anemll `nvfp4_ds_mla` 当作物理 FP4 baseline。
