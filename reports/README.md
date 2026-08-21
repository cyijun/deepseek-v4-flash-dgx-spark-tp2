# Report artifacts

- `flow-comparison-report.html` is the primary, self-contained technical report. It embeds its data and reader runtime and can be opened offline.
- `flow-comparison-report-artifact.json` is the canonical source artifact used to generate the HTML.

The report includes exact benchmark and profile tables but does not embed raw model weights, profiler traces, host logs, local absolute paths, credentials, or container inspect dumps. Machine-readable summaries used for repository-level review also live under `../data/`.

The chart map is:

| Section | Question | Form | Dataset | Claim |
| --- | --- | --- | --- | --- |
| Official control | Does the bounded legacy-B12X optimization reach Anemll's hot C6 band? | Four-category bar | `official_control` | 106.07 Anemll; 104.35 chunk-only; 106.39/104.94 optimized repeats |
| Post-profile steady lanes | What throughput is available from each lane that actually starts? | Three-category bar | `steady_c6` | 124.86 adapted official MTP-5; 106.42 adapted NVFP4 MTP-5; 85.84 stock official target-only |
| Earlier format study | How does the true-FP4 deployment scale by concurrency? | Grouped bars | `throughput_long` | Both acceptance and iteration rate contributed to the older gap |

All visuals use discrete comparison bars because the evidence consists of benchmark anchor runs, not a sufficiently dense time series. Exact two-rank Nsight and stock-compatibility values remain in report tables because audit lookup, rather than visual shape, is their primary purpose.
