ARG BASE_IMAGE=vllm/vllm-openai:v0.27.1@sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967
FROM ${BASE_IMAGE}

ARG B12X_VERSION=1.2.4
ARG VLLM_SOURCE_COMMIT
ARG FLASHINFER_SOURCE_COMMIT

LABEL org.opencontainers.image.title="DeepSeek V4 Flash TP=2 runtime for DGX Spark" \
      org.opencontainers.image.source="https://github.com/cyijun/deepseek-v4-flash-dgx-spark-tp2" \
      org.opencontainers.image.description="vLLM 0.27.1 with linked DeepSeek V4, NVFP4 DS-MLA, and FlashInfer B12X SM121 adaptations" \
      io.deepseek-v4-dgx-spark.vllm-source-commit="${VLLM_SOURCE_COMMIT}" \
      io.deepseek-v4-dgx-spark.flashinfer-source-commit="${FLASHINFER_SOURCE_COMMIT}"

# The validated SM121 B12X stack uses this b12x/CUTLASS DSL pairing.
RUN uv pip install --system \
      --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
      "b12x==${B12X_VERSION}"

# Keep binary CUDA extensions from the v0.27.1 image and overlay the pinned
# Python/JIT sources from the linked forks.
COPY vllm/ /usr/local/lib/python3.12/dist-packages/vllm/
COPY flashinfer/cute_dsl/fp4_common.py /usr/local/lib/python3.12/dist-packages/flashinfer/cute_dsl/fp4_common.py
COPY flashinfer/cute_dsl/utils.py /usr/local/lib/python3.12/dist-packages/flashinfer/cute_dsl/utils.py
COPY flashinfer/autotuner/ /usr/local/lib/python3.12/dist-packages/flashinfer/autotuner/
COPY flashinfer/fused_moe/cute_dsl/b12x_moe.py /usr/local/lib/python3.12/dist-packages/flashinfer/fused_moe/cute_dsl/b12x_moe.py
COPY flashinfer/fused_moe/cute_dsl/blackwell_sm12x/ /usr/local/lib/python3.12/dist-packages/flashinfer/fused_moe/cute_dsl/blackwell_sm12x/
COPY flashinfer/mla/_sparse_mla_sm120.py /usr/local/lib/python3.12/dist-packages/flashinfer/mla/_sparse_mla_sm120.py
COPY flashinfer_data/csrc/ /usr/local/lib/python3.12/dist-packages/flashinfer/data/csrc/
COPY flashinfer_data/include/ /usr/local/lib/python3.12/dist-packages/flashinfer/data/include/

# The base wheel contains an older AOT sparse-MLA module. Preserve it for
# diagnosis but force this operator to JIT from the matching pinned sources.
RUN sparse_mla_aot=/usr/local/lib/python3.12/dist-packages/flashinfer_jit_cache/jit_cache/sparse_mla_sm120/sparse_mla_sm120.so \
    && if [ -f "${sparse_mla_aot}" ]; then \
         mv "${sparse_mla_aot}" "${sparse_mla_aot}.base-wheel"; \
       fi

ENV FLASHINFER_DISABLE_VERSION_CHECK=1 \
    TORCH_CUDA_ARCH_LIST=12.1a \
    FLASHINFER_CUDA_ARCH_LIST=12.1a \
    CUTE_DSL_ARCH=sm_121a

RUN python3 -m compileall -q \
      /usr/local/lib/python3.12/dist-packages/vllm \
      /usr/local/lib/python3.12/dist-packages/flashinfer/autotuner \
      /usr/local/lib/python3.12/dist-packages/flashinfer/fused_moe/cute_dsl \
      /usr/local/lib/python3.12/dist-packages/flashinfer/mla \
    && python3 -c "import b12x, flashinfer, importlib.metadata, vllm; from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_direct_micro_kernel import build_direct_micro_kernel; print(vllm.__version__, flashinfer.__version__, importlib.metadata.version('b12x'), build_direct_micro_kernel.__name__)"
