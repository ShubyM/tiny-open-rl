FROM vllm/vllm-openai:v0.32.0

# Dependencies first and tinyrl last, so a code change rebuilds, pushes and pulls only the small layers at the end.
# vLLM's image pins torch and transformers, so install tinyrl's other dependencies and tinyrl itself without upgrading them
RUN pip install --no-cache-dir "peft>=0.15" "redis>=7.1" fsspec gcsfs
# Qwen3.5's linear-attention layers fall back to a slow reference implementation in transformers without these.
# Prebuilt for the image's CUDA 13.0 and torch 2.13 on the Astral GPU index, so nothing compiles here
RUN pip install --no-cache-dir --no-deps causal-conv1d==1.7.0+cu.13.0.torch.2.13 flash-linear-attention fla-core \
  --extra-index-url https://wheels.astral.sh/simple/cu130/
RUN pip install --no-cache-dir --no-deps liger-kernel \
  "https://github.com/Dao-AILab/flash-attention/archive/f63c472c4dc19237c0635d6a629df065ff64796e.tar.gz#subdirectory=flash_attn/cute"

WORKDIR /app
COPY pyproject.toml README.md ./
COPY tinyrl tinyrl
RUN pip install --no-cache-dir --no-deps .

# FlashAttention 4 compiles a kernel the first time a process uses it, unless it finds it in this cache
ENV FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1 FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=/root/.cache/flash-attention-cute

ENTRYPOINT []
CMD ["python3", "-m", "tinyrl.control.node_daemon"]
