FROM vllm/vllm-openai:v0.32.0

WORKDIR /app
COPY pyproject.toml README.md ./
COPY tinyrl tinyrl
# vLLM's image pins torch and transformers, so install tinyrl's other dependencies and tinyrl itself without upgrading them
RUN pip install --no-cache-dir "peft>=0.15" "redis>=7.1" fsspec gcsfs && pip install --no-cache-dir --no-deps .

ENTRYPOINT []
CMD ["python3", "-m", "tinyrl.node_daemon"]
