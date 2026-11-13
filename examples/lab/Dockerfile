# The LAB harness as an image, to run in the cluster: LAB with its grading venv, the harness's few packages, the Agent Substrate
# client, and this folder. The build folder is this folder with sandboxenv-rl's clients/python beside it as ate-env-client/
# (README.md): the client is copied in, as its repository is not open to every build.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl git pandoc \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ARG LAB_REPO=https://github.com/ShubyM/harvey-labs
ARG LAB_REF=401bf7f0448044cb112556d26492652d56e8c317
RUN git clone --filter=blob:none "$LAB_REPO" /lab && git -C /lab checkout "$LAB_REF"
# LAB's own venv, which its grader runs in
RUN --mount=type=cache,target=/root/.cache/uv cd /lab && uv sync --frozen

WORKDIR /app
COPY requirements.txt ./
COPY ate-env-client /src/ate-env-client
RUN --mount=type=cache,target=/root/.cache/uv uv venv .venv \
    && uv pip install --python .venv/bin/python --index-strategy unsafe-best-match -r requirements.txt \
    && uv pip install --python .venv/bin/python --no-deps /src/ate-env-client
COPY lab_rl.py substrate.py ./

# LAB's harness and sandbox modules are imported from its checkout
ENV LAB_ROOT=/lab PYTHONPATH=/lab PATH=/app/.venv/bin:$PATH
ENTRYPOINT ["python", "lab_rl.py"]
