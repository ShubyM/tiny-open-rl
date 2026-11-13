# The LAB recipe as an image, to run in the cluster: LAB with its grading venv, open-rl's sandbox clients, the Agent Substrate
# client, and this folder. Everything up to the recipe is open-rl's examples/harvey_labs/Dockerfile.
# The build folder holds open-rl's examples/ as examples/, sandboxenv-rl's clients/python as ate-env-client/, and this folder
# as tinyrl_lab/ (README.md). The Substrate client is copied in, as its repository is not open to every build.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl git pandoc \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uv/bin/uv
ENV PATH="/uv/bin:$PATH"

ARG LAB_REPO=https://github.com/ShubyM/harvey-labs
ARG LAB_REF=401bf7f0448044cb112556d26492652d56e8c317
RUN git clone --filter=blob:none "$LAB_REPO" /app/harvey_labs/harvey-labs \
    && git -C /app/harvey_labs/harvey-labs checkout "$LAB_REF"
RUN --mount=type=cache,target=/root/.cache/uv \
    cd /app/harvey_labs/harvey-labs && uv sync --frozen

WORKDIR /app
COPY examples/pyproject.toml examples/uv.lock examples/README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --extra agent-sandbox

COPY ate-env-client /src/ate-env-client
RUN uv pip install --python /app/.venv/bin/python --no-deps /src/ate-env-client

COPY examples/common ./common
COPY examples/harvey_labs ./harvey_labs
COPY tinyrl_lab ./tinyrl_lab

ENV LAB_ROOT=/app/harvey_labs/harvey-labs
ENV PYTHONPATH=/app:$LAB_ROOT
WORKDIR /app/tinyrl_lab
ENTRYPOINT ["/app/.venv/bin/python", "lab_rl.py"]
