# ats-tailor in a container: Python, Tectonic, poppler. No TeX Live.
#
# The image is the engine; the repository is a bind mount. Nothing about one
# person's index, postings or runs is baked in, which is what lets the same
# image serve the TUI over HTTP, run one `tailor` and score an eval sweep.
#
# Tectonic instead of TeX Live is the whole reason this is a few hundred
# megabytes rather than a few thousand: one static binary that fetches the
# packages a document actually uses into a cache. That first fetch is the only
# network this image needs after the build — see HF_HUB_OFFLINE in the README
# for running it air-gapped afterwards.

FROM ghcr.io/astral-sh/uv:0.13.4 AS uv

FROM python:3.12-slim

ARG TECTONIC_VERSION=0.17.0
ENV DEBIAN_FRONTEND=noninteractive

# poppler-utils is not optional: `pdftotext` is how every check reads the PDF,
# and `pdftoppm` is what the TUI rasters a page with.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        poppler-utils \
    && rm -rf /var/lib/apt/lists/*

RUN set -eux; \
    case "$(dpkg --print-architecture)" in \
        amd64) target=x86_64-unknown-linux-musl ;; \
        arm64) target=aarch64-unknown-linux-musl ;; \
        *) echo "unsupported architecture: $(dpkg --print-architecture)" >&2; exit 1 ;; \
    esac; \
    curl -fsSL -o /tmp/tectonic.tar.gz \
        "https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%40${TECTONIC_VERSION}/tectonic-${TECTONIC_VERSION}-${target}.tar.gz"; \
    tar -xzf /tmp/tectonic.tar.gz -C /usr/local/bin tectonic; \
    rm -f /tmp/tectonic.tar.gz; \
    chmod +x /usr/local/bin/tectonic; \
    tectonic --version

COPY --from=uv /uv /uvx /usr/local/bin/

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PATH=/opt/venv/bin:$PATH

WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY packages/ ./packages/
COPY ats_tailor/ ./ats_tailor/
COPY examples/ ./examples/
# The lock is authoritative for every dependency but one. `sentence-transformers`
# pulls PyTorch, and PyPI's torch is the CUDA build on Linux — on both
# architectures, since NVIDIA publishes aarch64 wheels too. That is 3.5 GB of
# CUDA runtime in an image that will never see a GPU (on macOS there is none to
# see; the provider talks to an Ollama on the host instead). The *same* torch
# version is reinstalled from PyTorch's CPU index and the GPU-only packages are
# dropped. `uv cache clean` matters as much: without it the wheels uv
# downloaded — the CUDA ones included — stay in this layer forever.
RUN set -eux; \
    uv sync --frozen --no-dev --extra tui; \
    torch_version="$(uv pip show --python "$UV_PROJECT_ENVIRONMENT" torch \
        | awk '/^Version:/ {print $2}' | cut -d+ -f1)"; \
    uv pip install --python "$UV_PROJECT_ENVIRONMENT" \
        --index-url https://download.pytorch.org/whl/cpu \
        --reinstall-package torch "torch==${torch_version}"; \
    uv pip list --python "$UV_PROJECT_ENVIRONMENT" --format json \
      | python -c 'import json,sys; print("\n".join(p["name"] for p in json.load(sys.stdin) if p["name"].startswith(("nvidia-","cuda-")) or p["name"] in {"triton","pytorch-triton"}))' \
      > /tmp/gpu-only.txt; \
    if [ -s /tmp/gpu-only.txt ]; then \
        xargs -a /tmp/gpu-only.txt uv pip uninstall --python "$UV_PROJECT_ENVIRONMENT"; \
    fi; \
    rm -f /tmp/gpu-only.txt; \
    python -c "import torch, sentence_transformers; print(torch.__version__)"; \
    uv cache clean; \
    find "$UV_PROJECT_ENVIRONMENT" -name '__pycache__' -prune -exec rm -rf {} +

# The caches are the two things worth keeping between runs: the embedding model
# (hundreds of megabytes, downloaded once) and Tectonic's package bundle. Both
# are named volumes in the compose file. World-writable because the repository
# is a bind mount and the container may be told to run as the host's uid so it
# can write into it.
ENV HF_HOME=/cache/hf \
    TECTONIC_CACHE_DIR=/cache/tectonic \
    XDG_CACHE_HOME=/cache/xdg \
    MPLCONFIGDIR=/cache/matplotlib \
    ATS_LATEX_ENGINE=tectonic \
    ATS_ENGINE=v2
RUN mkdir -p /cache/hf /cache/tectonic /cache/xdg /cache/matplotlib /repo \
    && chmod -R 0777 /cache /repo

RUN useradd --create-home --uid 10001 --shell /bin/bash ats
USER 10001
WORKDIR /repo

EXPOSE 8080
ENTRYPOINT ["ats-tailor"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8080"]
