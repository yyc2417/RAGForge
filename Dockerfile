FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    API_HOST=0.0.0.0 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# torch CPU 运行时依赖；避免 linux wheel 缺失 libgomp
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# uv：使用 lockfile 安装生产依赖；跳过 CUDA 版 torch 及其依赖，随后装 CPU wheel
RUN pip install --no-cache-dir uv
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev \
      --no-install-package torch \
      --no-install-package triton \
      --no-install-package nvidia-cublas \
      --no-install-package nvidia-cuda-cupti \
      --no-install-package nvidia-cuda-nvrtc \
      --no-install-package nvidia-cuda-runtime \
      --no-install-package nvidia-cudnn-cu13 \
      --no-install-package nvidia-cufft \
      --no-install-package nvidia-cufile \
      --no-install-package nvidia-curand \
      --no-install-package nvidia-cusolver \
      --no-install-package nvidia-cusparse \
      --no-install-package nvidia-cusparselt-cu13 \
      --no-install-package nvidia-nccl-cu13 \
      --no-install-package nvidia-nvjitlink \
      --no-install-package nvidia-nvshmem-cu13 \
      --no-install-package nvidia-nvtx
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python .venv/bin/python \
      --index-url https://download.pytorch.org/whl/cpu torch==2.12.0

# 预取默认 embedding 模型：src/__init__ 默认 HF 离线，缓存为空会导致启动失败
RUN .venv/bin/python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY . .

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

CMD [".venv/bin/python", "-m", "src.main", "--serve"]
