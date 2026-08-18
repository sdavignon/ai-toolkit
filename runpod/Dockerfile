FROM nvidia/cuda:13.0.3-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    AITK_ROOT=/app/ai-toolkit \
    RUNPOD_VOLUME_PATH=/runpod-volume \
    HF_HOME=/runpod-volume/huggingface \
    TRANSFORMERS_CACHE=/runpod-volume/huggingface/transformers \
    TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0 10.0 12.0"

RUN apt-get update && apt-get install --no-install-recommends -y \
    git curl build-essential cmake wget python3.12 python3-pip python3-dev \
    python3-setuptools python3-wheel python3-venv ffmpeg python3-opencv rsync \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /app/ai-toolkit

COPY requirements.txt requirements_base.txt ./

RUN pip install --no-cache-dir --break-system-packages \
      torch==2.13.0 torchvision==0.28.0 torchaudio==2.11.0 \
      --index-url https://download.pytorch.org/whl/cu130 \
    && pip install --no-cache-dir --break-system-packages -r requirements.txt \
    && pip install --no-cache-dir --break-system-packages setuptools==69.5.1 runpod pyyaml \
    && pip install --no-cache-dir --break-system-packages -U \
      torchcodec==0.15.0 \
      natten==0.21.7+torch2130cu130 --find-links https://whl.natten.org \
      https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.47/flash_attn-2.8.3+cu130torch2.13-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl

COPY . /app/ai-toolkit

CMD ["python", "/app/ai-toolkit/runpod/handler.py"]
