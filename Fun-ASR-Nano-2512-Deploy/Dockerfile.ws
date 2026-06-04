FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_ROOT_USER_ACTION=ignore
ENV MODEL_PATH=/app/funasr-deploy/models/FunAudioLLM/Fun-ASR-Nano-2512
ENV DEVICE=cuda
ENV PORT=10095

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    ffmpeg \
    git \
    git-lfs \
    curl \
    build-essential \
    g++ \
    make \
    cmake \
    swig \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --upgrade pip
RUN python3 -m pip install --no-cache-dir \
    torch \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu121
RUN python3 -m pip install --no-cache-dir \
    funasr \
    modelscope \
    websockets \
    transformers \
    sentencepiece \
    protobuf \
    tqdm \
    requests \
    aiohttp \
    tiktoken

WORKDIR /app/funasr-deploy
COPY . /app/funasr-deploy

EXPOSE 10095

CMD python3 funasr_wss_server.py \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --asr_model "${MODEL_PATH}" \
    --asr_model_online "${MODEL_PATH}" \
    --device "${DEVICE}"
