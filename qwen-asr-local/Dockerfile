FROM python:3.12-slim-bookworm

ARG DEBIAN_FRONTEND=noninteractive

ENV PYTHONUNBUFFERED=1
ENV QWEN_ASR_MODEL=Qwen/Qwen3-ASR-0.6B
ENV QWEN_ASR_DEVICE=cpu
ENV QWEN_ASR_BACKEND=transformers
ENV QWEN_ASR_HOST=0.0.0.0
ENV QWEN_ASR_PORT=10100
ENV HF_HOME=/models/huggingface
ENV MODELSCOPE_CACHE=/models/modelscope

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY qwen_asr_local_server.py /app/qwen_asr_local_server.py
COPY start_qwen_asr_local.sh /app/start_qwen_asr_local.sh
RUN chmod +x /app/start_qwen_asr_local.sh

EXPOSE 10100

CMD ["/app/start_qwen_asr_local.sh"]
