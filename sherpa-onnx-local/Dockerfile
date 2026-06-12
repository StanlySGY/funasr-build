FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1
ENV SHERPA_ONNX_HOST=0.0.0.0
ENV SHERPA_ONNX_PORT=10110
ENV SHERPA_ONNX_MODEL_DIR=/models
ENV SHERPA_ONNX_MODEL_TYPE=paraformer
ENV SHERPA_ONNX_PROVIDER=cpu
ENV SHERPA_ONNX_NUM_THREADS=4

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY sherpa_onnx_local_server.py /app/sherpa_onnx_local_server.py
COPY start_sherpa_onnx_local.sh /app/start_sherpa_onnx_local.sh
RUN chmod +x /app/start_sherpa_onnx_local.sh

EXPOSE 10110

CMD ["/app/start_sherpa_onnx_local.sh"]
