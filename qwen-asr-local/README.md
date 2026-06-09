# 本地 Qwen-ASR 服务（ARM CPU PoC）

这个目录提供一个最小本地 Qwen-ASR HTTP 服务，用于在当前项目里验证 `Qwen/Qwen3-ASR-0.6B` 是否能在 ARM CPU 环境跑通。

## 接口

服务暴露 OpenAI 风格接口：

- `GET /health`
- `POST /v1/audio/transcriptions`

`funasr-sse-adapter` 通过以下环境变量切到该服务：

```bash
QWEN_ASR_API_STYLE=transcriptions
QWEN_ASR_BASE_URL=http://127.0.0.1:10100/v1
QWEN_ASR_MODEL=/models/Qwen3-ASR-0.6B
QWEN_ASR_TIMEOUT_SEC=600
```

然后继续使用已有的项目接口：

- `POST /qwen-asr/file-sse`
- `POST /qwen-asr/base64-sse`
- `/qwen-asr/upload-wav` + `/qwen-asr/uploaded-file-session/{audio_id}` + `/qwen-asr/sse/{session_id}`

## 启动

在服务器项目根目录执行：

```bash
./start_qwen_asr_local_server.sh
```

这个脚本会用 `docker-compose.full.cpu.server.yml` 和 `docker-compose.qwen-asr-local.yml` 启动：

- `qwen-asr-local`：本地 Qwen-ASR 服务，端口 `10100`
- `funasr-sse-adapter`：重新加载 Qwen-ASR 本地服务配置，端口 `10098`

`qwen-asr-local` 镜像基于本项目已构建的 `funasr-ws-cpu:latest`，避免服务器再次拉取外部 Python 基础镜像。

模型缓存目录默认挂载到：

```text
/data/maas/sgy_arm/qwen-asr-models
```

先执行项目根目录的 `./download_qwen_asr_model.sh` 下载模型到该目录；如果不下载，模型加载会尝试访问 HuggingFace。

## 测试

```bash
./download_qwen_asr_model.sh
curl http://127.0.0.1:10100/health
./test_qwen_asr_local_file_sse.sh /path/to/audio.wav
```

或直接请求适配层：

```bash
curl -N -X POST http://127.0.0.1:10098/qwen-asr/file-sse \
  -F "file=@example.wav" \
  -F "mode=online" \
  -F "audio_fs=16000" \
  -F "hotwords="
```

## 重要限制

- 这是 ARM CPU 本地部署 PoC，不是默认生产链路。
- 官方 Qwen3-ASR 的主推本地服务形态偏 vLLM/GPU；ARM CPU 需要实测确认依赖包、模型加载、内存和耗时。
- 本服务使用 `qwen-asr` 包的 `Qwen3ASRModel.from_pretrained(..., device_map="cpu")`，默认 `float32`，可通过 `QWEN_ASR_DTYPE` 调整。
- 当前服务按整段音频识别，SSE 返回 `final` 和 `done`，不是 FunASR 那种实时 `online` 小段结果。
- 如果模型加载失败，现有 FunASR `/asr/*` 不受影响；只影响 `/qwen-asr/*`。
