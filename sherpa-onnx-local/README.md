# sherpa-onnx 本地 ASR 服务

这是一个独立的 `sherpa-onnx` HTTP/SSE 封装服务，用来在 ARM CPU 上对比 FunASR-Nano 的中文语音转文字速度。

默认端口：`10110`

## 推荐测试模型

优先测试 sherpa-onnx 的中文 Paraformer 离线模型，例如包含以下文件的模型目录：

```text
model.int8.onnx
tokens.txt
```

把模型放到服务器：

```bash
mkdir -p /data/maas/sgy_arm/sherpa-onnx-models
```

如果你的文件名不是 `model.int8.onnx` / `tokens.txt`，可以在 `docker-compose.sherpa-onnx.yml` 里显式设置：

```yaml
- SHERPA_ONNX_PARA_MODEL=/models/your-model.onnx
- SHERPA_ONNX_TOKENS=/models/your-tokens.txt
```

也支持 transducer 模型：

```yaml
- SHERPA_ONNX_MODEL_TYPE=transducer
- SHERPA_ONNX_ENCODER=/models/encoder.onnx
- SHERPA_ONNX_DECODER=/models/decoder.onnx
- SHERPA_ONNX_JOINER=/models/joiner.onnx
- SHERPA_ONNX_TOKENS=/models/tokens.txt
```

## 启动

在仓库根目录执行：

```bash
docker compose -f docker-compose.sherpa-onnx.yml up -d --build
curl -s http://127.0.0.1:10110/health
```

## OpenAI 风格接口

```bash
/usr/bin/time -f 'elapsed=%E' curl -sS \
  -F "file=@./test.wav" \
  -F "model=sherpa-onnx" \
  http://127.0.0.1:10110/v1/audio/transcriptions
```

## SSE 接口

```bash
/usr/bin/time -f 'elapsed=%E' curl -N -sS \
  -X POST http://127.0.0.1:10110/asr/file-sse \
  -F "file=@./test.wav" \
  -F "mode=offline"
```

预期输出形态：

```text
event: final
data: {"mode": "sherpa-onnx", "text": "...", "wav_name": "test.wav", "is_final": true, "provider": "sherpa-onnx"}

event: done
data: {}
```

## 关键环境变量

```text
SHERPA_ONNX_MODEL_DIR=/models
SHERPA_ONNX_MODEL_TYPE=paraformer
SHERPA_ONNX_PARA_MODEL=/models/model.int8.onnx
SHERPA_ONNX_TOKENS=/models/tokens.txt
SHERPA_ONNX_NUM_THREADS=4
SHERPA_ONNX_PROVIDER=cpu
```

## 对比方式

同一个 `test.wav`：

```bash
# FunASR-Nano
/usr/bin/time -f 'elapsed=%E' curl -N -sS \
  -X POST http://127.0.0.1:10098/asr/file-sse \
  -F "file=@./test.wav" \
  -F "mode=online"

# sherpa-onnx
/usr/bin/time -f 'elapsed=%E' curl -N -sS \
  -X POST http://127.0.0.1:10110/asr/file-sse \
  -F "file=@./test.wav" \
  -F "mode=offline"
```

如果 `sherpa-onnx` 明显快，说明 FunASR-Nano 的 PyTorch 推理路径不适合当前 ARM CPU。
