# Fun-ASR-Nano-2512 Linux 部署指南

Fun-ASR-Nano-2512官方发布的内容有点多，部署起来问题还是比较多，本项目提供一个简化的部署方案。

本项目可在 Linux 服务器上部署 Fun-ASR-Nano-2512，并启动 WebSocket 服务提供外部调用，另外包含 SSE Adapter 和测试验证工具。

如果使用 ARM CPU 服务器，并且准备上传整个 `funasr-build` 文件夹，请直接看后面的“Docker 启动完整双服务（ARM CPU，整包上传）”。该方式不需要 NVIDIA 驱动。

在 NVIDIA 3090 显卡上部署，启动完成后占用2590MiB显存，有请求调用后上升到3858MiB显存占用，可以按这个来评估显存需求，目前官方明确说不支持FP16部署，显存只能占么这多了。

## 目录结构

上传本目录所有文件到服务器的 `/data/asr/` 目录：

- `install.sh`: 环境安装脚本
- `start_server.sh`: 启动 Fun-ASR WebSocket 服务脚本
- `funasr_wss_server.py`: WebSocket 服务主程序
- `download_model.py`: 模型下载脚本（安装时下载模型）
- `test_inference.py`: 本地推理测试脚本（验证环境）
- `funasr_wss_client.py`: 测试客户端（验证部署是否OK）
- `web_client`: Web 测试客户端目录，方便WEB页面测试（未实现VAD检测，仅用于测试流式识别）

## 部署步骤

### 0. 环境检查 (Pre-check)

在执行安装前，建议检查服务器的 CUDA 版本，以确保 PyTorch 版本匹配。

**检查命令**:

```bash
# 方法 1: 查看 NVCC 编译器版本 (推荐，查看实际安装的 Toolkit)
nvcc -V

# 方法 2: 查看 GPU 驱动状态 (右上角 CUDA Version 为驱动支持的最高版本)
nvidia-smi
```

- **CUDA 11.x**: 脚本默认安装 PyTorch (cu118)，直接运行即可。
- **CUDA 12.x**: 建议修改 `install.sh`，将 install torch 的命令改为仅 `pip install torch torchaudio` (通常会自动匹配最新 CUDA 12) 或指定 `--index-url .../cu121`。

**验证安装**:

```bash
python -c "import torch, torchaudio; print(f'Torch: {torch.__version__}, Audio: {torchaudio.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### 1. 安装环境

```bash
cd /data/asr
chmod +x install.sh start_server.sh
./install.sh
```

此步骤会创建 python 虚拟环境，并安装 pytorch, funasr 等依赖。

### 2. 下载模型

```bash
# 激活环境
source venv/bin/activate
# 下载模型
python download_model.py
```

**注意**: 该脚本会自动下载 `Fun-ASR-Nano-2512` 主模型以及其依赖的 `Qwen3-0.6B` 子模型，并自动将其放置在正确的子目录结构中。请耐心等待所有下载完成。
模型将保存在当前目录的 `models/` 文件夹下。

### 3. 测试本地推理 (可选)

```bash
python test_inference.py
```

用于验证 GPU 是否正常工作以及显存占用情况。

### 4. 启动服务

```bash
./start_server.sh
```

此脚本会调用 `funasr_wss_server.py` 启动服务，监听 `0.0.0.0:10095` 端口。

## 客户端连接

Java 客户端或测试脚本可以通过 WebSocket 连接：

- URL: `ws://<SERVER_IP>:10095`
- 协议: FunASR 协议

## 显存优化说明

- 暂无 (FP16 模式目前在部分环境下存在兼容性问题，暂不推荐开启)

## WebSocket 接口文档

服务端提供基于 WebSocket 的实时语音识别服务，完全兼容 FunASR 客户端协议。

### 1. 连接地址

- **URL**: `ws://<SERVER_IP>:10095`
- **协议**: WebSocket (Binary Frames)

### 2. 通信流程

整个识别过程包含三个阶段：**握手配置 -> 音频流传输 -> 结果接收**。

#### a. 握手配置 (First Message)

建立连接后，客户端发送的**第一帧**必须是 JSON 格式的配置信息：

```json
{
  "mode": "2pass",                   // 推荐使用 2pass (流式+离线修正) 或 online
  "chunk_size": [5, 10, 5],          // 分块大小配置 [编码器历史, 当前块, 编码器未来]
  "chunk_interval": 10,              // 发送间隔 (ms)
  "encoder_chunk_look_back": 4,      // 编码器回溯步数
  "decoder_chunk_look_back": 1,      // 解码器回溯步数
  "audio_fs": 16000,                 // 音频采样率 (必须是 16000)
  "wav_name": "demo",                // 音频标识
  "is_speaking": true,               // 标记开始说话
  "hotwords": "{\"阿里巴巴\": 20, \"达摩院\": 30}", // 热词配置 (可选)
  "itn": true                        // 开启逆文本标准化 (数字转汉字等)
}
```

> **自动兼容**: 如果客户端请求 `mode: "online"`，服务端会自动将其升级为 `mode: "2pass"`，以确保在流式结束后能触发离线修正并返回最终结果（防止部分客户端死等 is_final: true）。

#### b. 音频流传输 (Streaming)

- 配置帧发送后，客户端持续发送**二进制音频数据 (Binary Frame)**。
- 格式：PCM, 16000Hz, 16bit, 单声道。
- 建议分块发送，每块大小约 60ms - 100ms 的数据。

#### c. 结束信号 (End of Stream)

- 当用户停止说话时，客户端发送一帧 JSON 结束信号：
  
  ```json
  {"is_speaking": false}
  ```

### 3. 服务端响应格式

服务端会通过 WebSocket 持续返回 JSON 格式的识别结果。

#### 流式中间结果 (Variable)

当 `mode="online"` 或 `mode="2pass"` 时，服务端会实时返回当前识别片段：

```json
{
  "mode": "2pass-online",
  "text": "正在识别的内容",
  "wav_name": "demo",
  "is_final": false // 通常为 false，但当检测到语音结束(is_speaking: false)时的最后一帧可能为 true
}
```

#### 最终结果 (Final)

当一句话结束 (VAD 检测到静音) 或收到 `is_speaking: false` 后，服务端会进行离线修正，并返回最终结果：

```json
{
  "mode": "2pass-offline",
  "text": "最终识别的修正结果。",
  "wav_name": "demo",
  "is_final": true
}
```

> **注意**: 
> 
> 1. 为了防止客户端超时，即使离线识别结果为空（如误触 VAD），服务端也会发送一个 `text: ""` 且 `is_final: true` 的空包。
> 2. Java 客户端通常只处理 `is_final: true` 的消息。

## Web 测试客户端 (New)

本项目提供了一个轻量级的 Web 页面，用于快速验证 ASR 服务及其 VAD 效果。

### 1. 启动 Web 服务

```bash
cd deploy/asr/web_client
python serve_client.py
```

服务默认监听 `8000` 端口。

### 2. 访问测试

- **推荐 (本地)**: 直接访问 `http://localhost:8000`。
  - 浏览器会自动允许麦克风权限。
  - 页面中 WebSocket 地址填入远程服务器 IP 即可 (例如 `ws://10.11.x.x:10095`)。
- **高级 (远程)**: 如果浏览器和 Web 服务不在同一台机器，需访问 `http://<Web_Server_IP>:8000`。
  - **注意**: Chrome 默认禁止非 HTTPS 网页使用麦克风。
  - **解决**: 需配置 `chrome://flags/#unsafely-treat-insecure-origin-as-secure` 才能使用麦克风。

## 作者信息

- **作者**：凌封
- **来源**：[https://aibook.ren (AI全书)](https://aibook.ren)

## SSE Adapter

新增 `asr_sse_adapter.py` 用于把现有 FunASR WebSocket 流式 ASR 封装成 SSE 结果流。

说明：

- FunASR 底层仍然使用 WebSocket 双向流式协议。
- SSE Adapter 对外输出识别结果流。
- 音频上传和 SSE 返回是两条 HTTP 通道。
- 默认 FunASR WebSocket 后端地址：`ws://127.0.0.1:10095`。
- 默认 SSE Adapter 端口：`10097`。

### 安装 SSE 依赖

```bash
cd /home/sgy/work_/gd-dev/funasr-build/Fun-ASR-Nano-2512-Deploy
pip install -r requirements.sse.txt
```

### 启动 SSE Adapter

如果 FunASR WebSocket 服务在本机 `10095`：

```bash
./start_sse_adapter.sh
```

如果 FunASR WebSocket 服务在远程服务器：

```bash
FUNASR_BACKEND_WS=ws://172.16.100.26:10095 ./start_sse_adapter.sh
```

也可以指定 SSE Adapter 监听端口：

```bash
PORT=10097 FUNASR_BACKEND_WS=ws://172.16.100.26:10095 ./start_sse_adapter.sh
```

### Docker 启动 SSE Adapter

如果服务器只允许 Docker 启动，使用独立 compose 文件：

```bash
cd /path/to/Fun-ASR-Nano-2512-Deploy
docker compose -f docker-compose.sse.yml up -d --build
```

默认配置会让容器访问宿主机上的 FunASR WebSocket：

```text
ws://host.docker.internal:10095
```

这要求 FunASR WebSocket 服务已经在宿主机 `10095` 端口运行。

查看日志：

```bash
docker logs -f funasr-sse-adapter
```

健康检查：

```bash
curl http://服务器IP:10097/health
```

如果 FunASR WebSocket 不在同一台宿主机，修改 `docker-compose.sse.yml`：

```yaml
FUNASR_BACKEND_WS=ws://172.16.100.26:10095
```

### Docker 启动完整双服务（ARM CPU，整包上传）

如果你上传的是整个 `funasr-build` 文件夹，并且服务器是 ARM CPU 架构，优先使用根目录的 `docker-compose.full.cpu.server.yml`。这是当前服务器验证通过的版本：复用旧 `fun-asr-nano` 的 ModelScope 缓存，并让两个服务都使用宿主机网络，避免端口映射重启异常。

- `funasr-ws`: CPU WebSocket 模型服务，使用 `network_mode: host`，仅监听宿主机 `127.0.0.1:10095`。
- `funasr-sse-adapter`: SSE Adapter 使用 `network_mode: host`，直接监听宿主机 `10098`，内部连接 `ws://127.0.0.1:10095`。
- 模型缓存: `/data/maas/sgy_arm/fun-asr-nano/models` 挂载到容器内 `/app/funasr-deploy/models`。

该 CPU 方案只需要 Docker 和 Docker Compose v2，不需要 NVIDIA 驱动，也不需要 NVIDIA Container Toolkit。为了避开你现有服务占用的宿主机 `10097`，新 SSE Adapter 对外端口使用 `10098`。

上传目录示例：

```text
/data/maas/sgy_arm/funasr-build
├── Dockerfile.cpu
├── docker-compose.full.cpu.server.yml
├── FunASR-src/
└── Fun-ASR-Nano-2512-Deploy/
```

首次构建镜像：

```bash
cd /data/maas/sgy_arm/funasr-build
docker compose -f docker-compose.full.cpu.server.yml build
```

当前服务器无法访问 ModelScope，且 `docker-compose.full.cpu.server.yml` 已复用旧模型缓存，因此不需要再执行 `download_model.py`。

启动两个服务：

```bash
docker compose -f docker-compose.full.cpu.server.yml up -d --no-build
```

查看状态和日志：

```bash
docker compose -f docker-compose.full.cpu.server.yml ps
docker logs -n 100 funasr-ws
docker logs -n 100 funasr-sse-adapter
```

健康检查：

```bash
curl http://127.0.0.1:10098/health
```

停止服务：

```bash
docker compose -f docker-compose.full.cpu.server.yml down
```

如果服务器是 x86 + NVIDIA GPU，才使用 `Fun-ASR-Nano-2512-Deploy/docker-compose.full.yml`。你的 ARM CPU 服务器不要走这个 GPU compose。

使用 `docker-compose.full.cpu.server.yml` 时，下面所有 SSE 示例里的 `10097` 都替换为 `10098`。

### 健康检查

```bash
curl http://127.0.0.1:10097/health
```

### 文件级 SSE 测试

该接口适合先用 `curl` 验证 SSE 输出格式。

```bash
curl -N -X POST "http://127.0.0.1:10097/asr/file-sse" \
  -F "file=@example.wav" \
  -F "mode=2pass"
```

正常会看到类似：

```text
event: online
data: {"mode":"2pass-online","text":"..."}

event: final
data: {"mode":"2pass-offline","text":"..."}

event: done
data: {}
```

### Base64 文件级 SSE 测试

该接口适合调用方只能传 JSON 的场景。`audio_base64` 支持普通 base64 字符串，也支持 `data:audio/wav;base64,...` 形式。

```bash
AUDIO_B64=$(base64 -w 0 example.wav)

curl -N -X POST "http://127.0.0.1:10097/asr/base64-sse" \
  -H "Content-Type: application/json" \
  -d "$(jq -n --arg audio "$AUDIO_B64" '{audio_base64:$audio, filename:"example.wav", mode:"2pass", audio_fs:16000}')"
```

请求体字段：

```json
{
  "audio_base64": "...",
  "filename": "example.wav",
  "mode": "2pass",
  "audio_fs": 16000,
  "chunk_size": "5,10,5",
  "chunk_interval": 10,
  "encoder_chunk_look_back": 4,
  "decoder_chunk_look_back": 0,
  "hotwords": ""
}
```

返回格式与 `/asr/file-sse` 一致，仍然是 SSE `online` / `final` / `error` / `done` 事件。

### 实时会话接口

创建会话：

```bash
curl -X POST "http://127.0.0.1:10097/asr/session" \
  -F "mode=2pass" \
  -F "audio_fs=16000"
```

返回：

```json
{"session_id":"...","backend":"ws://..."}
```

打开 SSE 订阅：

```bash
curl -N "http://127.0.0.1:10097/asr/sse/<session_id>"
```

上传 PCM chunk：

```bash
curl -X POST "http://127.0.0.1:10097/asr/chunk/<session_id>" \
  --data-binary @chunk.pcm
```

实时 base64 chunk 上传：

```bash
CHUNK_B64=$(base64 -w 0 chunk.pcm)

curl -X POST "http://127.0.0.1:10097/asr/chunk-b64/<session_id>" \
  -H "Content-Type: application/json" \
  -d "$(jq -n --arg audio "$CHUNK_B64" '{audio_base64:$audio}')"
```

结束会话：

```bash
curl -X POST "http://127.0.0.1:10097/asr/end/<session_id>"
```

### 音频格式要求

上传给 SSE Adapter 的音频建议统一为：

```text
16000 Hz
16bit
mono
PCM/WAV
```

转换命令：

```bash
ffmpeg -y -i input.wav -ar 16000 -ac 1 -sample_fmt s16 output-16k.wav
```

## 本地 Qwen-ASR 实验服务

项目根目录提供 `docker-compose.qwen-asr-local.yml` 和 `start_qwen_asr_local_server.sh`，用于尝试在 ARM CPU 上启动本地 `Qwen/Qwen3-ASR-0.6B` 服务。

启动命令：

```bash
./start_qwen_asr_local_server.sh
```

测试命令：

```bash
./download_qwen_asr_model.sh
curl http://127.0.0.1:10100/health
./test_qwen_asr_local_file_sse.sh /path/to/audio.wav
```

说明：

- 本地 Qwen-ASR 服务暴露 `POST /v1/audio/transcriptions`。
- `funasr-sse-adapter` 会通过 `QWEN_ASR_API_STYLE=transcriptions` 调用本地服务。
- 对外仍使用已有 `/qwen-asr/*` 接口。
- 这是 ARM CPU PoC，需要先通过 `./download_qwen_asr_model.sh` 把模型下载到服务器本地；性能和兼容性必须以服务器实测为准。
