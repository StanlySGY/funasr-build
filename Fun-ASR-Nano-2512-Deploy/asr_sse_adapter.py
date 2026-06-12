import argparse
import asyncio
import audioop
import base64
import binascii
import json
import os
import time
import urllib.error
import urllib.request
import uuid
import wave
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import uvicorn
import websockets
from websockets.exceptions import InvalidHandshake, InvalidMessage
from fastapi import FastAPI, File, Form, HTTPException, Path, Request, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

DEFAULT_BACKEND = os.environ.get("FUNASR_BACKEND_WS", "ws://127.0.0.1:10095")
DEFAULT_BACKEND_CONNECT_RETRIES = int(os.environ.get("FUNASR_BACKEND_CONNECT_RETRIES", "30"))
DEFAULT_BACKEND_CONNECT_DELAY = float(os.environ.get("FUNASR_BACKEND_CONNECT_DELAY", "1"))
TARGET_SAMPLE_RATE = int(os.environ.get("FUNASR_TARGET_SAMPLE_RATE", "16000"))
DEFAULT_CHUNK_SIZE = [5, 10, 5]
DEFAULT_CHUNK_INTERVAL = 10
BACKEND_CONNECT_EXCEPTIONS = (OSError, EOFError, InvalidHandshake, InvalidMessage)
DIAGNOSTIC_LOG_PATH = os.environ.get(
    "FUNASR_DIAGNOSTIC_LOG",
    "/app/funasr-deploy/asr_logs/web_diagnostic.log",
)
DIAGNOSTIC_TIMEOUT_SEC = float(os.environ.get("FUNASR_DIAGNOSTIC_TIMEOUT_SEC", "120"))
DIAGNOSTIC_END_WAIT_SEC = float(os.environ.get("FUNASR_DIAGNOSTIC_END_WAIT_SEC", "35"))
ONLINE_RESULT_WAIT_SEC = float(os.environ.get("FUNASR_ONLINE_RESULT_WAIT_SEC", "35"))
CHUNK_FRAME_DELAY_SEC = float(os.environ.get("FUNASR_CHUNK_FRAME_DELAY_SEC", "0"))
QWEN_ASR_BASE_URL = os.environ.get("QWEN_ASR_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1").rstrip("/")
QWEN_ASR_API_KEY = os.environ.get("QWEN_ASR_API_KEY") or os.environ.get("DASHSCOPE_API_KEY", "")
QWEN_ASR_MODEL = os.environ.get("QWEN_ASR_MODEL", "qwen3-asr-flash")
QWEN_ASR_API_STYLE = os.environ.get("QWEN_ASR_API_STYLE", "chat").lower()
QWEN_ASR_TIMEOUT_SEC = float(os.environ.get("QWEN_ASR_TIMEOUT_SEC", "120"))
MAX_AUDIO_BYTES = int(os.environ.get("ASR_MAX_AUDIO_BYTES", str(50 * 1024 * 1024)))
SESSION_IDLE_TTL_SEC = float(os.environ.get("ASR_SESSION_IDLE_TTL_SEC", str(30 * 60)))
CLEANUP_INTERVAL_SEC = float(os.environ.get("ASR_CLEANUP_INTERVAL_SEC", "60"))

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}

API_DESCRIPTION = """
FunASR WebSocket 的 SSE 封装服务，面向业务系统提供语音识别接口。

## 推荐用法

### 0. 引擎选择
- `/asr/*`：使用本地 FunASR WebSocket 后端，适合 ARM CPU 稳定版实时识别。
- `/qwen-asr/*`：使用 Qwen-ASR 适配层，接口形态与 `/asr/*` 尽量保持一致；当前按整段音频调用 Qwen-ASR，并通过 SSE 返回 `final` 与 `done`。

### 1. 上传完整音频文件
使用 `POST /asr/file-sse`，适合已有 `.wav` 或 `.pcm` 文件并希望一次请求直接返回 SSE 识别流的场景。
完整文件默认使用快速发送模式；如需模拟实时播放，可传 `realtime=true`。
Qwen-ASR 对应接口为 `POST /qwen-asr/file-sse`。

### 2. 上传完整 Base64 音频
使用 `POST /asr/base64-sse`，适合前端或业务系统已经拿到完整音频 Base64 的场景。
Qwen-ASR 对应接口为 `POST /qwen-asr/base64-sse`。

### 3. 先上传 WAV，再创建流式识别会话
适合文件上传可能较慢、不希望前端一直等待识别请求的场景：
1. `POST /asr/upload-wav` 上传 WAV 文件，立即返回 `audio_id`
2. `POST /asr/uploaded-file-session/{audio_id}` 创建识别会话，立即返回 `session_id`
3. `GET /asr/sse/{session_id}` 订阅识别结果；服务端会在后台把已上传文件按实时分片推给 FunASR
Qwen-ASR 对应接口为 `/qwen-asr/upload-wav`、`/qwen-asr/uploaded-file-session/{audio_id}`、`/qwen-asr/sse/{session_id}`。

### 4. 实时推送 Base64 音频流
依次调用：
1. `POST /asr/session` 创建会话，拿到 `session_id`
2. `GET /asr/sse/{session_id}` 订阅识别结果
3. `POST /asr/chunk-b64/{session_id}` 持续推送 Base64 音频分片
4. `POST /asr/end/{session_id}` 通知服务端音频结束

`/asr/chunk-b64/{session_id}` 支持两类分片：
- 16kHz 单声道 PCM16 的 Base64
- 每段都是完整 WAV 文件的 Base64；服务端会自动去掉 WAV 头并转换为 16kHz 单声道 PCM16

## 返回事件

SSE 返回格式为 `event: 事件名` 和 `data: JSON`：
- `online`：实时中间识别结果
- `final`：最终识别结果
- `message`：其他后端消息
- `error`：错误信息
- `done`：本次识别结束

## 音频要求

- 推荐 16kHz、单声道、16-bit PCM/WAV
- WAV 上传支持自动重采样到 16kHz
- 实时 Base64 分片若传 WAV，必须保证每个分片本身都是完整 WAV
- 若传 PCM，需要通过 `audio_fs` 告诉服务端原始采样率
- 单次完整音频和单个会话累计音频默认最大 50MB，超过会返回 HTTP 413；可用 `ASR_MAX_AUDIO_BYTES` 调整
- 未订阅、断连或长期无活动的会话默认 30 分钟后自动清理；可用 `ASR_SESSION_IDLE_TTL_SEC` 调整

## Qwen-ASR 配置

- `QWEN_ASR_API_KEY` 或 `DASHSCOPE_API_KEY`：Qwen-ASR API Key
- `QWEN_ASR_BASE_URL`：默认 `https://dashscope.aliyuncs.com/compatible-mode/v1`
- `QWEN_ASR_MODEL`：默认 `qwen3-asr-flash`
- `QWEN_ASR_API_STYLE`：默认 `chat`；若接本地 OpenAI 兼容转写服务，可设为 `transcriptions`
"""

app = FastAPI(title="FunASR SSE 语音识别服务", description=API_DESCRIPTION, version="1.0.0")
sessions: dict[str, "AsrSession"] = {}
uploaded_audios: dict[str, "UploadedAudio"] = {}
qwen_sessions: dict[str, "QwenSession"] = {}
qwen_uploaded_audios: dict[str, "UploadedAudio"] = {}
backend_ws = DEFAULT_BACKEND


@dataclass
class AsrSession:
    websocket: Any
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    recv_task: asyncio.Task | None = None
    ending: bool = False
    chunk_stride_bytes: int | None = None
    pending_audio: bytearray = field(default_factory=bytearray)
    send_task: asyncio.Task | None = None
    audio_bytes_received: int = 0
    created_at: float = field(default_factory=time.monotonic)
    last_activity_at: float = field(default_factory=time.monotonic)


@dataclass
class UploadedAudio:
    filename: str
    audio_bytes: bytes
    sample_rate: int
    created_at: float = field(default_factory=time.monotonic)
    last_activity_at: float = field(default_factory=time.monotonic)


@dataclass
class QwenSession:
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    filename: str = "qwen-stream.wav"
    audio_bytes: bytearray = field(default_factory=bytearray)
    sample_rate: int = TARGET_SAMPLE_RATE
    hotwords: str = ""
    task: asyncio.Task | None = None
    ending: bool = False
    created_at: float = field(default_factory=time.monotonic)
    last_activity_at: float = field(default_factory=time.monotonic)


class Base64AsrRequest(BaseModel):
    audio_base64: str = Field(..., description="完整音频的 Base64 字符串，支持纯 Base64 或 data:audio/wav;base64,... 格式")
    filename: str = Field("audio.pcm", description="音频文件名，用于判断格式；WAV 请以 .wav 结尾，PCM 可用 .pcm")
    mode: str = Field("online", description="识别模式：online 为实时结果；2pass 为两遍识别；offline 为离线结果")
    audio_fs: int = Field(16000, description="PCM 原始采样率；WAV 会自动读取文件头采样率")
    chunk_size: str | list[int] = Field("5,10,5", description="FunASR 分块参数，默认 5,10,5；一般不用改")
    chunk_interval: int = Field(DEFAULT_CHUNK_INTERVAL, description="FunASR 分块间隔，默认 10；一般不用改")
    encoder_chunk_look_back: int = Field(4, description="编码器回看块数，默认 4；一般不用改")
    decoder_chunk_look_back: int = Field(0, description="解码器回看块数，默认 0；一般不用改")
    hotwords: str = Field("", description="热词，多个热词可按 FunASR 后端格式传入；没有可留空")
    realtime: bool = Field(False, description="是否按实时速度分片发送；完整文件默认关闭以加快识别")


class Base64ChunkRequest(BaseModel):
    audio_base64: str = Field(
        ...,
        description="单个音频分片的 Base64 字符串。支持 16k PCM16 Base64，或完整 WAV 分片 Base64/data:audio/...;base64,...",
    )


def utc_now_text() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def append_diagnostic_log(entry: dict) -> dict:
    payload = {"ts": utc_now_text(), **entry}
    log_dir = os.path.dirname(DIAGNOSTIC_LOG_PATH)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    with open(DIAGNOSTIC_LOG_PATH, "a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    return payload


def read_diagnostic_log_tail(limit: int = 200) -> list[str]:
    if not os.path.exists(DIAGNOSTIC_LOG_PATH):
        return []
    with open(DIAGNOSTIC_LOG_PATH, encoding="utf-8") as log_file:
        lines = log_file.readlines()
    return [line.rstrip("\n") for line in lines[-max(1, min(limit, 2000)):]]


def normalize_diagnostic_modes(value: str) -> list[str]:
    modes = []
    for item in value.split(","):
        mode = item.strip()
        if not mode:
            continue
        if mode not in {"online", "2pass", "offline"}:
            raise HTTPException(status_code=400, detail=f"Unsupported diagnostic mode: {mode}")
        modes.append(mode)
    return modes or ["online", "2pass"]


def parse_chunk_size(value: str | list[int]) -> list[int]:
    if isinstance(value, list):
        return value
    return [int(item.strip()) for item in value.split(",")]


def decode_audio_base64(audio_base64: str) -> bytes:
    payload = audio_base64.strip()
    if payload.startswith("data:") and "," in payload:
        payload = payload.split(",", 1)[1]
    payload = "".join(payload.split())
    try:
        data = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Invalid base64 audio payload") from exc
    if not data:
        raise HTTPException(status_code=400, detail="Empty audio payload")
    return data


def touch_session(session: AsrSession | QwenSession) -> None:
    session.last_activity_at = time.monotonic()


def ensure_audio_size_allowed(size: int, *, existing_size: int = 0) -> None:
    if existing_size + size > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail=f"Audio payload exceeds {MAX_AUDIO_BYTES} bytes limit")


def ensure_audio_payload_allowed(data: bytes) -> None:
    ensure_audio_size_allowed(len(data))


def normalize_sample_rate(pcm_data: bytes, sample_rate: int) -> tuple[bytes, int]:
    if sample_rate == TARGET_SAMPLE_RATE:
        return pcm_data, sample_rate
    converted, _ = audioop.ratecv(pcm_data, 2, 1, sample_rate, TARGET_SAMPLE_RATE, None)
    return converted, TARGET_SAMPLE_RATE


def normalize_wav_bytes(data: bytes) -> tuple[bytes, int]:
    import io

    try:
        with wave.open(io.BytesIO(data), "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            frames = wav_file.readframes(wav_file.getnframes())
    except wave.Error as exc:
        raise HTTPException(status_code=400, detail="Invalid WAV audio payload") from exc
    if channels != 1:
        frames = audioop.tomono(frames, sample_width, 0.5, 0.5)
    if sample_width != 2:
        frames = audioop.lin2lin(frames, sample_width, 2)
    return normalize_sample_rate(frames, sample_rate)


def pcm_to_wav_bytes(pcm_data: bytes, sample_rate: int = TARGET_SAMPLE_RATE) -> bytes:
    import io

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_data)
    return buffer.getvalue()


def read_audio_payload(filename: str, data: bytes, audio_fs: int) -> tuple[bytes, int]:
    if filename.lower().endswith(".wav"):
        return normalize_wav_bytes(data)
    return normalize_sample_rate(data, audio_fs)


def normalize_realtime_chunk_payload(data: bytes) -> bytes:
    if data.startswith(b"RIFF"):
        pcm_data, _ = normalize_wav_bytes(data)
        return pcm_data
    return data


def normalize_qwen_audio_payload(filename: str, data: bytes, audio_fs: int) -> tuple[bytes, int]:
    if data.startswith(b"RIFF") or filename.lower().endswith(".wav"):
        return normalize_wav_bytes(data)
    return normalize_sample_rate(data, audio_fs)


def qwen_headers(content_type: str) -> dict[str, str]:
    headers = {"Content-Type": content_type}
    if QWEN_ASR_API_KEY:
        headers["Authorization"] = f"Bearer {QWEN_ASR_API_KEY}"
    return headers


def qwen_text_from_response(payload: dict) -> str:
    if "text" in payload:
        return str(payload.get("text") or "")
    choices = payload.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") in {"text", "output_text"}:
                parts.append(str(item.get("text", "")))
        return "".join(parts)
    return str(content or "")


def post_json(url: str, payload: dict) -> dict:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=qwen_headers("application/json"), method="POST")
    try:
        with urllib.request.urlopen(request, timeout=QWEN_ASR_TIMEOUT_SEC) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "ignore")
        raise RuntimeError(f"Qwen ASR HTTP {exc.code}: {detail}") from exc


def post_multipart_transcription(url: str, wav_bytes: bytes, filename: str) -> dict:
    boundary = f"----qwen-asr-{uuid.uuid4().hex}"
    fields = [
        ("model", QWEN_ASR_MODEL),
    ]
    body = bytearray()
    for name, value in fields:
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(f'Content-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
    body.extend(f"--{boundary}\r\n".encode())
    body.extend(
        (
            f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
            "Content-Type: audio/wav\r\n\r\n"
        ).encode()
    )
    body.extend(wav_bytes)
    body.extend(f"\r\n--{boundary}--\r\n".encode())

    headers = qwen_headers(f"multipart/form-data; boundary={boundary}")
    request = urllib.request.Request(url, data=bytes(body), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=QWEN_ASR_TIMEOUT_SEC) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "ignore")
        raise RuntimeError(f"Qwen ASR HTTP {exc.code}: {detail}") from exc


def call_qwen_asr(wav_bytes: bytes, filename: str = "audio.wav", hotwords: str = "") -> str:
    if QWEN_ASR_API_STYLE == "transcriptions":
        payload = post_multipart_transcription(f"{QWEN_ASR_BASE_URL}/audio/transcriptions", wav_bytes, filename)
        return qwen_text_from_response(payload)
    if not QWEN_ASR_API_KEY:
        raise RuntimeError("QWEN_ASR_API_KEY or DASHSCOPE_API_KEY is required for Qwen ASR")
    audio_base64 = base64.b64encode(wav_bytes).decode("ascii")
    prompt = "请准确转写这段音频。"
    if hotwords:
        prompt += f" 识别时优先参考这些热词：{hotwords}"
    payload = {
        "model": QWEN_ASR_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "input_audio", "input_audio": {"data": f"data:audio/wav;base64,{audio_base64}"}},
                ],
            }
        ],
    }
    response = post_json(f"{QWEN_ASR_BASE_URL}/chat/completions", payload)
    return qwen_text_from_response(response)


async def transcribe_qwen_audio(wav_bytes: bytes, filename: str = "audio.wav", hotwords: str = "") -> str:
    return await asyncio.to_thread(call_qwen_asr, wav_bytes, filename, hotwords)


def build_init_message(
    *,
    mode: str,
    chunk_size: list[int],
    chunk_interval: int,
    audio_fs: int,
    wav_name: str,
    encoder_chunk_look_back: int,
    decoder_chunk_look_back: int,
    hotwords: str,
) -> str:
    return json.dumps(
        {
            "mode": mode,
            "chunk_size": chunk_size,
            "chunk_interval": chunk_interval,
            "encoder_chunk_look_back": encoder_chunk_look_back,
            "decoder_chunk_look_back": decoder_chunk_look_back,
            "audio_fs": audio_fs,
            "wav_name": wav_name,
            "is_speaking": True,
            "hotwords": hotwords,
            "itn": True,
        },
        ensure_ascii=False,
    )


def chunk_stride(sample_rate: int, chunk_size: list[int], chunk_interval: int) -> int:
    return int(60 * chunk_size[1] / chunk_interval / 1000 * sample_rate * 2)


def sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def event_name(message: dict) -> str:
    mode = message.get("mode", "")
    if mode in {"2pass-online", "online"}:
        return "online"
    if mode in {"2pass-offline", "offline"} or message.get("is_final"):
        return "final"
    return "message"


def compact_backend_message(data: dict) -> dict:
    compact = {}
    for key in ("mode", "is_final", "text", "text_tn", "wav_name"):
        if key in data:
            compact[key] = data[key]
    if "message" in data:
        compact["message"] = data["message"]
    return compact or data


async def run_diagnostic_mode(
    *,
    run_id: str,
    mode: str,
    audio_bytes: bytes,
    sample_rate: int,
    chunk_size: list[int],
    chunk_interval: int,
    chunk_delay_ms: int,
    hotwords: str,
) -> dict:
    started = time.time()
    wav_name = f"diagnostic-{run_id}-{mode}"
    result = {
        "mode": mode,
        "status": "running",
        "events": [],
        "elapsed_ms": None,
    }
    queue: asyncio.Queue = asyncio.Queue()
    recv_task = None
    ws = None
    append_diagnostic_log({"run_id": run_id, "event": "mode_start", "mode": mode, "bytes": len(audio_bytes)})
    try:
        ws = await connect_backend()
        await ws.send(
            build_init_message(
                mode=mode,
                chunk_size=chunk_size,
                chunk_interval=chunk_interval,
                audio_fs=sample_rate,
                wav_name=wav_name,
                encoder_chunk_look_back=4,
                decoder_chunk_look_back=0,
                hotwords=hotwords,
            )
        )
        recv_task = asyncio.create_task(receive_to_queue(ws, queue))
        stride = chunk_stride(sample_rate, chunk_size, chunk_interval)
        delay_sec = max(0, min(chunk_delay_ms, 1000)) / 1000
        for start in range(0, len(audio_bytes), stride):
            await ws.send(audio_bytes[start:start + stride])
            if delay_sec:
                await asyncio.sleep(delay_sec)
        await ws.send(json.dumps({"is_speaking": False}))

        while time.time() - started < DIAGNOSTIC_TIMEOUT_SEC:
            try:
                event, data = await asyncio.wait_for(queue.get(), timeout=DIAGNOSTIC_END_WAIT_SEC)
            except asyncio.TimeoutError:
                result["status"] = "timeout"
                break
            compact = compact_backend_message(data)
            result["events"].append({"event": event, "data": compact})
            append_diagnostic_log({"run_id": run_id, "event": "backend_event", "mode": mode, "name": event, "data": compact})
            if event == "error":
                result["status"] = "error"
                break
            if event == "online" and mode == "online":
                result["status"] = "ok"
                break
            if event == "done":
                result["status"] = "ok"
                break
            if event == "final" and mode in {"2pass", "offline"}:
                result["status"] = "ok"
                break
        else:
            result["status"] = "timeout"
    except Exception as exc:
        result["status"] = "error"
        result["events"].append({"event": "exception", "data": {"message": str(exc)}})
        append_diagnostic_log(
            {"run_id": run_id, "event": "mode_exception", "mode": mode, "exc_type": type(exc).__name__, "message": str(exc)}
        )
    finally:
        if recv_task:
            recv_task.cancel()
        if ws is not None:
            with suppress(Exception):
                await ws.close()
        result["elapsed_ms"] = round((time.time() - started) * 1000, 3)
        append_diagnostic_log(
            {"run_id": run_id, "event": "mode_end", "mode": mode, "status": result["status"], "elapsed_ms": result["elapsed_ms"]}
        )
    return result


def is_backend_close_without_frame(event: str, data: dict) -> bool:
    return event == "error" and "no close frame" in data.get("message", "")


async def connect_backend():
    last_error = None
    for attempt in range(1, DEFAULT_BACKEND_CONNECT_RETRIES + 1):
        try:
            return await websockets.connect(backend_ws, subprotocols=["binary"], ping_interval=None)
        except BACKEND_CONNECT_EXCEPTIONS as exc:
            last_error = exc
            if attempt >= DEFAULT_BACKEND_CONNECT_RETRIES:
                break
            await asyncio.sleep(DEFAULT_BACKEND_CONNECT_DELAY)
    raise RuntimeError(
        f"FunASR backend is not ready after {DEFAULT_BACKEND_CONNECT_RETRIES} attempts: {last_error}"
    )


async def receive_to_queue(ws, queue: asyncio.Queue) -> None:
    try:
        async for response in ws:
            try:
                message = json.loads(response)
            except json.JSONDecodeError:
                await queue.put(("error", {"message": "Invalid JSON from FunASR", "raw": response}))
                continue
            await queue.put((event_name(message), message))
    except Exception as exc:
        await queue.put(("error", {"message": str(exc)}))
    finally:
        await queue.put(("done", {}))


async def cleanup_session(session_id: str) -> None:
    session = sessions.pop(session_id, None)
    if session is None:
        return
    if session.send_task:
        session.send_task.cancel()
    if session.recv_task:
        session.recv_task.cancel()
    try:
        await session.websocket.close()
    except Exception:
        pass


async def cleanup_qwen_session(session_id: str) -> None:
    session = qwen_sessions.pop(session_id, None)
    if session is None:
        return
    if session.task:
        session.task.cancel()


async def cleanup_idle_resources(max_idle_sec: float = SESSION_IDLE_TTL_SEC) -> dict[str, int]:
    now = time.monotonic()
    removed = {"sessions": 0, "qwen_sessions": 0, "uploaded_audios": 0, "qwen_uploaded_audios": 0}
    for session_id, session in list(sessions.items()):
        if now - session.last_activity_at > max_idle_sec:
            await cleanup_session(session_id)
            removed["sessions"] += 1
    for session_id, session in list(qwen_sessions.items()):
        if now - session.last_activity_at > max_idle_sec:
            await cleanup_qwen_session(session_id)
            removed["qwen_sessions"] += 1
    for audio_id, uploaded in list(uploaded_audios.items()):
        if now - uploaded.last_activity_at > max_idle_sec:
            uploaded_audios.pop(audio_id, None)
            removed["uploaded_audios"] += 1
    for audio_id, uploaded in list(qwen_uploaded_audios.items()):
        if now - uploaded.last_activity_at > max_idle_sec:
            qwen_uploaded_audios.pop(audio_id, None)
            removed["qwen_uploaded_audios"] += 1
    return removed


async def cleanup_idle_resources_loop() -> None:
    while True:
        await asyncio.sleep(CLEANUP_INTERVAL_SEC)
        await cleanup_idle_resources()


@app.on_event("startup")
async def start_cleanup_task() -> None:
    asyncio.create_task(cleanup_idle_resources_loop())


async def send_session_audio(session: AsrSession, data: bytes) -> None:
    ensure_audio_size_allowed(len(data), existing_size=session.audio_bytes_received)
    session.audio_bytes_received += len(data)
    touch_session(session)
    stride = session.chunk_stride_bytes
    if not stride or stride <= 0:
        await session.websocket.send(data)
        return
    session.pending_audio.extend(data)
    while len(session.pending_audio) >= stride:
        frame = bytes(session.pending_audio[:stride])
        del session.pending_audio[:stride]
        await session.websocket.send(frame)
        if CHUNK_FRAME_DELAY_SEC > 0 and len(session.pending_audio) >= stride:
            await asyncio.sleep(CHUNK_FRAME_DELAY_SEC)


async def flush_session_audio(session: AsrSession) -> None:
    if session.pending_audio:
        await session.websocket.send(bytes(session.pending_audio))
        session.pending_audio.clear()


async def stream_audio_to_session(session: AsrSession, audio_bytes: bytes) -> None:
    try:
        await send_session_audio(session, audio_bytes)
        session.ending = True
        await flush_session_audio(session)
        await session.websocket.send(json.dumps({"is_speaking": False}))
    except Exception as exc:
        await session.queue.put(("error", {"message": str(exc)}))


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def frontend_index():
    index_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    try:
        with open(index_path, encoding="utf-8") as index_file:
            return index_file.read()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Frontend page not found") from exc


@app.get("/health", summary="健康检查", description="检查 SSE 适配器是否启动，并返回当前后端 WebSocket 地址和会话数量。")
async def health():
    return {"status": "ok", "backend": backend_ws, "sessions": len(sessions)}


@app.get(
    "/diagnostics/log",
    summary="读取 Web 诊断日志",
    description="读取固定诊断日志文件尾部内容，便于把诊断结果提交到 GitHub 后远程分析。",
)
async def diagnostic_log(limit: int = 200):
    return {"log_path": DIAGNOSTIC_LOG_PATH, "lines": read_diagnostic_log_tail(limit)}


@app.post(
    "/diagnostics/realtime-ab",
    summary="一键实时识别 A/B 诊断",
    description="上传同一段音频，自动依次跑 online/2pass 等模式，并把每步结果写入固定日志文件。",
)
async def realtime_ab_diagnostic(
    file: UploadFile = File(..., description="用于诊断的音频文件；推荐真人语音 WAV"),
    modes: str = Form("online,2pass", description="逗号分隔的诊断模式，默认 online,2pass"),
    chunk_delay_ms: int = Form(10, description="每个后端分片发送后的等待毫秒数，默认 10"),
    hotwords: str = Form("", description="热词；没有可留空"),
):
    run_id = uuid.uuid4().hex[:12]
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty diagnostic audio file")
    mode_list = normalize_diagnostic_modes(modes)
    try:
        audio_bytes, sample_rate = read_audio_payload(file.filename or "diagnostic.wav", raw, TARGET_SAMPLE_RATE)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid diagnostic audio file: {exc}") from exc

    append_diagnostic_log(
        {
            "run_id": run_id,
            "event": "run_start",
            "filename": file.filename,
            "modes": mode_list,
            "sample_rate": sample_rate,
            "bytes": len(audio_bytes),
            "backend": backend_ws,
        }
    )
    results = []
    for mode in mode_list:
        results.append(
            await run_diagnostic_mode(
                run_id=run_id,
                mode=mode,
                audio_bytes=audio_bytes,
                sample_rate=sample_rate,
                chunk_size=DEFAULT_CHUNK_SIZE,
                chunk_interval=DEFAULT_CHUNK_INTERVAL,
                chunk_delay_ms=chunk_delay_ms,
                hotwords=hotwords,
            )
        )
    append_diagnostic_log({"run_id": run_id, "event": "run_end", "results": results})
    return {"run_id": run_id, "log_path": DIAGNOSTIC_LOG_PATH, "results": results}


def build_audio_sse_response(
    *,
    raw: bytes,
    filename: str,
    mode: str,
    audio_fs: int,
    chunk_size: str | list[int],
    chunk_interval: int,
    encoder_chunk_look_back: int,
    decoder_chunk_look_back: int,
    hotwords: str,
    realtime: bool = False,
) -> StreamingResponse:
    ensure_audio_payload_allowed(raw)
    chunks = parse_chunk_size(chunk_size)
    audio_bytes, sample_rate = read_audio_payload(filename, raw, audio_fs)
    stride = chunk_stride(sample_rate, chunks, chunk_interval)
    wav_name = filename or "upload"
    backend_chunk_interval = chunk_interval
    if not realtime and mode == "online":
        backend_chunk_interval = 1

    async def events():
        queue: asyncio.Queue = asyncio.Queue()
        try:
            ws = await connect_backend()
            try:
                await ws.send(
                    build_init_message(
                        mode=mode,
                        chunk_size=chunks,
                        chunk_interval=backend_chunk_interval,
                        audio_fs=sample_rate,
                        wav_name=wav_name,
                        encoder_chunk_look_back=encoder_chunk_look_back,
                        decoder_chunk_look_back=decoder_chunk_look_back,
                        hotwords=hotwords,
                    )
                )
                recv_task = asyncio.create_task(receive_to_queue(ws, queue))

                async def send_audio():
                    try:
                        if not realtime:
                            await ws.send(audio_bytes)
                            await ws.send(json.dumps({"is_speaking": False}))
                            return
                        sleep_sec = 60 * chunks[1] / chunk_interval / 1000
                        for start in range(0, len(audio_bytes), stride):
                            await ws.send(audio_bytes[start : start + stride])
                            await asyncio.sleep(sleep_sec)
                        await ws.send(json.dumps({"is_speaking": False}))
                    except Exception as exc:
                        await queue.put(("error", {"message": str(exc)}))

                end_sent = asyncio.Event()

                async def send_audio_with_end_marker():
                    await send_audio()
                    end_sent.set()

                send_task = asyncio.create_task(send_audio_with_end_marker())
                should_emit_done = True
                received_result = False
                while True:
                    try:
                        timeout = ONLINE_RESULT_WAIT_SEC if end_sent.is_set() and mode == "online" else None
                        event, data = await asyncio.wait_for(queue.get(), timeout=timeout)
                    except asyncio.TimeoutError:
                        break
                    if event in {"online", "final", "message"}:
                        received_result = True
                    if (
                        end_sent.is_set()
                        and received_result
                        and event == "error"
                        and "no close frame" in data.get("message", "")
                    ):
                        break
                    yield sse_event(event, data)
                    if event == "done":
                        should_emit_done = False
                        break
                    if not realtime and end_sent.is_set() and event in {"online", "final"}:
                        break
                    if end_sent.is_set() and event in {"error", "final"}:
                        break
                send_task.cancel()
                recv_task.cancel()
                if should_emit_done:
                    yield sse_event("done", {})
            finally:
                with suppress(Exception):
                    await ws.close()
        except Exception as exc:
            yield sse_event("error", {"message": str(exc)})
            yield sse_event("done", {})

    return StreamingResponse(events(), media_type="text/event-stream", headers=SSE_HEADERS)


def qwen_final_payload(text: str, filename: str) -> dict:
    return {
        "mode": "qwen-asr",
        "text": text,
        "wav_name": filename,
        "is_final": True,
        "provider": "qwen-asr",
    }


def build_qwen_audio_sse_response(
    *,
    raw: bytes,
    filename: str,
    audio_fs: int,
    hotwords: str,
) -> StreamingResponse:
    ensure_audio_payload_allowed(raw)
    audio_bytes, sample_rate = normalize_qwen_audio_payload(filename, raw, audio_fs)
    wav_bytes = pcm_to_wav_bytes(audio_bytes, sample_rate)

    async def events():
        try:
            text = await transcribe_qwen_audio(wav_bytes, filename or "audio.wav", hotwords)
            yield sse_event("final", qwen_final_payload(text, filename or "audio.wav"))
        except Exception as exc:
            yield sse_event("error", {"message": str(exc), "provider": "qwen-asr"})
        yield sse_event("done", {})

    return StreamingResponse(events(), media_type="text/event-stream", headers=SSE_HEADERS)


async def run_qwen_session_task(session: QwenSession) -> None:
    try:
        wav_bytes = pcm_to_wav_bytes(bytes(session.audio_bytes), session.sample_rate)
        text = await transcribe_qwen_audio(wav_bytes, session.filename, session.hotwords)
        await session.queue.put(("final", qwen_final_payload(text, session.filename)))
    except Exception as exc:
        await session.queue.put(("error", {"message": str(exc), "provider": "qwen-asr"}))
    finally:
        await session.queue.put(("done", {}))


@app.post(
    "/asr/file-sse",
    summary="上传音频文件并返回 SSE 识别流",
    description=(
        "上传一个完整音频文件，接口会通过 SSE 返回识别结果。默认按完整文件快速发送，"
        "适合测试、批量文件识别、已有 WAV/PCM 文件的业务场景；如需模拟实时流可传 realtime=true。"
    ),
)
async def asr_file_sse(
    file: UploadFile = File(..., description="要识别的音频文件；推荐 16kHz 单声道 16-bit WAV/PCM，WAV 支持自动重采样"),
    mode: str = Form("online", description="识别模式：online 实时；2pass 两遍识别；offline 离线"),
    audio_fs: int = Form(16000, description="PCM 采样率；如果上传 WAV，会自动读取 WAV 文件头"),
    chunk_size: str = Form("5,10,5", description="FunASR 分块参数，默认 5,10,5；一般不用改"),
    chunk_interval: int = Form(DEFAULT_CHUNK_INTERVAL, description="FunASR 分块间隔，默认 10；一般不用改"),
    encoder_chunk_look_back: int = Form(4, description="编码器回看块数，默认 4；一般不用改"),
    decoder_chunk_look_back: int = Form(0, description="解码器回看块数，默认 0；一般不用改"),
    hotwords: str = Form("", description="热词；没有可留空"),
    realtime: bool = Form(False, description="是否按实时速度分片发送完整文件；默认关闭以加快文件识别"),
):
    raw = await file.read()
    ensure_audio_payload_allowed(raw)
    return build_audio_sse_response(
        raw=raw,
        filename=file.filename or "audio.pcm",
        mode=mode,
        audio_fs=audio_fs,
        chunk_size=chunk_size,
        chunk_interval=chunk_interval,
        encoder_chunk_look_back=encoder_chunk_look_back,
        decoder_chunk_look_back=decoder_chunk_look_back,
        hotwords=hotwords,
        realtime=realtime,
    )


@app.post(
    "/asr/base64-sse",
    summary="上传完整 Base64 音频并返回 SSE 识别流",
    description="请求体传完整音频 Base64，接口会返回 SSE 识别结果。完整文件默认快速发送，可设置 realtime=true 模拟实时流。",
)
async def asr_base64_sse(request: Base64AsrRequest):
    raw = decode_audio_base64(request.audio_base64)
    ensure_audio_payload_allowed(raw)
    return build_audio_sse_response(
        raw=raw,
        filename=request.filename,
        mode=request.mode,
        audio_fs=request.audio_fs,
        chunk_size=request.chunk_size,
        chunk_interval=request.chunk_interval,
        encoder_chunk_look_back=request.encoder_chunk_look_back,
        decoder_chunk_look_back=request.decoder_chunk_look_back,
        hotwords=request.hotwords,
        realtime=request.realtime,
    )


@app.post(
    "/asr/upload-wav",
    summary="上传 WAV 文件并返回音频 ID",
    description=(
        "只上传 WAV 文件并立即返回 audio_id，不在本请求里等待识别结果。"
        "前端拿到 audio_id 后再调用 /asr/uploaded-file-session/{audio_id} 创建识别会话并订阅 SSE。"
    ),
)
async def upload_wav_file(
    file: UploadFile = File(..., description="要上传的 WAV 文件；服务端会转换为 16kHz 单声道 PCM16 后暂存"),
):
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty audio file")
    ensure_audio_payload_allowed(raw)
    filename = file.filename or "upload.wav"
    if not filename.lower().endswith(".wav") and not raw.startswith(b"RIFF"):
        raise HTTPException(status_code=400, detail="Only WAV upload is supported")
    audio_bytes, sample_rate = normalize_wav_bytes(raw)
    audio_id = uuid.uuid4().hex
    uploaded_audios[audio_id] = UploadedAudio(
        filename=filename,
        audio_bytes=audio_bytes,
        sample_rate=sample_rate,
    )
    return {
        "audio_id": audio_id,
        "filename": filename,
        "sample_rate": sample_rate,
        "bytes": len(audio_bytes),
    }


@app.post(
    "/asr/session",
    summary="创建实时识别会话",
    description="创建一个实时语音识别会话，返回 session_id。之后用该 session_id 订阅 SSE，并持续推送音频分片。",
)
async def create_session(
    mode: str = Form("online", description="识别模式：online 实时；2pass 两遍识别；offline 离线"),
    audio_fs: int = Form(16000, description="后续推送 PCM 分片的采样率；推荐 16000"),
    chunk_size: str = Form("5,10,5", description="FunASR 分块参数，默认 5,10,5；一般不用改"),
    chunk_interval: int = Form(DEFAULT_CHUNK_INTERVAL, description="FunASR 分块间隔，默认 10；一般不用改"),
    encoder_chunk_look_back: int = Form(4, description="编码器回看块数，默认 4；一般不用改"),
    decoder_chunk_look_back: int = Form(0, description="解码器回看块数，默认 0；一般不用改"),
    hotwords: str = Form("", description="热词；没有可留空"),
):
    session_id = uuid.uuid4().hex
    chunks = parse_chunk_size(chunk_size)
    try:
        ws = await connect_backend()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    await ws.send(
        build_init_message(
            mode=mode,
            chunk_size=chunks,
            chunk_interval=chunk_interval,
            audio_fs=audio_fs,
            wav_name=session_id,
            encoder_chunk_look_back=encoder_chunk_look_back,
            decoder_chunk_look_back=decoder_chunk_look_back,
            hotwords=hotwords,
        )
    )
    session = AsrSession(websocket=ws, chunk_stride_bytes=chunk_stride(audio_fs, chunks, chunk_interval))
    session.recv_task = asyncio.create_task(receive_to_queue(ws, session.queue))
    sessions[session_id] = session
    return {"session_id": session_id, "backend": backend_ws}


@app.post(
    "/asr/uploaded-file-session/{audio_id}",
    summary="用已上传 WAV 创建流式识别会话",
    description=(
        "前端先调用 /asr/upload-wav 得到 audio_id，再调用本接口创建 session_id。"
        "创建成功后服务端会在后台把已上传文件按实时分片推给 FunASR，前端用 /asr/sse/{session_id} 接收流式识别结果。"
    ),
)
async def create_uploaded_file_session(
    audio_id: str = Path(..., description="通过 POST /asr/upload-wav 返回的音频 ID"),
    mode: str = Form("online", description="识别模式：online 实时；2pass 两遍识别；offline 离线"),
    chunk_size: str = Form("5,10,5", description="FunASR 分块参数，默认 5,10,5；一般不用改"),
    chunk_interval: int = Form(DEFAULT_CHUNK_INTERVAL, description="FunASR 分块间隔，默认 10；一般不用改"),
    encoder_chunk_look_back: int = Form(4, description="编码器回看块数，默认 4；一般不用改"),
    decoder_chunk_look_back: int = Form(0, description="解码器回看块数，默认 0；一般不用改"),
    hotwords: str = Form("", description="热词；没有可留空"),
):
    chunks = parse_chunk_size(chunk_size)
    uploaded = uploaded_audios.pop(audio_id, None)
    if uploaded is None:
        raise HTTPException(status_code=404, detail="Uploaded audio not found")
    session_id = uuid.uuid4().hex
    try:
        ws = await connect_backend()
    except RuntimeError as exc:
        uploaded_audios[audio_id] = uploaded
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    await ws.send(
        build_init_message(
            mode=mode,
            chunk_size=chunks,
            chunk_interval=chunk_interval,
            audio_fs=uploaded.sample_rate,
            wav_name=uploaded.filename,
            encoder_chunk_look_back=encoder_chunk_look_back,
            decoder_chunk_look_back=decoder_chunk_look_back,
            hotwords=hotwords,
        )
    )
    session = AsrSession(
        websocket=ws,
        chunk_stride_bytes=chunk_stride(uploaded.sample_rate, chunks, chunk_interval),
    )
    session.recv_task = asyncio.create_task(receive_to_queue(ws, session.queue))
    session.send_task = asyncio.create_task(stream_audio_to_session(session, uploaded.audio_bytes))
    sessions[session_id] = session
    return {
        "session_id": session_id,
        "audio_id": audio_id,
        "filename": uploaded.filename,
        "backend": backend_ws,
    }


@app.get(
    "/asr/sse/{session_id}",
    summary="订阅实时识别 SSE 结果",
    description="订阅指定会话的 SSE 识别结果。客户端应保持连接，服务端会持续返回 online/final/error/done 等事件。",
)
async def session_sse(session_id: str = Path(..., description="通过 POST /asr/session 创建得到的会话 ID")):
    session = sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    async def events():
        try:
            saw_final_after_end = False
            while True:
                event, data = await session.queue.get()
                if session.ending and is_backend_close_without_frame(event, data):
                    continue
                yield sse_event(event, data)
                if session.ending and event == "final":
                    saw_final_after_end = True
                if event == "done" or saw_final_after_end:
                    break
        finally:
            await cleanup_session(session_id)

    return StreamingResponse(events(), media_type="text/event-stream", headers=SSE_HEADERS)


@app.post(
    "/asr/chunk/{session_id}",
    summary="推送二进制音频分片",
    description="向指定会话推送一段二进制 PCM 音频分片。请求体直接放原始音频 bytes，不是 JSON。",
)
async def send_chunk(
    request: Request,
    session_id: str = Path(..., description="通过 POST /asr/session 创建得到的会话 ID"),
):
    session = sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    data = await request.body()
    if not data:
        raise HTTPException(status_code=400, detail="Empty audio chunk")
    ensure_audio_payload_allowed(data)
    await send_session_audio(session, data)
    return {"ok": True, "bytes": len(data)}


@app.post(
    "/asr/chunk-b64/{session_id}",
    summary="推送 Base64 音频分片",
    description=(
        "向指定会话推送一段 Base64 音频分片。支持 16k PCM16 Base64，"
        "也支持每段都是完整 WAV 的 Base64；WAV 会自动转换为 16k 单声道 PCM16 后再喂给 FunASR。"
    ),
)
async def send_chunk_base64(session_id: str, request: Base64ChunkRequest):
    session = sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    data = normalize_realtime_chunk_payload(decode_audio_base64(request.audio_base64))
    ensure_audio_payload_allowed(data)
    await send_session_audio(session, data)
    return {"ok": True, "bytes": len(data)}


@app.post(
    "/asr/end/{session_id}",
    summary="结束实时识别会话",
    description="通知服务端当前会话的音频已经发送完毕。调用后后端会输出剩余识别结果并结束 SSE 流。",
)
async def end_session(session_id: str = Path(..., description="通过 POST /asr/session 创建得到的会话 ID")):
    session = sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    session.ending = True
    await flush_session_audio(session)
    await session.websocket.send(json.dumps({"is_speaking": False}))
    return {"ok": True}


@app.post(
    "/qwen-asr/file-sse",
    summary="Qwen-ASR 上传音频文件并返回 SSE 识别流",
    description="上传完整音频文件，调用 Qwen-ASR 识别，并通过 SSE 返回 final/done 事件。",
)
async def qwen_asr_file_sse(
    file: UploadFile = File(..., description="要识别的音频文件；推荐 WAV，服务端会统一转为 16kHz 单声道 WAV"),
    mode: str = Form("online", description="兼容 FunASR 参数；Qwen-ASR 当前按整段识别返回 final"),
    audio_fs: int = Form(16000, description="PCM 采样率；WAV 会自动读取文件头"),
    hotwords: str = Form("", description="热词；没有可留空"),
):
    _ = mode
    raw = await file.read()
    ensure_audio_payload_allowed(raw)
    return build_qwen_audio_sse_response(
        raw=raw,
        filename=file.filename or "audio.wav",
        audio_fs=audio_fs,
        hotwords=hotwords,
    )


@app.post(
    "/qwen-asr/base64-sse",
    summary="Qwen-ASR 上传完整 Base64 音频并返回 SSE 识别流",
    description="请求体传完整音频 Base64，调用 Qwen-ASR 识别，并通过 SSE 返回 final/done 事件。",
)
async def qwen_asr_base64_sse(request: Base64AsrRequest):
    raw = decode_audio_base64(request.audio_base64)
    ensure_audio_payload_allowed(raw)
    return build_qwen_audio_sse_response(
        raw=raw,
        filename=request.filename,
        audio_fs=request.audio_fs,
        hotwords=request.hotwords,
    )


@app.post(
    "/qwen-asr/upload-wav",
    summary="Qwen-ASR 上传 WAV 文件并返回音频 ID",
    description="只上传 WAV 文件并立即返回 audio_id。前端再用 audio_id 创建 Qwen-ASR 流式会话。",
)
async def upload_qwen_wav_file(
    file: UploadFile = File(..., description="要上传的 WAV 文件；服务端会转换为 16kHz 单声道 PCM16 后暂存"),
):
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty audio file")
    ensure_audio_payload_allowed(raw)
    filename = file.filename or "qwen-upload.wav"
    if not filename.lower().endswith(".wav") and not raw.startswith(b"RIFF"):
        raise HTTPException(status_code=400, detail="Only WAV upload is supported")
    audio_bytes, sample_rate = normalize_wav_bytes(raw)
    audio_id = uuid.uuid4().hex
    qwen_uploaded_audios[audio_id] = UploadedAudio(filename=filename, audio_bytes=audio_bytes, sample_rate=sample_rate)
    return {
        "audio_id": audio_id,
        "filename": filename,
        "sample_rate": sample_rate,
        "bytes": len(audio_bytes),
        "provider": "qwen-asr",
    }


@app.post(
    "/qwen-asr/uploaded-file-session/{audio_id}",
    summary="Qwen-ASR 用已上传 WAV 创建识别会话",
    description="前端先上传 WAV 得到 audio_id，再调用本接口创建 session_id，然后订阅 /qwen-asr/sse/{session_id}。",
)
async def create_qwen_uploaded_file_session(
    audio_id: str = Path(..., description="通过 POST /qwen-asr/upload-wav 返回的音频 ID"),
    mode: str = Form("online", description="兼容 FunASR 参数；Qwen-ASR 当前按整段识别返回 final"),
    hotwords: str = Form("", description="热词；没有可留空"),
):
    _ = mode
    uploaded = qwen_uploaded_audios.pop(audio_id, None)
    if uploaded is None:
        raise HTTPException(status_code=404, detail="Uploaded audio not found")
    session_id = uuid.uuid4().hex
    session = QwenSession(
        filename=uploaded.filename,
        audio_bytes=bytearray(uploaded.audio_bytes),
        sample_rate=uploaded.sample_rate,
        hotwords=hotwords,
    )
    session.task = asyncio.create_task(run_qwen_session_task(session))
    qwen_sessions[session_id] = session
    return {
        "session_id": session_id,
        "audio_id": audio_id,
        "filename": uploaded.filename,
        "provider": "qwen-asr",
    }


@app.post(
    "/qwen-asr/session",
    summary="Qwen-ASR 创建分片上传会话",
    description="创建 Qwen-ASR 音频收集会话。持续推送分片后，调用 /qwen-asr/end/{session_id} 触发整段识别。",
)
async def create_qwen_session(
    mode: str = Form("online", description="兼容 FunASR 参数；Qwen-ASR 当前按整段识别返回 final"),
    audio_fs: int = Form(16000, description="后续推送 PCM 分片的采样率；推荐 16000"),
    hotwords: str = Form("", description="热词；没有可留空"),
):
    _ = mode
    session_id = uuid.uuid4().hex
    qwen_sessions[session_id] = QwenSession(sample_rate=audio_fs, hotwords=hotwords)
    return {"session_id": session_id, "provider": "qwen-asr"}


@app.get(
    "/qwen-asr/sse/{session_id}",
    summary="Qwen-ASR 订阅 SSE 识别结果",
    description="订阅 Qwen-ASR 会话结果。Qwen-ASR 当前输出 final/error/done 事件。",
)
async def qwen_session_sse(session_id: str = Path(..., description="通过 Qwen-ASR 会话接口创建得到的会话 ID")):
    session = qwen_sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Qwen session not found")
    touch_session(session)

    async def events():
        try:
            while True:
                event, data = await session.queue.get()
                yield sse_event(event, data)
                if event == "done":
                    break
        finally:
            old_session = qwen_sessions.pop(session_id, None)
            if old_session and old_session.task:
                old_session.task.cancel()

    return StreamingResponse(events(), media_type="text/event-stream", headers=SSE_HEADERS)


@app.post(
    "/qwen-asr/chunk-b64/{session_id}",
    summary="Qwen-ASR 推送 Base64 音频分片",
    description="向 Qwen-ASR 会话推送一段 Base64 音频。WAV 分片会转成 PCM 后累计；调用 end 后触发整段识别。",
)
async def send_qwen_chunk_base64(session_id: str, request: Base64ChunkRequest):
    session = qwen_sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Qwen session not found")
    data = normalize_realtime_chunk_payload(decode_audio_base64(request.audio_base64))
    ensure_audio_size_allowed(len(data), existing_size=len(session.audio_bytes))
    touch_session(session)
    session.audio_bytes.extend(data)
    return {"ok": True, "bytes": len(data), "provider": "qwen-asr"}


@app.post(
    "/qwen-asr/end/{session_id}",
    summary="Qwen-ASR 结束分片上传并触发识别",
    description="结束 Qwen-ASR 分片上传会话，并在后台调用 Qwen-ASR。前端继续通过 SSE 接收 final/done。",
)
async def end_qwen_session(session_id: str = Path(..., description="通过 POST /qwen-asr/session 创建得到的会话 ID")):
    session = qwen_sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Qwen session not found")
    if session.ending:
        return {"ok": True, "provider": "qwen-asr"}
    if not session.audio_bytes:
        raise HTTPException(status_code=400, detail="Empty Qwen audio session")
    session.ending = True
    session.task = asyncio.create_task(run_qwen_session_task(session))
    return {"ok": True, "provider": "qwen-asr"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=10097)
    parser.add_argument("--backend", default=DEFAULT_BACKEND)
    args = parser.parse_args()
    backend_ws = args.backend
    uvicorn.run(app, host=args.host, port=args.port)
