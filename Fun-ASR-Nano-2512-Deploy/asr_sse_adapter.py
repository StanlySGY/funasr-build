import argparse
import asyncio
import audioop
import base64
import binascii
import json
import os
import uuid
import wave
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any

import uvicorn
import websockets
from fastapi import FastAPI, File, Form, HTTPException, Path, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

DEFAULT_BACKEND = os.environ.get("FUNASR_BACKEND_WS", "ws://127.0.0.1:10095")
DEFAULT_BACKEND_CONNECT_RETRIES = int(os.environ.get("FUNASR_BACKEND_CONNECT_RETRIES", "30"))
DEFAULT_BACKEND_CONNECT_DELAY = float(os.environ.get("FUNASR_BACKEND_CONNECT_DELAY", "1"))
TARGET_SAMPLE_RATE = int(os.environ.get("FUNASR_TARGET_SAMPLE_RATE", "16000"))
DEFAULT_CHUNK_SIZE = [5, 10, 5]
DEFAULT_CHUNK_INTERVAL = 10

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}

API_DESCRIPTION = """
FunASR WebSocket 的 SSE 封装服务，面向业务系统提供语音识别接口。

## 推荐用法

### 1. 上传完整音频文件
使用 `POST /asr/file-sse`，适合已有 `.wav` 或 `.pcm` 文件的场景，接口会直接返回 SSE 流式识别结果。

### 2. 上传完整 Base64 音频
使用 `POST /asr/base64-sse`，适合前端或业务系统已经拿到完整音频 Base64 的场景。

### 3. 实时推送 Base64 音频流
依次调用：
1. `POST /asr/session` 创建会话，拿到 `session_id`
2. `GET /asr/sse/{session_id}` 订阅识别结果
3. `POST /asr/chunk-b64/{session_id}` 持续推送 Base64 音频分片
4. `POST /asr/end/{session_id}` 通知服务端音频结束

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
- 若传 PCM，需要通过 `audio_fs` 告诉服务端原始采样率
"""

app = FastAPI(title="FunASR SSE 语音识别服务", description=API_DESCRIPTION, version="1.0.0")
sessions: dict[str, "AsrSession"] = {}
backend_ws = DEFAULT_BACKEND


@dataclass
class AsrSession:
    websocket: Any
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    recv_task: asyncio.Task | None = None
    ending: bool = False


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


class Base64ChunkRequest(BaseModel):
    audio_base64: str = Field(..., description="单个音频分片的 Base64 字符串，支持纯 Base64 或 data:audio/...;base64,... 格式")


def parse_chunk_size(value: str | list[int]) -> list[int]:
    if isinstance(value, list):
        return value
    return [int(item.strip()) for item in value.split(",")]


def decode_audio_base64(audio_base64: str) -> bytes:
    payload = audio_base64.strip()
    if payload.startswith("data:") and "," in payload:
        payload = payload.split(",", 1)[1]
    try:
        data = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Invalid base64 audio payload") from exc
    if not data:
        raise HTTPException(status_code=400, detail="Empty audio payload")
    return data


def normalize_sample_rate(pcm_data: bytes, sample_rate: int) -> tuple[bytes, int]:
    if sample_rate == TARGET_SAMPLE_RATE:
        return pcm_data, sample_rate
    converted, _ = audioop.ratecv(pcm_data, 2, 1, sample_rate, TARGET_SAMPLE_RATE, None)
    return converted, TARGET_SAMPLE_RATE


def read_audio_payload(filename: str, data: bytes, audio_fs: int) -> tuple[bytes, int]:
    if filename.lower().endswith(".wav"):
        import io

        with wave.open(io.BytesIO(data), "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            if channels != 1 or sample_width != 2:
                raise HTTPException(
                    status_code=400,
                    detail="WAV must be 16-bit mono PCM. Convert with: ffmpeg -i input.wav -ar 16000 -ac 1 -sample_fmt s16 output.wav",
                )
            return normalize_sample_rate(wav_file.readframes(wav_file.getnframes()), sample_rate)
    return normalize_sample_rate(data, audio_fs)


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


async def connect_backend():
    last_error = None
    for attempt in range(1, DEFAULT_BACKEND_CONNECT_RETRIES + 1):
        try:
            return await websockets.connect(backend_ws, subprotocols=["binary"], ping_interval=None)
        except OSError as exc:
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
    if session.recv_task:
        session.recv_task.cancel()
    try:
        await session.websocket.close()
    except Exception:
        pass


@app.get("/health", summary="健康检查", description="检查 SSE 适配器是否启动，并返回当前后端 WebSocket 地址和会话数量。")
async def health():
    return {"status": "ok", "backend": backend_ws, "sessions": len(sessions)}


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
) -> StreamingResponse:
    chunks = parse_chunk_size(chunk_size)
    audio_bytes, sample_rate = read_audio_payload(filename, raw, audio_fs)
    stride = chunk_stride(sample_rate, chunks, chunk_interval)
    wav_name = filename or "upload"

    async def events():
        queue: asyncio.Queue = asyncio.Queue()
        try:
            ws = await connect_backend()
            try:
                await ws.send(
                    build_init_message(
                        mode=mode,
                        chunk_size=chunks,
                        chunk_interval=chunk_interval,
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
                        timeout = 1.0 if end_sent.is_set() and mode == "online" else None
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


@app.post(
    "/asr/file-sse",
    summary="上传音频文件并返回 SSE 识别流",
    description="上传一个完整音频文件，接口会边处理边通过 SSE 返回识别结果。适合测试、批量文件识别、已有 WAV/PCM 文件的业务场景。",
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
):
    return build_audio_sse_response(
        raw=await file.read(),
        filename=file.filename or "audio.pcm",
        mode=mode,
        audio_fs=audio_fs,
        chunk_size=chunk_size,
        chunk_interval=chunk_interval,
        encoder_chunk_look_back=encoder_chunk_look_back,
        decoder_chunk_look_back=decoder_chunk_look_back,
        hotwords=hotwords,
    )


@app.post(
    "/asr/base64-sse",
    summary="上传完整 Base64 音频并返回 SSE 识别流",
    description="请求体传完整音频 Base64，接口会返回 SSE 流式识别结果。适合前端或业务系统一次性提交完整音频内容。",
)
async def asr_base64_sse(request: Base64AsrRequest):
    return build_audio_sse_response(
        raw=decode_audio_base64(request.audio_base64),
        filename=request.filename,
        mode=request.mode,
        audio_fs=request.audio_fs,
        chunk_size=request.chunk_size,
        chunk_interval=request.chunk_interval,
        encoder_chunk_look_back=request.encoder_chunk_look_back,
        decoder_chunk_look_back=request.decoder_chunk_look_back,
        hotwords=request.hotwords,
    )


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
    ws = await connect_backend()
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
    session = AsrSession(websocket=ws)
    session.recv_task = asyncio.create_task(receive_to_queue(ws, session.queue))
    sessions[session_id] = session
    return {"session_id": session_id, "backend": backend_ws}


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
    await session.websocket.send(data)
    return {"ok": True, "bytes": len(data)}


@app.post(
    "/asr/chunk-b64/{session_id}",
    summary="推送 Base64 音频分片",
    description="向指定会话推送一段 Base64 音频分片。适合前端实时采集音频后按片段上传的场景。",
)
async def send_chunk_base64(session_id: str, request: Base64ChunkRequest):
    session = sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    data = decode_audio_base64(request.audio_base64)
    await session.websocket.send(data)
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
    await session.websocket.send(json.dumps({"is_speaking": False}))
    return {"ok": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=10097)
    parser.add_argument("--backend", default=DEFAULT_BACKEND)
    args = parser.parse_args()
    backend_ws = args.backend
    uvicorn.run(app, host=args.host, port=args.port)
