import argparse
import asyncio
import base64
import binascii
import json
import os
import uuid
import wave
from dataclasses import dataclass, field
from typing import Any

import uvicorn
import websockets
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

DEFAULT_BACKEND = os.environ.get("FUNASR_BACKEND_WS", "ws://127.0.0.1:10095")
DEFAULT_BACKEND_CONNECT_RETRIES = int(os.environ.get("FUNASR_BACKEND_CONNECT_RETRIES", "30"))
DEFAULT_BACKEND_CONNECT_DELAY = float(os.environ.get("FUNASR_BACKEND_CONNECT_DELAY", "1"))
DEFAULT_CHUNK_SIZE = [5, 10, 5]
DEFAULT_CHUNK_INTERVAL = 10

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}

app = FastAPI()
sessions: dict[str, "AsrSession"] = {}
backend_ws = DEFAULT_BACKEND


@dataclass
class AsrSession:
    websocket: Any
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    recv_task: asyncio.Task | None = None
    ending: bool = False


class Base64AsrRequest(BaseModel):
    audio_base64: str
    filename: str = "audio.pcm"
    mode: str = "2pass"
    audio_fs: int = 16000
    chunk_size: str | list[int] = "5,10,5"
    chunk_interval: int = DEFAULT_CHUNK_INTERVAL
    encoder_chunk_look_back: int = 4
    decoder_chunk_look_back: int = 0
    hotwords: str = ""


class Base64ChunkRequest(BaseModel):
    audio_base64: str


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
            return wav_file.readframes(wav_file.getnframes()), sample_rate
    return data, audio_fs


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


@app.get("/health")
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
            async with await connect_backend() as ws:
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
                while True:
                    event, data = await queue.get()
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
        except Exception as exc:
            yield sse_event("error", {"message": str(exc)})
            yield sse_event("done", {})

    return StreamingResponse(events(), media_type="text/event-stream", headers=SSE_HEADERS)


@app.post("/asr/file-sse")
async def asr_file_sse(
    file: UploadFile = File(...),
    mode: str = Form("2pass"),
    audio_fs: int = Form(16000),
    chunk_size: str = Form("5,10,5"),
    chunk_interval: int = Form(DEFAULT_CHUNK_INTERVAL),
    encoder_chunk_look_back: int = Form(4),
    decoder_chunk_look_back: int = Form(0),
    hotwords: str = Form(""),
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


@app.post("/asr/base64-sse")
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


@app.post("/asr/session")
async def create_session(
    mode: str = Form("2pass"),
    audio_fs: int = Form(16000),
    chunk_size: str = Form("5,10,5"),
    chunk_interval: int = Form(DEFAULT_CHUNK_INTERVAL),
    encoder_chunk_look_back: int = Form(4),
    decoder_chunk_look_back: int = Form(0),
    hotwords: str = Form(""),
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


@app.get("/asr/sse/{session_id}")
async def session_sse(session_id: str):
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


@app.post("/asr/chunk/{session_id}")
async def send_chunk(session_id: str, request: Request):
    session = sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    data = await request.body()
    if not data:
        raise HTTPException(status_code=400, detail="Empty audio chunk")
    await session.websocket.send(data)
    return {"ok": True, "bytes": len(data)}


@app.post("/asr/chunk-b64/{session_id}")
async def send_chunk_base64(session_id: str, request: Base64ChunkRequest):
    session = sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    data = decode_audio_base64(request.audio_base64)
    await session.websocket.send(data)
    return {"ok": True, "bytes": len(data)}


@app.post("/asr/end/{session_id}")
async def end_session(session_id: str):
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
