import argparse
import asyncio
import audioop
import io
import json
import os
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse


@dataclass(frozen=True)
class SherpaOnnxConfig:
    model_type: str
    model_dir: Path
    tokens: Path
    paraformer: Path | None
    encoder: Path | None
    decoder: Path | None
    joiner: Path | None
    sample_rate: int
    feature_dim: int
    num_threads: int
    decoding_method: str
    provider: str
    debug: bool


app = FastAPI(
    title="Local sherpa-onnx ASR service",
    description="Small HTTP/SSE wrapper around sherpa-onnx offline recognizers.",
    version="0.1.0",
)

_recognizer: Any | None = None
_recognizer_error: str | None = None
_recognizer_loaded_at: float | None = None
_recognizer_lock = asyncio.Lock()


def env_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def resolve_path(model_dir: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return model_dir / path


def first_existing(model_dir: Path, explicit: str | None, candidates: list[str]) -> Path | None:
    path = resolve_path(model_dir, explicit)
    if path is not None:
        return path
    for name in candidates:
        candidate = model_dir / name
        if candidate.exists():
            return candidate
    return model_dir / candidates[0] if candidates else None


def build_config(env: Mapping[str, str] | None = None) -> SherpaOnnxConfig:
    values = os.environ if env is None else env
    model_dir = Path(values.get("SHERPA_ONNX_MODEL_DIR", "/models"))
    model_type = values.get("SHERPA_ONNX_MODEL_TYPE", "paraformer").strip().lower()
    return SherpaOnnxConfig(
        model_type=model_type,
        model_dir=model_dir,
        tokens=first_existing(model_dir, values.get("SHERPA_ONNX_TOKENS"), ["tokens.txt"]),
        paraformer=first_existing(
            model_dir,
            values.get("SHERPA_ONNX_PARA_MODEL"),
            ["model.int8.onnx", "model.onnx", "paraformer.int8.onnx", "paraformer.onnx"],
        ),
        encoder=first_existing(
            model_dir,
            values.get("SHERPA_ONNX_ENCODER"),
            ["encoder.int8.onnx", "encoder.onnx"],
        ),
        decoder=first_existing(
            model_dir,
            values.get("SHERPA_ONNX_DECODER"),
            ["decoder.int8.onnx", "decoder.onnx"],
        ),
        joiner=first_existing(
            model_dir,
            values.get("SHERPA_ONNX_JOINER"),
            ["joiner.int8.onnx", "joiner.onnx"],
        ),
        sample_rate=int(values.get("SHERPA_ONNX_SAMPLE_RATE", "16000")),
        feature_dim=int(values.get("SHERPA_ONNX_FEATURE_DIM", "80")),
        num_threads=int(values.get("SHERPA_ONNX_NUM_THREADS", "4")),
        decoding_method=values.get("SHERPA_ONNX_DECODING_METHOD", "greedy_search"),
        provider=values.get("SHERPA_ONNX_PROVIDER", "cpu"),
        debug=env_bool(values.get("SHERPA_ONNX_DEBUG"), False),
    )


def require_file(path: Path | None, label: str) -> str:
    if path is None or not path.exists():
        raise RuntimeError(f"Missing sherpa-onnx {label}: {path}")
    return str(path)


def create_recognizer(config: SherpaOnnxConfig) -> Any:
    try:
        import sherpa_onnx
    except Exception as exc:
        raise RuntimeError("Failed to import sherpa_onnx. Check the container package install.") from exc

    common = {
        "tokens": require_file(config.tokens, "tokens"),
        "num_threads": config.num_threads,
        "sample_rate": config.sample_rate,
        "feature_dim": config.feature_dim,
        "decoding_method": config.decoding_method,
        "provider": config.provider,
        "debug": config.debug,
    }
    if config.model_type == "paraformer":
        return sherpa_onnx.OfflineRecognizer.from_paraformer(
            paraformer=require_file(config.paraformer, "paraformer model"),
            **common,
        )
    if config.model_type == "transducer":
        return sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=require_file(config.encoder, "encoder"),
            decoder=require_file(config.decoder, "decoder"),
            joiner=require_file(config.joiner, "joiner"),
            **common,
        )
    raise RuntimeError("Unsupported SHERPA_ONNX_MODEL_TYPE. Use paraformer or transducer.")


async def get_recognizer() -> Any:
    global _recognizer, _recognizer_error, _recognizer_loaded_at
    if _recognizer is not None:
        return _recognizer
    async with _recognizer_lock:
        if _recognizer is not None:
            return _recognizer
        config = build_config()
        try:
            _recognizer = await asyncio.to_thread(create_recognizer, config)
            _recognizer_error = None
            _recognizer_loaded_at = time.time()
            return _recognizer
        except Exception as exc:
            _recognizer_error = repr(exc)
            raise


def read_wav_as_float32(raw: bytes, target_sample_rate: int = 16000) -> tuple[np.ndarray, int]:
    try:
        with wave.open(io.BytesIO(raw), "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            pcm = wav_file.readframes(wav_file.getnframes())
    except wave.Error as exc:
        raise ValueError("Only WAV input is supported by sherpa-onnx-local") from exc

    if channels != 1:
        pcm = audioop.tomono(pcm, sample_width, 0.5, 0.5)
        channels = 1
    if sample_width != 2:
        pcm = audioop.lin2lin(pcm, sample_width, 2)
        sample_width = 2
    if sample_rate != target_sample_rate:
        pcm, _ = audioop.ratecv(pcm, sample_width, channels, sample_rate, target_sample_rate, None)
        sample_rate = target_sample_rate

    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    return samples, sample_rate


def result_text(result: Any) -> str:
    if result is None:
        return ""
    text = getattr(result, "text", None)
    if isinstance(text, str):
        return text
    if isinstance(result, dict):
        value = result.get("text")
        return value if isinstance(value, str) else str(result)
    return str(result)


def transcribe_samples(recognizer: Any, samples: Any, sample_rate: int) -> str:
    stream = recognizer.create_stream()
    stream.accept_waveform(sample_rate, samples)
    recognizer.decode_stream(stream)
    return result_text(stream.result).strip()


async def transcribe_raw_audio(raw: bytes) -> str:
    if not raw:
        raise HTTPException(status_code=400, detail="Empty audio file")
    config = build_config()
    try:
        samples, sample_rate = read_wav_as_float32(raw, config.sample_rate)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        recognizer = await get_recognizer()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"sherpa-onnx model load failed: {exc}") from exc
    try:
        return await asyncio.to_thread(transcribe_samples, recognizer, samples, sample_rate)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"sherpa-onnx transcription failed: {exc}") from exc


def sse_event(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.get("/health")
async def health() -> dict[str, Any]:
    config = build_config()
    return {
        "status": "ok" if _recognizer_error is None else "error",
        "backend": "sherpa-onnx",
        "model_type": config.model_type,
        "model_dir": str(config.model_dir),
        "provider": config.provider,
        "num_threads": config.num_threads,
        "loaded": _recognizer is not None,
        "loaded_at": _recognizer_loaded_at,
        "error": _recognizer_error,
    }


@app.post("/v1/audio/transcriptions")
async def transcriptions(
    file: UploadFile = File(...),
    model: str = Form("sherpa-onnx"),
    language: str = Form("zh"),
    prompt: str = Form(""),
) -> dict[str, Any]:
    _ = model, language, prompt
    raw = await file.read()
    text = await transcribe_raw_audio(raw)
    config = build_config()
    return {"text": text, "model": "sherpa-onnx", "backend": config.model_type, "provider": config.provider}


@app.post("/asr/file-sse")
async def file_sse(
    file: UploadFile = File(...),
    mode: str = Form("offline"),
    hotwords: str = Form(""),
) -> StreamingResponse:
    _ = mode, hotwords
    raw = await file.read()
    filename = file.filename or "audio.wav"

    async def events():
        try:
            text = await transcribe_raw_audio(raw)
            yield sse_event(
                "final",
                {
                    "mode": "sherpa-onnx",
                    "text": text,
                    "wav_name": filename,
                    "is_final": True,
                    "provider": "sherpa-onnx",
                },
            )
        except HTTPException as exc:
            yield sse_event("error", {"message": str(exc.detail)})
        except Exception as exc:
            yield sse_event("error", {"message": str(exc)})
        yield sse_event("done", {})

    return StreamingResponse(events(), media_type="text/event-stream")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.environ.get("SHERPA_ONNX_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("SHERPA_ONNX_PORT", "10110")))
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
