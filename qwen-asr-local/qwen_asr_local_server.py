import argparse
import asyncio
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

app = FastAPI(
    title="Local Qwen-ASR service",
    description=(
        "Minimal local Qwen-ASR transcription service. "
        "It exposes an OpenAI-compatible /v1/audio/transcriptions endpoint "
        "for the main ASR adapter."
    ),
    version="0.1.0",
)

MODEL_ID = os.environ.get("QWEN_ASR_MODEL", "Qwen/Qwen3-ASR-0.6B")
DEVICE = os.environ.get("QWEN_ASR_DEVICE", "cpu")
BACKEND = os.environ.get("QWEN_ASR_BACKEND", "transformers")
LANGUAGE = os.environ.get("QWEN_ASR_LANGUAGE", "auto")
DTYPE = os.environ.get("QWEN_ASR_DTYPE", "float32").lower()
MAX_NEW_TOKENS = int(os.environ.get("QWEN_ASR_MAX_NEW_TOKENS", "512"))

_model: Any | None = None
_model_error: str | None = None
_model_lock = asyncio.Lock()
_model_loaded_at: float | None = None


def _load_model_sync() -> Any:
    global _model_loaded_at
    try:
        from qwen_asr import Qwen3ASRModel
    except Exception as exc:
        raise RuntimeError(
            "Failed to import qwen_asr. Check that the qwen-asr package is installed "
            "and supports this CPU/ARM environment."
        ) from exc

    if BACKEND != "transformers":
        raise RuntimeError("qwen-asr-local only supports QWEN_ASR_BACKEND=transformers")

    dtype = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
    }.get(DTYPE, torch.float32)

    model = Qwen3ASRModel.from_pretrained(
        MODEL_ID,
        dtype=dtype,
        device_map=DEVICE or "cpu",
        max_new_tokens=MAX_NEW_TOKENS,
    )
    _model_loaded_at = time.time()
    return model


async def get_model() -> Any:
    global _model, _model_error
    if _model is not None:
        return _model
    async with _model_lock:
        if _model is not None:
            return _model
        try:
            _model = await asyncio.to_thread(_load_model_sync)
            _model_error = None
            return _model
        except Exception as exc:
            _model_error = repr(exc)
            raise


def _extract_text(result: Any) -> str:
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        for key in ("text", "transcription", "result"):
            value = result.get(key)
            if isinstance(value, str):
                return value
        if "segments" in result and isinstance(result["segments"], list):
            return "".join(str(item.get("text", "")) for item in result["segments"] if isinstance(item, dict))
        return str(result)
    if isinstance(result, list):
        return "".join(_extract_text(item) for item in result)
    text = getattr(result, "text", None)
    if isinstance(text, str):
        return text
    return str(result)


def _transcribe_sync(model: Any, audio_path: str, language: str, prompt: str) -> str:
    qwen_language = None if language in {"", "auto", "none", "null"} else language
    context = prompt or ""
    attempts = [
        lambda: model.transcribe(audio=audio_path, language=qwen_language, context=context, return_time_stamps=False),
        lambda: model.transcribe(audio=audio_path, language=qwen_language, return_time_stamps=False),
        lambda: model.transcribe(audio=audio_path),
    ]
    last_error: Exception | None = None
    for attempt in attempts:
        try:
            return _extract_text(attempt())
        except TypeError as exc:
            last_error = exc
            continue
    if last_error is not None:
        raise last_error
    return ""


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok" if _model_error is None else "error",
        "model": MODEL_ID,
        "backend": BACKEND,
        "device": DEVICE,
        "loaded": _model is not None,
        "loaded_at": _model_loaded_at,
        "error": _model_error,
    }


@app.post("/v1/audio/transcriptions")
async def transcriptions(
    file: UploadFile = File(...),
    model: str = Form(MODEL_ID),
    language: str = Form(LANGUAGE),
    prompt: str = Form(""),
) -> dict[str, Any]:
    _ = model
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty audio file")

    try:
        local_model = await get_model()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Qwen-ASR model load failed: {exc}") from exc

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as temp_file:
        temp_file.write(raw)
        temp_file.flush()
        try:
            text = await asyncio.to_thread(_transcribe_sync, local_model, temp_file.name, language, prompt)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Qwen-ASR transcription failed: {exc}") from exc

    return {"text": text, "model": MODEL_ID, "backend": BACKEND, "device": DEVICE}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=10100)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
