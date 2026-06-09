import asyncio
import base64
import io
import os
import tempfile
import unittest
import wave
from unittest.mock import patch

from fastapi import HTTPException
from websockets.exceptions import InvalidMessage

from asr_sse_adapter import (
    AsrSession,
    Base64ChunkRequest,
    MAX_AUDIO_BYTES,
    append_diagnostic_log,
    app,
    cleanup_idle_resources,
    connect_backend,
    create_session,
    create_uploaded_file_session,
    create_qwen_session,
    decode_audio_base64,
    end_session,
    normalize_diagnostic_modes,
    qwen_sessions,
    qwen_uploaded_audios,
    qwen_session_sse,
    qwen_asr_file_sse,
    read_diagnostic_log_tail,
    send_chunk_base64,
    send_qwen_chunk_base64,
    session_sse,
    sessions,
    upload_wav_file,
    upload_qwen_wav_file,
    end_qwen_session,
    uploaded_audios,
    UploadedAudio,
    QwenSession,
)


class FakeWebSocket:
    def __init__(self):
        self.sent = []
        self.closed = False

    async def send(self, data):
        self.sent.append(data)

    async def close(self):
        self.closed = True


class FakeUploadFile:
    def __init__(self, filename, data):
        self.filename = filename
        self._data = data

    async def read(self):
        return self._data


def build_wav_bytes(sample_rate: int, channels: int, sample_width: int, frames: bytes) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(frames)
    return buffer.getvalue()


class DecodeAudioBase64Test(unittest.TestCase):
    def test_decodes_plain_base64_audio(self):
        expected = b"\x01\x02pcm"
        encoded = base64.b64encode(expected).decode("ascii")

        actual = decode_audio_base64(encoded)

        self.assertEqual(expected, actual)

    def test_decodes_data_url_base64_audio(self):
        expected = b"\x03\x04wav"
        encoded = base64.b64encode(expected).decode("ascii")

        actual = decode_audio_base64(f"data:audio/wav;base64,{encoded}")

        self.assertEqual(expected, actual)

    def test_decodes_base64_audio_with_whitespace(self):
        expected = b"\x05\x06pcm"
        encoded = base64.b64encode(expected).decode("ascii")

        actual = decode_audio_base64(f"{encoded[:4]}\n {encoded[4:]}")

        self.assertEqual(expected, actual)

    def test_rejects_invalid_base64_audio(self):
        with self.assertRaises(HTTPException) as context:
            decode_audio_base64("not valid base64")

        self.assertEqual(400, context.exception.status_code)

    def test_registers_base64_sse_route(self):
        paths = {route.path for route in app.routes}

        self.assertIn("/asr/base64-sse", paths)

    def test_registers_base64_chunk_route(self):
        paths = {route.path for route in app.routes}

        self.assertIn("/asr/chunk-b64/{session_id}", paths)

    def test_registers_diagnostic_routes(self):
        paths = {route.path for route in app.routes}

        self.assertIn("/diagnostics/realtime-ab", paths)
        self.assertIn("/diagnostics/log", paths)

    def test_registers_uploaded_wav_session_routes(self):
        paths = {route.path for route in app.routes}

        self.assertIn("/asr/upload-wav", paths)
        self.assertIn("/asr/uploaded-file-session/{audio_id}", paths)

    def test_registers_qwen_asr_routes(self):
        paths = {route.path for route in app.routes}

        self.assertIn("/qwen-asr/file-sse", paths)
        self.assertIn("/qwen-asr/base64-sse", paths)
        self.assertIn("/qwen-asr/upload-wav", paths)
        self.assertIn("/qwen-asr/uploaded-file-session/{audio_id}", paths)
        self.assertIn("/qwen-asr/session", paths)
        self.assertIn("/qwen-asr/sse/{session_id}", paths)
        self.assertIn("/qwen-asr/chunk-b64/{session_id}", paths)
        self.assertIn("/qwen-asr/end/{session_id}", paths)

    def test_normalizes_diagnostic_modes(self):
        self.assertEqual(["online", "2pass"], normalize_diagnostic_modes("online, 2pass"))

    def test_rejects_unknown_diagnostic_mode(self):
        with self.assertRaises(HTTPException) as context:
            normalize_diagnostic_modes("online,bad")

        self.assertEqual(400, context.exception.status_code)

    def test_appends_and_reads_diagnostic_log_tail(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = os.path.join(temp_dir, "web_diagnostic.log")
            with patch("asr_sse_adapter.DIAGNOSTIC_LOG_PATH", log_path):
                append_diagnostic_log({"event": "first"})
                append_diagnostic_log({"event": "second"})

                lines = read_diagnostic_log_tail(1)

        self.assertEqual(1, len(lines))
        self.assertIn('"event": "second"', lines[0])

    def test_sends_decoded_base64_chunk_to_session_websocket(self):
        session_id = "chunk-b64-test"
        websocket = FakeWebSocket()
        expected = b"\x10\x11pcm"
        encoded = base64.b64encode(expected).decode("ascii")
        sessions[session_id] = AsrSession(websocket=websocket)

        try:
            response = asyncio.run(
                send_chunk_base64(session_id, Base64ChunkRequest(audio_base64=encoded))
            )
        finally:
            sessions.pop(session_id, None)

        self.assertEqual([expected], websocket.sent)
        self.assertEqual({"ok": True, "bytes": len(expected)}, response)

    def test_normalizes_wav_base64_chunk_before_sending_to_backend(self):
        session_id = "chunk-b64-wav-test"
        websocket = FakeWebSocket()
        frames_48k = b"\x00\x00" * 4800
        wav_payload = build_wav_bytes(48000, 1, 2, frames_48k)
        encoded = base64.b64encode(wav_payload).decode("ascii")
        sessions[session_id] = AsrSession(websocket=websocket)

        try:
            response = asyncio.run(
                send_chunk_base64(session_id, Base64ChunkRequest(audio_base64=encoded))
            )
        finally:
            sessions.pop(session_id, None)

        self.assertEqual(1, len(websocket.sent))
        self.assertFalse(websocket.sent[0].startswith(b"RIFF"))
        self.assertEqual(3200, len(websocket.sent[0]))
        self.assertEqual({"ok": True, "bytes": 3200}, response)

    def test_upload_wav_file_returns_audio_id_and_stores_normalized_audio(self):
        wav_payload = build_wav_bytes(48000, 1, 2, b"\x00\x00" * 4800)

        response = asyncio.run(upload_wav_file(FakeUploadFile("input.wav", wav_payload)))
        audio_id = response["audio_id"]

        try:
            self.assertIn(audio_id, uploaded_audios)
            self.assertEqual("input.wav", response["filename"])
            self.assertEqual(16000, response["sample_rate"])
            self.assertEqual(3200, response["bytes"])
            self.assertEqual(3200, len(uploaded_audios[audio_id].audio_bytes))
        finally:
            uploaded_audios.pop(audio_id, None)

    def test_create_uploaded_file_session_starts_streaming_uploaded_audio(self):
        async def run_case():
            wav_payload = build_wav_bytes(16000, 1, 2, b"abcdef")
            upload_response = await upload_wav_file(FakeUploadFile("input.wav", wav_payload))
            audio_id = upload_response["audio_id"]
            websocket = FakeWebSocket()

            async def fake_connect_backend():
                return websocket

            async def fake_receive_to_queue(ws, queue):
                return None

            async def no_sleep(delay):
                return None

            with patch("asr_sse_adapter.connect_backend", fake_connect_backend), \
                 patch("asr_sse_adapter.receive_to_queue", fake_receive_to_queue), \
                 patch("asr_sse_adapter.asyncio.sleep", no_sleep):
                response = await create_uploaded_file_session(
                    audio_id,
                    mode="online",
                    chunk_size="5,10,5",
                    chunk_interval=10,
                    encoder_chunk_look_back=4,
                    decoder_chunk_look_back=0,
                    hotwords="",
                )
                await asyncio.sleep(0)
                await asyncio.sleep(0)

            session_id = response["session_id"]
            sessions.pop(session_id, None)
            uploaded_audios.pop(audio_id, None)
            return response, websocket.sent

        response, sent = asyncio.run(run_case())

        self.assertIn("session_id", response)
        self.assertEqual("input.wav", response["filename"])
        self.assertIn('"is_speaking": true', sent[0])
        self.assertEqual(b"abcdef", sent[1])
        self.assertIn('"is_speaking": false', sent[2])

    def test_qwen_file_sse_emits_final_and_done_events(self):
        async def collect_events():
            async def fake_transcribe(*args, **kwargs):
                return "你好"

            with patch("asr_sse_adapter.transcribe_qwen_audio", fake_transcribe):
                response = await qwen_asr_file_sse(
                    FakeUploadFile("input.wav", build_wav_bytes(16000, 1, 2, b"abcdef")),
                    mode="online",
                    audio_fs=16000,
                    hotwords="",
                )
                chunks = []
                async for chunk in response.body_iterator:
                    if isinstance(chunk, bytes):
                        chunk = chunk.decode("utf-8")
                    chunks.append(chunk)
                return "".join(chunks)

        body = asyncio.run(collect_events())

        self.assertIn("event: final", body)
        self.assertIn('"text": "你好"', body)
        self.assertIn('"provider": "qwen-asr"', body)
        self.assertIn("event: done", body)

    def test_qwen_chunk_session_runs_transcription_on_end(self):
        async def run_case():
            async def fake_transcribe(*args, **kwargs):
                return "分片结果"

            session_response = await create_qwen_session(mode="online", audio_fs=16000, hotwords="")
            session_id = session_response["session_id"]
            qwen_sessions[session_id].audio_bytes.extend(b"abcdef")
            with patch("asr_sse_adapter.transcribe_qwen_audio", fake_transcribe):
                await end_qwen_session(session_id)
                response = await qwen_session_sse(session_id)
                chunks = []
                async for chunk in response.body_iterator:
                    if isinstance(chunk, bytes):
                        chunk = chunk.decode("utf-8")
                    chunks.append(chunk)
                return "".join(chunks), session_id in qwen_sessions

        body, exists = asyncio.run(run_case())

        self.assertIn("event: final", body)
        self.assertIn('"text": "分片结果"', body)
        self.assertIn("event: done", body)
        self.assertFalse(exists)

    def test_qwen_upload_wav_returns_audio_id(self):
        response = asyncio.run(
            upload_qwen_wav_file(FakeUploadFile("qwen.wav", build_wav_bytes(16000, 1, 2, b"abcdef")))
        )
        audio_id = response["audio_id"]

        try:
            self.assertIn(audio_id, qwen_uploaded_audios)
            self.assertEqual("qwen.wav", response["filename"])
            self.assertEqual(16000, response["sample_rate"])
        finally:
            qwen_uploaded_audios.pop(audio_id, None)

    def test_cleanup_idle_resources_removes_abandoned_sessions_and_uploads(self):
        async def run_case():
            websocket = FakeWebSocket()
            session_id = "old-asr-session"
            qwen_session_id = "old-qwen-session"
            audio_id = "old-upload"
            qwen_audio_id = "old-qwen-upload"
            sessions[session_id] = AsrSession(websocket=websocket)
            qwen_sessions[qwen_session_id] = QwenSession()
            uploaded_audios[audio_id] = UploadedAudio("old.wav", b"abcdef", 16000)
            qwen_uploaded_audios[qwen_audio_id] = UploadedAudio("old-qwen.wav", b"abcdef", 16000)
            sessions[session_id].last_activity_at = 1
            qwen_sessions[qwen_session_id].last_activity_at = 1
            uploaded_audios[audio_id].last_activity_at = 1
            qwen_uploaded_audios[qwen_audio_id].last_activity_at = 1

            try:
                with patch("asr_sse_adapter.time.monotonic", return_value=10_000):
                    await cleanup_idle_resources(max_idle_sec=1)
                return (
                    session_id in sessions,
                    qwen_session_id in qwen_sessions,
                    audio_id in uploaded_audios,
                    qwen_audio_id in qwen_uploaded_audios,
                    websocket.closed,
                )
            finally:
                sessions.pop(session_id, None)
                qwen_sessions.pop(qwen_session_id, None)
                uploaded_audios.pop(audio_id, None)
                qwen_uploaded_audios.pop(qwen_audio_id, None)

        exists = asyncio.run(run_case())

        self.assertEqual((False, False, False, False, True), exists)

    def test_upload_wav_file_rejects_audio_larger_than_limit(self):
        payload = b"RIFF" + (b"\x00" * MAX_AUDIO_BYTES)

        with self.assertRaises(HTTPException) as context:
            asyncio.run(upload_wav_file(FakeUploadFile("too-large.wav", payload)))

        self.assertEqual(413, context.exception.status_code)

    def test_qwen_chunk_session_rejects_audio_larger_than_limit(self):
        async def run_case():
            session_response = await create_qwen_session(mode="online", audio_fs=16000, hotwords="")
            session_id = session_response["session_id"]
            qwen_sessions[session_id].audio_bytes.extend(b"\x00" * MAX_AUDIO_BYTES)
            encoded = base64.b64encode(b"\x00").decode("ascii")
            try:
                with self.assertRaises(HTTPException) as context:
                    await send_qwen_chunk_base64(session_id, Base64ChunkRequest(audio_base64=encoded))
                return context.exception.status_code
            finally:
                qwen_sessions.pop(session_id, None)

        status_code = asyncio.run(run_case())

        self.assertEqual(413, status_code)

    def test_splits_realtime_base64_chunk_to_backend_stride(self):
        session_id = "chunk-split-test"
        websocket = FakeWebSocket()
        payload = b"abcdefg"
        encoded = base64.b64encode(payload).decode("ascii")
        session = AsrSession(websocket=websocket, chunk_stride_bytes=3)
        sessions[session_id] = session

        try:
            response = asyncio.run(
                send_chunk_base64(session_id, Base64ChunkRequest(audio_base64=encoded))
            )
        finally:
            sessions.pop(session_id, None)

        self.assertEqual([b"abc", b"def"], websocket.sent)
        self.assertEqual(bytearray(b"g"), session.pending_audio)
        self.assertEqual({"ok": True, "bytes": len(payload)}, response)

    def test_paces_split_realtime_chunks_when_configured(self):
        session_id = "chunk-paced-test"
        websocket = FakeWebSocket()
        payload = b"abcdef"
        encoded = base64.b64encode(payload).decode("ascii")
        sleep_calls = []

        async def fake_sleep(delay):
            sleep_calls.append(delay)

        session = AsrSession(websocket=websocket, chunk_stride_bytes=3)
        sessions[session_id] = session

        try:
            with patch("asr_sse_adapter.CHUNK_FRAME_DELAY_SEC", 0.06), \
                 patch("asr_sse_adapter.asyncio.sleep", fake_sleep):
                response = asyncio.run(
                    send_chunk_base64(session_id, Base64ChunkRequest(audio_base64=encoded))
                )
        finally:
            sessions.pop(session_id, None)

        self.assertEqual([b"abc", b"def"], websocket.sent)
        self.assertEqual([0.06], sleep_calls)
        self.assertEqual({"ok": True, "bytes": len(payload)}, response)

    def test_end_session_flushes_pending_realtime_audio(self):
        session_id = "chunk-flush-test"
        websocket = FakeWebSocket()
        session = AsrSession(websocket=websocket, chunk_stride_bytes=3)
        session.pending_audio.extend(b"tail")
        sessions[session_id] = session

        try:
            response = asyncio.run(end_session(session_id))
        finally:
            sessions.pop(session_id, None)

        self.assertEqual(b"tail", websocket.sent[0])
        self.assertIn('"is_speaking": false', websocket.sent[1])
        self.assertEqual({"ok": True}, response)

    def test_session_sse_suppresses_expected_backend_close_after_end(self):
        async def collect_events():
            session_id = "close-frame-test"
            websocket = FakeWebSocket()
            session = AsrSession(websocket=websocket)
            session.ending = True
            sessions[session_id] = session
            await session.queue.put(("online", {"text": "你好"}))
            await session.queue.put(("error", {"message": "no close frame received or sent"}))
            await session.queue.put(("done", {}))

            response = await session_sse(session_id)
            chunks = []
            async for chunk in response.body_iterator:
                if isinstance(chunk, bytes):
                    chunk = chunk.decode("utf-8")
                chunks.append(chunk)
            return "".join(chunks), websocket.closed, session_id in sessions

        body, closed, session_exists = asyncio.run(collect_events())

        self.assertIn('event: online', body)
        self.assertIn('event: done', body)
        self.assertNotIn('no close frame', body)
        self.assertTrue(closed)
        self.assertFalse(session_exists)

    def test_create_session_returns_503_when_backend_is_not_ready(self):
        async def fail_connect_backend():
            raise RuntimeError("backend loading")

        with patch("asr_sse_adapter.connect_backend", fail_connect_backend):
            with self.assertRaises(HTTPException) as context:
                asyncio.run(
                    create_session(
                        mode="online",
                        audio_fs=16000,
                        chunk_size="5,10,5",
                        chunk_interval=10,
                        encoder_chunk_look_back=4,
                        decoder_chunk_look_back=0,
                        hotwords="",
                    )
                )

        self.assertEqual(503, context.exception.status_code)
        self.assertEqual("backend loading", context.exception.detail)

    def test_connect_backend_retries_invalid_handshake_response(self):
        calls = []
        websocket = FakeWebSocket()

        async def flaky_connect(*args, **kwargs):
            calls.append((args, kwargs))
            if len(calls) == 1:
                raise InvalidMessage("did not receive a valid HTTP response")
            return websocket

        async def no_sleep(delay):
            return None

        with patch("asr_sse_adapter.DEFAULT_BACKEND_CONNECT_RETRIES", 2), \
             patch("asr_sse_adapter.websockets.connect", flaky_connect), \
             patch("asr_sse_adapter.asyncio.sleep", no_sleep):
            result = asyncio.run(connect_backend())

        self.assertIs(websocket, result)
        self.assertEqual(2, len(calls))


if __name__ == "__main__":
    unittest.main()
