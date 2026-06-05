import asyncio
import base64
import unittest

from fastapi import HTTPException

from asr_sse_adapter import (
    AsrSession,
    Base64ChunkRequest,
    app,
    decode_audio_base64,
    send_chunk_base64,
    session_sse,
    sessions,
)


class FakeWebSocket:
    def __init__(self):
        self.sent = []
        self.closed = False

    async def send(self, data):
        self.sent.append(data)

    async def close(self):
        self.closed = True


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


if __name__ == "__main__":
    unittest.main()
