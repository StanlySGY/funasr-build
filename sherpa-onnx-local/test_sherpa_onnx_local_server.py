import io
import os
import sys
import tempfile
import types
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(__file__))

from sherpa_onnx_local_server import (
    build_config,
    create_recognizer,
    read_wav_as_float32,
    transcribe_samples,
)


def build_wav_bytes(sample_rate: int, channels: int, sample_width: int, frames: bytes) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(frames)
    return buffer.getvalue()


class SherpaOnnxLocalServerTest(unittest.TestCase):
    def test_builds_paraformer_config_from_model_dir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            (model_dir / "model.int8.onnx").write_bytes(b"model")
            (model_dir / "tokens.txt").write_text("a\n", encoding="utf-8")

            config = build_config(
                {
                    "SHERPA_ONNX_MODEL_DIR": str(model_dir),
                    "SHERPA_ONNX_MODEL_TYPE": "paraformer",
                    "SHERPA_ONNX_NUM_THREADS": "4",
                }
            )

        self.assertEqual("paraformer", config.model_type)
        self.assertEqual("model.int8.onnx", config.paraformer.name)
        self.assertEqual("tokens.txt", config.tokens.name)
        self.assertEqual(4, config.num_threads)

    def test_create_paraformer_recognizer_uses_sherpa_factory(self):
        calls = []

        class FakeOfflineRecognizer:
            @staticmethod
            def from_paraformer(**kwargs):
                calls.append(kwargs)
                return object()

        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            (model_dir / "model.int8.onnx").write_bytes(b"model")
            (model_dir / "tokens.txt").write_text("a\n", encoding="utf-8")
            config = build_config({"SHERPA_ONNX_MODEL_DIR": str(model_dir)})
            fake_sherpa = types.SimpleNamespace(OfflineRecognizer=FakeOfflineRecognizer)
            with patch.dict(sys.modules, {"sherpa_onnx": fake_sherpa}):
                create_recognizer(config)

        self.assertEqual(str(config.paraformer), calls[0]["paraformer"])
        self.assertEqual(str(config.tokens), calls[0]["tokens"])
        self.assertEqual("cpu", calls[0]["provider"])

    def test_reads_wav_as_16k_float32_mono(self):
        wav_bytes = build_wav_bytes(24000, 1, 2, b"\x00\x00" * 2400)

        samples, sample_rate = read_wav_as_float32(wav_bytes, target_sample_rate=16000)

        self.assertEqual(16000, sample_rate)
        self.assertEqual("float32", str(samples.dtype))
        self.assertGreater(len(samples), 0)

    def test_transcribes_samples_with_offline_recognizer(self):
        class FakeResult:
            text = "你好"

        class FakeStream:
            def accept_waveform(self, sample_rate, samples):
                self.sample_rate = sample_rate
                self.samples = samples
                self.result = FakeResult()

        class FakeRecognizer:
            def create_stream(self):
                self.stream = FakeStream()
                return self.stream

            def decode_stream(self, stream):
                stream.decoded = True

        text = transcribe_samples(FakeRecognizer(), [0.0, 0.1], 16000)

        self.assertEqual("你好", text)


if __name__ == "__main__":
    unittest.main()
