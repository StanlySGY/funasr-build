import argparse
import audioop
import base64
import io
import json
import re
import threading
import time
import urllib.parse
import urllib.request
import wave


def parse_segments(text_path):
    text = open(text_path, encoding="utf-8").read()
    return re.findall(
        r"(base64_\d+)\s*:\s*([A-Za-z0-9+/=\r\n]+?)(?=\n\s*base64_\d+\s*:|\Z)",
        text,
        re.S,
    )


def post(base_url, path, data=b"", content_type="application/json", timeout=120):
    request = urllib.request.Request(
        base_url + path,
        data=data,
        headers={"Content-Type": content_type},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def decode_segment(payload):
    raw = base64.b64decode(re.sub(r"\s+", "", payload))
    if raw.startswith(b"RIFF"):
        with wave.open(io.BytesIO(raw), "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            frames = wav_file.readframes(wav_file.getnframes())
            if channels != 1:
                frames = audioop.tomono(frames, sample_width, 0.5, 0.5)
            if sample_width != 2:
                frames = audioop.lin2lin(frames, sample_width, 2)
            if sample_rate != 16000:
                frames, _ = audioop.ratecv(frames, 2, 1, sample_rate, 16000, None)
            if sample_rate != 16000 or channels != 1 or sample_width != 2:
                print(
                    f"converted WAV {sample_rate}Hz {channels}ch {sample_width} bytes/sample -> 16000Hz 1ch 2 bytes/sample",
                    flush=True,
                )
            return frames
    return raw


def read_sse(base_url, session_id, stop_event):
    try:
        with urllib.request.urlopen(base_url + f"/asr/sse/{session_id}", timeout=600) as response:
            for line in response:
                if stop_event.is_set():
                    break
                print(line.decode("utf-8", "ignore").rstrip(), flush=True)
    except Exception as exc:
        print(f"SSE closed: {exc}", flush=True)


def create_session(base_url, mode):
    form = urllib.parse.urlencode({"mode": mode, "audio_fs": 16000}).encode()
    response = post(base_url, "/asr/session", form, "application/x-www-form-urlencoded")
    return json.loads(response)["session_id"]


def send_segment(base_url, session_id, name, payload):
    pcm = decode_segment(payload)
    encoded = base64.b64encode(pcm).decode()
    body = json.dumps({"audio_base64": encoded}).encode()
    print(f"send {name}: pcm_bytes={len(pcm)}", flush=True)
    post(base_url, f"/asr/chunk-b64/{session_id}", body, "application/json")


def main():
    parser = argparse.ArgumentParser(description="Send text-file Base64 audio segments to FunASR SSE realtime API.")
    parser.add_argument("text_file", nargs="?", default="语音测试数据.txt")
    parser.add_argument("--base-url", default="http://127.0.0.1:10098")
    parser.add_argument("--mode", default="2pass", choices=["online", "2pass", "offline"])
    parser.add_argument("--delay-sec", type=float, default=1.0)
    parser.add_argument("--wait-after-end-sec", type=float, default=60.0)
    args = parser.parse_args()

    segments = parse_segments(args.text_file)
    print(f"segments={len(segments)} mode={args.mode} base_url={args.base_url}", flush=True)
    if not segments:
        raise RuntimeError(f"No base64_N segments found in {args.text_file}")

    session_id = create_session(args.base_url, args.mode)
    print(f"session_id={session_id}", flush=True)

    stop_event = threading.Event()
    sse_thread = threading.Thread(target=read_sse, args=(args.base_url, session_id, stop_event), daemon=True)
    sse_thread.start()
    time.sleep(1)

    for name, payload in segments:
        send_segment(args.base_url, session_id, name, payload)
        time.sleep(args.delay_sec)

    print("send end", flush=True)
    post(args.base_url, f"/asr/end/{session_id}", b"", "application/x-www-form-urlencoded")
    time.sleep(args.wait_after_end_sec)
    stop_event.set()


if __name__ == "__main__":
    main()
