"""Stream a WAV file through Suki Dictation STT and print the transcripts.

A manual smoke test for ``livekit-plugins-suki`` against a real Suki environment.

Requires ``SUKI_PARTNER_ID`` and ``SUKI_PARTNER_TOKEN`` (plus ``SUKI_PROVIDER_ID`` for
Bearer / Single Auth Token partners). Set ``SUKI_BASE_URL=https://sdp.suki-stage.com``
to use staging.

The WAV must be 16-bit PCM; any sample rate or channel count is accepted (downmixed
to mono here, resampled to 16 kHz by the stream). Audio is pushed in real time.

What to check on a live run:

- At least one FINAL should print *before* the "end_input()" line. The agent keeps one
  stream open for the whole call, so if finals only arrive after AUDIO_END, the plugin
  never yields mid-call user turns (that would need a per-turn AUDIO_END/reconnect).
- No "FINAL:   EOF" line should appear: the terminal EOF frame must not carry is_final.

Run with:

    uv run python examples/other/suki_stt_smoke.py path/to/dictation.wav --hold-open 10
"""

import argparse
import asyncio
import logging
import time
import wave

import aiohttp
import numpy as np
from dotenv import load_dotenv

from livekit import rtc
from livekit.agents import stt
from livekit.plugins import suki

_FRAME_MS = 20


def _read_wav(path: str) -> tuple[bytes, int, int]:
    """Read a 16-bit WAV as mono PCM (the stream's resampler expects mono input)."""
    with wave.open(path, "rb") as wav:
        if wav.getsampwidth() != 2:
            raise SystemExit("the WAV file must be 16-bit PCM")
        pcm, sample_rate, channels = (
            wav.readframes(wav.getnframes()),
            wav.getframerate(),
            wav.getnchannels(),
        )
    if channels > 1:
        samples = np.frombuffer(pcm, dtype=np.int16).reshape(-1, channels)
        pcm = samples.mean(axis=1).astype(np.int16).tobytes()
    return pcm, sample_rate, 1


async def _push_audio(
    stream: stt.SpeechStream, pcm: bytes, sample_rate: int, hold_open: float, started: float
) -> None:
    samples_per_frame = sample_rate * _FRAME_MS // 1000
    bytes_per_frame = samples_per_frame * 2
    # after the file, keep the stream open on silence the way a live call would, so
    # finals have to arrive on Suki's own segmentation rather than after AUDIO_END
    silence = bytes(bytes_per_frame) * int(hold_open * 1000 / _FRAME_MS)
    audio = pcm + silence
    for i, offset in enumerate(range(0, len(audio), bytes_per_frame)):
        if offset == len(pcm):
            print(f"[{time.monotonic() - started:6.2f}s] -- end of WAV, holding open on silence")
        chunk = audio[offset : offset + bytes_per_frame]
        stream.push_frame(
            rtc.AudioFrame(
                data=chunk,
                sample_rate=sample_rate,
                num_channels=1,
                samples_per_channel=len(chunk) // 2,
            )
        )
        # pace to real time
        await asyncio.sleep(max(0.0, started + (i + 1) * _FRAME_MS / 1000 - time.monotonic()))
    print(f"[{time.monotonic() - started:6.2f}s] -- end_input() (AUDIO_END)")
    stream.end_input()


async def main(path: str, language: str, hold_open: float) -> None:
    pcm, sample_rate, _ = _read_wav(path)
    print(f"streaming {len(pcm) / (sample_rate * 2):.1f}s of audio from {path}")

    async with aiohttp.ClientSession() as http_session:
        suki_stt = suki.STT(language=language, http_session=http_session)
        stream = suki_stt.stream()
        started = time.monotonic()
        push_task = asyncio.create_task(_push_audio(stream, pcm, sample_rate, hold_open, started))
        try:
            async for event in stream:
                stamp = f"[{time.monotonic() - started:6.2f}s]"
                if event.type == stt.SpeechEventType.INTERIM_TRANSCRIPT:
                    print(f"{stamp} interim: {event.alternatives[0].text}")
                elif event.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                    print(f"{stamp} FINAL:   {event.alternatives[0].text}")
                elif event.type == stt.SpeechEventType.RECOGNITION_USAGE:
                    assert event.recognition_usage is not None
                    print(f"{stamp} usage:   {event.recognition_usage.audio_duration:.2f}s")
                else:
                    print(f"{stamp} {event.type.value}")
        finally:
            await push_task
            await stream.aclose()
            await suki_stt.aclose()


if __name__ == "__main__":
    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("wav", help="16-bit PCM WAV file to stream")
    parser.add_argument("--language", default="English", help="Suki audio_language")
    parser.add_argument(
        "--hold-open",
        type=float,
        default=10.0,
        help="seconds of silence to stream after the WAV before ending input",
    )
    args = parser.parse_args()
    asyncio.run(main(args.wav, args.language, args.hold_open))
