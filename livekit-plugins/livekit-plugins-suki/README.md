# Suki plugin for LiveKit Agents

Support for streaming speech-to-text with [Suki](https://www.suki.ai)'s Dictation API, which is
tuned for clinical dictation.

See [Suki's developer docs](https://developer.suki.ai/documentation/how-to/dictation/dictation-basic-usage)
for more information about the Dictation API.

## Installation

```bash
pip install livekit-plugins-suki
```

## Pre-requisites

You need Suki partner credentials. Set them as environment variables or pass them to the
constructor (constructor arguments take precedence):

| Variable | Description |
| --- | --- |
| `SUKI_PARTNER_ID` | Partner ID issued by Suki |
| `SUKI_PARTNER_TOKEN` | Partner JWT (RS256) used to log in; it must not be expired |
| `SUKI_PROVIDER_ID` | Provider ID; required for Bearer and Single Auth Token partners |
| `SUKI_BASE_URL` | REST base URL, defaults to `https://sdp.suki.ai`; use `https://sdp.suki-stage.com` for staging |

## Usage

```python
from livekit.agents import AgentSession
from livekit.plugins import silero, suki

session = AgentSession(
    stt=suki.STT(language="English"),
    vad=silero.VAD.load(),
    # ...
)
```

The plugin:

- logs in with the partner credentials and caches the Suki token (refreshed before its one-hour
  expiry, and once more on a `401`),
- creates a Dictation session per stream attempt and ends it when the stream closes,
- streams 16 kHz mono PCM16 in ~100 ms `AUDIO` frames over `/ws/transcribe`,
- emits interim and final transcripts, `START_OF_SPEECH` / `END_OF_SPEECH` around each final
  segment, and recognition usage.

Suki does not provide end-of-turn detection, so use VAD or the turn detector for turn
handling rather than `turn_detection="stt"`. Single-shot (`recognize`) is not supported.

## Smoke test

`examples/other/suki_stt_smoke.py` streams a 16-bit WAV file through the plugin in real time,
then keeps the stream open on silence (`--hold-open`, default 10 s) before ending input, and
prints every event with an elapsed-time stamp:

```bash
export SUKI_PARTNER_ID=... SUKI_PARTNER_TOKEN=... SUKI_BASE_URL=https://sdp.suki-stage.com
uv run python examples/other/suki_stt_smoke.py path/to/dictation.wav --hold-open 10
```

On a live run, check that:

- at least one `FINAL` prints *before* the `end_input()` line. An agent keeps one stream open
  for the whole call, so finals must arrive on Suki's own segmentation, not only after
  `AUDIO_END`;
- no `FINAL:   EOF` line appears (the terminal EOF frame must not carry `is_final`).
