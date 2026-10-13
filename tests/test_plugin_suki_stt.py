from __future__ import annotations

import asyncio
import base64
import json
import logging
import struct
from collections.abc import Awaitable, Callable
from typing import Any

import aiohttp
import pytest
from aiohttp import web

from livekit import rtc
from livekit.agents import APIConnectionError, APIConnectOptions, APIStatusError, stt
from livekit.plugins.suki import STT

pytestmark = pytest.mark.unit

_SECRET_PARTNER_TOKEN = "partner-jwt-secret"
_TEXT = "the patient reports chest pain"

WSScript = Callable[[web.WebSocketResponse, "_FakeSuki"], Awaitable[None]]


def _pcm(num_samples: int) -> bytes:
    return struct.pack(f"<{num_samples}h", *(((i * 97) % 2000) - 1000 for i in range(num_samples)))


def _frame(text: str, *, is_final: bool, **extra: Any) -> dict[str, Any]:
    return {
        "transcript": {"transcript": text, "words": []},
        "is_final": is_final,
        "transcript_id": "01J9XABCDEFGHJKMNPQRSTVWXYZ",
        **extra,
    }


_EOF_FRAME = {"transcript": {"transcript": "EOF", "words": []}}


async def _until_audio_end(ws: web.WebSocketResponse, server: _FakeSuki) -> None:
    async for msg in ws:
        message = json.loads(msg.data)
        server.received.append(message)
        if message == {"type": "EVENT", "event": "AUDIO_END"}:
            return


async def _default_script(ws: web.WebSocketResponse, server: _FakeSuki) -> None:
    async for msg in ws:
        message = json.loads(msg.data)
        server.received.append(message)
        if message["type"] == "AUDIO" and len(server.audio_frames()) == 1:
            await ws.send_json(_frame("the patient", is_final=False))
        elif message == {"type": "EVENT", "event": "AUDIO_END"}:
            await ws.send_json(_frame(_TEXT, is_final=True, lang_id="en"))
            await ws.send_json(_EOF_FRAME)
            await ws.close()
            return


class _FakeSuki:
    """Local stand-in for Suki's login, session and /ws/transcribe endpoints."""

    def __init__(self, ws_script: WSScript = _default_script) -> None:
        self.ws_script = ws_script
        self.calls: list[str] = []
        self.logins = 0
        self.received: list[dict[str, Any]] = []
        self.create_headers: list[dict[str, str]] = []
        self.create_bodies: list[dict[str, Any]] = []
        self.ws_headers: list[dict[str, str]] = []
        self.ended: list[str] = []
        self.rejected_tokens: set[str] = set()
        self.ws_rejected_tokens: set[str] = set()
        self.reject_all_creates = False
        self.login_status = 200
        self.ws_attempts = 0

    def audio_frames(self) -> list[dict[str, Any]]:
        return [m for m in self.received if m["type"] == "AUDIO"]

    async def _login(self, request: web.Request) -> web.Response:
        self.calls.append("login")
        body = await request.json()
        assert body["partner_id"] == "partner-1"
        assert body["partner_token"] == _SECRET_PARTNER_TOKEN
        if self.login_status != 200:
            return web.json_response(
                {"code": self.login_status, "message": "forbidden"}, status=self.login_status
            )
        self.logins += 1
        return web.json_response({"suki_token": f"tok-{self.logins}"})

    async def _create(self, request: web.Request) -> web.Response:
        self.calls.append("create")
        token = request.headers.get("sdp_suki_token", "")
        self.create_headers.append(dict(request.headers))
        if self.reject_all_creates or token in self.rejected_tokens:
            return web.json_response({"code": 401, "message": "invalid token"}, status=401)
        self.create_bodies.append(await request.json())
        return web.json_response(
            {"transcription_session_id": f"sess-{len(self.create_bodies)}"}, status=201
        )

    async def _end(self, request: web.Request) -> web.Response:
        self.calls.append("end")
        self.ended.append(request.match_info["session_id"])
        return web.json_response({})

    async def _ws(self, request: web.Request) -> web.StreamResponse:
        self.calls.append("ws")
        self.ws_attempts += 1
        self.ws_headers.append(dict(request.headers))
        if request.headers.get("sdp_suki_token", "") in self.ws_rejected_tokens:
            return web.Response(status=401)
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await self.ws_script(ws, self)
        return ws

    async def __aenter__(self) -> _FakeSuki:
        app = web.Application()
        app.router.add_post("/api/v1/auth/login", self._login)
        app.router.add_post("/api/v1/transcription/session/create", self._create)
        app.router.add_post("/api/v1/transcription/session/{session_id}/end", self._end)
        app.router.add_get("/ws/transcribe", self._ws)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await site.start()
        port = self._runner.addresses[0][1]
        self.base_url = f"http://127.0.0.1:{port}"
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.session.close()
        await self._runner.cleanup()

    def stt(self, **kwargs: Any) -> STT:
        return STT(
            partner_id="partner-1",
            partner_token=_SECRET_PARTNER_TOKEN,
            base_url=self.base_url,
            http_session=self.session,
            **kwargs,
        )


async def _run_stream(
    suki_stt: STT,
    audio: bytes = b"",
    *,
    max_retry: int = 0,
) -> list[stt.SpeechEvent]:
    audio = audio or _pcm(16000 // 4)
    stream = suki_stt.stream(
        conn_options=APIConnectOptions(max_retry=max_retry, retry_interval=0.0, timeout=5.0)
    )
    stream.push_frame(
        rtc.AudioFrame(
            data=audio, sample_rate=16000, num_channels=1, samples_per_channel=len(audio) // 2
        )
    )
    stream.end_input()
    events: list[stt.SpeechEvent] = []
    try:
        async for event in stream:
            events.append(event)
    finally:
        await stream.aclose()
        await suki_stt.aclose()
    return events


def _types(events: list[stt.SpeechEvent]) -> list[stt.SpeechEventType]:
    return [e.type for e in events]


async def test_full_session_flow_headers_and_events() -> None:
    async with _FakeSuki() as server:
        audio = _pcm(16000 // 4)
        events = await _run_stream(server.stt(provider_id="provider-1"), audio)

    assert server.calls == ["login", "create", "ws", "end"]
    assert server.create_headers[0]["sdp_suki_token"] == "tok-1"
    assert server.create_headers[0]["sdp_provider_id"] == "provider-1"
    assert server.create_bodies[0] == {
        "audio_config": {
            "audio_encoding": "LINEAR16",
            "audio_language": "English",
            "sample_rate_hertz": 16000,
        }
    }
    assert server.ws_headers[0]["sdp_suki_token"] == "tok-1"
    assert server.ws_headers[0]["transcription_session_id"] == "sess-1"
    assert server.ws_headers[0]["sdp_provider_id"] == "provider-1"
    assert server.ended == ["sess-1"]

    audio_frames = server.audio_frames()
    # 250 ms of audio goes up as ~100 ms chunks
    assert len(audio_frames) == 3
    assert b"".join(base64.b64decode(m["audioData"]) for m in audio_frames) == audio
    assert server.received[-1] == {"type": "EVENT", "event": "AUDIO_END"}

    assert _types(events) == [
        stt.SpeechEventType.START_OF_SPEECH,
        stt.SpeechEventType.INTERIM_TRANSCRIPT,
        stt.SpeechEventType.FINAL_TRANSCRIPT,
        stt.SpeechEventType.RECOGNITION_USAGE,
        stt.SpeechEventType.END_OF_SPEECH,
    ]
    assert events[1].alternatives[0].text == "the patient"
    final = events[2].alternatives[0]
    assert final.text == _TEXT
    assert final.language == "en"
    assert events[2].request_id == "sess-1"
    assert events[3].recognition_usage is not None
    assert events[3].recognition_usage.audio_duration == pytest.approx(0.25)


async def test_dictated_eof_word_is_not_end_of_stream() -> None:
    async def script(ws: web.WebSocketResponse, server: _FakeSuki) -> None:
        await _until_audio_end(ws, server)
        await ws.send_json(_frame("EOF", is_final=True))
        await ws.send_json(_frame("next segment", is_final=True))
        await ws.send_json(_EOF_FRAME)
        await ws.close()

    async with _FakeSuki(script) as server:
        events = await _run_stream(server.stt())

    finals = [
        e.alternatives[0].text for e in events if e.type == stt.SpeechEventType.FINAL_TRANSCRIPT
    ]
    assert finals == ["EOF", "next segment"]


async def test_each_final_is_its_own_segment() -> None:
    async def script(ws: web.WebSocketResponse, server: _FakeSuki) -> None:
        await _until_audio_end(ws, server)
        await ws.send_json(_frame("first", is_final=False))
        await ws.send_json(_frame("first segment", is_final=True))
        await ws.send_json(_frame("second", is_final=False))
        await ws.send_json(_frame("second segment", is_final=True))
        await ws.send_json(_EOF_FRAME)
        await ws.close()

    async with _FakeSuki(script) as server:
        events = await _run_stream(server.stt())

    transcripts = [
        (e.type, e.alternatives[0].text)
        for e in events
        if e.type in (stt.SpeechEventType.INTERIM_TRANSCRIPT, stt.SpeechEventType.FINAL_TRANSCRIPT)
    ]
    assert transcripts == [
        (stt.SpeechEventType.INTERIM_TRANSCRIPT, "first"),
        (stt.SpeechEventType.FINAL_TRANSCRIPT, "first segment"),
        (stt.SpeechEventType.INTERIM_TRANSCRIPT, "second"),
        (stt.SpeechEventType.FINAL_TRANSCRIPT, "second segment"),
    ]
    assert _types(events).count(stt.SpeechEventType.START_OF_SPEECH) == 2
    assert _types(events).count(stt.SpeechEventType.END_OF_SPEECH) == 2


async def test_malformed_frames_are_skipped_without_logging_content(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def script(ws: web.WebSocketResponse, server: _FakeSuki) -> None:
        await _until_audio_end(ws, server)
        await ws.send_str("not json")
        await ws.send_json({"transcript": {"transcript": "secret words"}, "is_final": "yes"})
        await ws.send_json({"code": 500, "message": "secret words"})
        await ws.send_json(_frame(_TEXT, is_final=True))
        await ws.send_json(_EOF_FRAME)
        await ws.close()

    caplog.set_level(logging.DEBUG, logger="livekit.plugins.suki")
    async with _FakeSuki(script) as server:
        events = await _run_stream(server.stt())

    finals = [
        e.alternatives[0].text for e in events if e.type == stt.SpeechEventType.FINAL_TRANSCRIPT
    ]
    assert finals == [_TEXT]
    assert caplog.records
    for record in caplog.records:
        assert "secret" not in record.getMessage()
        assert "secret" not in str(record.__dict__)


async def test_token_is_cached_across_streams() -> None:
    async with _FakeSuki() as server:
        suki_stt = server.stt()
        await _run_stream(suki_stt)
        await _run_stream(suki_stt)

    assert server.logins == 1
    assert [h["sdp_suki_token"] for h in server.create_headers] == ["tok-1", "tok-1"]


async def test_rejected_cached_token_relogs_in_once_for_rest() -> None:
    async with _FakeSuki() as server:
        suki_stt = server.stt()
        assert await suki_stt._get_token(timeout=5.0) == "tok-1"
        server.rejected_tokens.add("tok-1")
        events = await _run_stream(suki_stt)

    assert server.logins == 2
    assert [h["sdp_suki_token"] for h in server.create_headers] == ["tok-1", "tok-2"]
    assert server.ws_headers[0]["sdp_suki_token"] == "tok-2"
    assert stt.SpeechEventType.FINAL_TRANSCRIPT in _types(events)


async def test_rejected_token_relogs_in_once_for_websocket() -> None:
    async with _FakeSuki() as server:
        server.ws_rejected_tokens.add("tok-1")
        events = await _run_stream(server.stt())

    assert server.logins == 2
    assert [h["sdp_suki_token"] for h in server.ws_headers] == ["tok-1", "tok-2"]
    assert stt.SpeechEventType.FINAL_TRANSCRIPT in _types(events)


async def test_persistent_401_fails_after_a_single_relogin() -> None:
    async with _FakeSuki() as server:
        server.reject_all_creates = True
        with pytest.raises(APIStatusError) as exc_info:
            await _run_stream(server.stt(), max_retry=0)

    assert server.logins == 2
    assert exc_info.value.status_code == 401
    assert exc_info.value.retryable is False
    assert "ws" not in server.calls


async def test_login_failure_does_not_leak_credentials() -> None:
    async with _FakeSuki() as server:
        server.login_status = 403
        with pytest.raises(APIStatusError) as exc_info:
            await _run_stream(server.stt(), max_retry=0)

    error = exc_info.value
    assert error.status_code == 403
    assert error.retryable is False
    assert _SECRET_PARTNER_TOKEN not in str(error)
    assert error.__cause__ is None


async def test_connection_errors_are_not_chained() -> None:
    suki_stt = STT(
        partner_id="partner-1",
        partner_token=_SECRET_PARTNER_TOKEN,
        base_url="http://127.0.0.1:9",
    )
    async with aiohttp.ClientSession() as session:
        suki_stt._session = session
        with pytest.raises(APIConnectionError) as exc_info:
            await _run_stream(suki_stt, max_retry=0)

    error = exc_info.value
    assert error.__cause__ is None
    assert error.__suppress_context__ or error.__context__ is None
    assert _SECRET_PARTNER_TOKEN not in str(error)


async def test_early_close_retries_with_a_fresh_session() -> None:
    attempts = 0

    async def script(ws: web.WebSocketResponse, server: _FakeSuki) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            await ws.close()
            return
        await _default_script(ws, server)

    async with _FakeSuki(script) as server:
        # keep input open so the first close lands mid-stream
        suki_stt = server.stt()
        stream = suki_stt.stream(
            conn_options=APIConnectOptions(max_retry=1, retry_interval=0.0, timeout=5.0)
        )
        audio = _pcm(1600)
        stream.push_frame(
            rtc.AudioFrame(data=audio, sample_rate=16000, num_channels=1, samples_per_channel=1600)
        )

        async def _end_later() -> None:
            while server.ws_attempts < 2:
                await asyncio.sleep(0.01)
            stream.push_frame(
                rtc.AudioFrame(
                    data=audio, sample_rate=16000, num_channels=1, samples_per_channel=1600
                )
            )
            stream.end_input()

        ender = asyncio.create_task(_end_later())
        events = [e async for e in stream]
        await ender
        await stream.aclose()
        await suki_stt.aclose()

    assert server.calls.count("create") == 2
    assert sorted(server.ended) == ["sess-1", "sess-2"]
    assert stt.SpeechEventType.FINAL_TRANSCRIPT in _types(events)


async def test_early_close_without_retries_raises_retryable_error() -> None:
    async def script(ws: web.WebSocketResponse, server: _FakeSuki) -> None:
        await ws.close()

    async with _FakeSuki(script) as server:
        suki_stt = server.stt()
        stream = suki_stt.stream(
            conn_options=APIConnectOptions(max_retry=0, retry_interval=0.0, timeout=5.0)
        )
        stream.push_frame(
            rtc.AudioFrame(
                data=_pcm(1600), sample_rate=16000, num_channels=1, samples_per_channel=1600
            )
        )
        with pytest.raises(APIConnectionError) as exc_info:
            async for _ in stream:
                pass
        await stream.aclose()
        await suki_stt.aclose()

    assert exc_info.value.retryable is True
    assert server.ended == ["sess-1"]


async def test_close_after_audio_end_without_eof_still_finishes() -> None:
    async def script(ws: web.WebSocketResponse, server: _FakeSuki) -> None:
        await _until_audio_end(ws, server)
        await ws.send_json(_frame(_TEXT, is_final=True))
        await ws.close()

    async with _FakeSuki(script) as server:
        events = await _run_stream(server.stt())

    assert stt.SpeechEventType.FINAL_TRANSCRIPT in _types(events)
    assert server.ended == ["sess-1"]


async def test_open_segment_is_closed_when_stream_ends() -> None:
    async def script(ws: web.WebSocketResponse, server: _FakeSuki) -> None:
        await _until_audio_end(ws, server)
        await ws.send_json(_frame("dangling", is_final=False))
        await ws.send_json(_EOF_FRAME)
        await ws.close()

    async with _FakeSuki(script) as server:
        events = await _run_stream(server.stt())

    assert _types(events) == [
        stt.SpeechEventType.START_OF_SPEECH,
        stt.SpeechEventType.INTERIM_TRANSCRIPT,
        stt.SpeechEventType.RECOGNITION_USAGE,
        stt.SpeechEventType.END_OF_SPEECH,
    ]


async def test_recognize_is_not_supported() -> None:
    suki_stt = STT(partner_id="p", partner_token="t")
    with pytest.raises(NotImplementedError):
        await suki_stt._recognize_impl(
            rtc.AudioFrame(data=b"", sample_rate=16000, num_channels=1, samples_per_channel=0)
        )


def test_credentials_from_env_with_constructor_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SUKI_PARTNER_ID", "env-partner")
    monkeypatch.setenv("SUKI_PARTNER_TOKEN", "env-token")
    monkeypatch.setenv("SUKI_PROVIDER_ID", "env-provider")
    monkeypatch.setenv("SUKI_BASE_URL", "https://sdp.suki-stage.com/")

    from_env = STT()
    assert from_env._partner_id == "env-partner"
    assert from_env._partner_token == "env-token"
    assert from_env._provider_id == "env-provider"
    assert from_env._base_url == "https://sdp.suki-stage.com"
    assert from_env._ws_url() == "wss://sdp.suki-stage.com/ws/transcribe"

    explicit = STT(partner_id="arg-partner", provider_id="arg-provider", base_url="http://x")
    assert explicit._partner_id == "arg-partner"
    assert explicit._partner_token == "env-token"
    assert explicit._provider_id == "arg-provider"
    assert explicit._ws_url() == "ws://x/ws/transcribe"


def test_missing_credentials_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SUKI_PARTNER_ID", raising=False)
    monkeypatch.delenv("SUKI_PARTNER_TOKEN", raising=False)
    with pytest.raises(ValueError):
        STT()


async def test_stereo_input_is_downmixed_to_mono() -> None:
    left, right = 1000, -400
    stereo = struct.pack("<2h", left, right) * 1600

    async with _FakeSuki() as server:
        suki_stt = server.stt()
        stream = suki_stt.stream(
            conn_options=APIConnectOptions(max_retry=0, retry_interval=0.0, timeout=5.0)
        )
        stream.push_frame(
            rtc.AudioFrame(data=stereo, sample_rate=16000, num_channels=2, samples_per_channel=1600)
        )
        stream.end_input()
        events = [e async for e in stream]
        await stream.aclose()
        await suki_stt.aclose()

    sent = b"".join(base64.b64decode(m["audioData"]) for m in server.audio_frames())
    assert sent == struct.pack("<h", (left + right) // 2) * 1600
    usage = [e for e in events if e.type == stt.SpeechEventType.RECOGNITION_USAGE]
    assert usage[0].recognition_usage is not None
    assert usage[0].recognition_usage.audio_duration == pytest.approx(0.1)


async def test_finals_arrive_mid_stream_before_audio_end() -> None:
    """The agent keeps one stream open per call, so finals must flow before AUDIO_END."""

    async def script(ws: web.WebSocketResponse, server: _FakeSuki) -> None:
        async for msg in ws:
            message = json.loads(msg.data)
            server.received.append(message)
            audio_count = len(server.audio_frames())
            if message["type"] == "AUDIO" and audio_count == 1:
                await ws.send_json(_frame("first", is_final=False))
                await ws.send_json(_frame("first segment", is_final=True))
            elif message["type"] == "AUDIO" and audio_count == 3:
                await ws.send_json(_frame("second segment", is_final=True))
            elif message == {"type": "EVENT", "event": "AUDIO_END"}:
                await ws.send_json(_EOF_FRAME)
                await ws.close()
                return

    chunk = _pcm(1600)

    def _push(stream: stt.SpeechStream) -> None:
        stream.push_frame(
            rtc.AudioFrame(data=chunk, sample_rate=16000, num_channels=1, samples_per_channel=1600)
        )

    async with _FakeSuki(script) as server:
        suki_stt = server.stt()
        stream = suki_stt.stream(
            conn_options=APIConnectOptions(max_retry=0, retry_interval=0.0, timeout=5.0)
        )
        events: list[stt.SpeechEvent] = []
        _push(stream)

        async def _next_final() -> None:
            async for event in stream:
                events.append(event)
                if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                    return

        await asyncio.wait_for(_next_final(), 5.0)
        assert server.received[-1]["type"] == "AUDIO"
        assert len(server.audio_frames()) == 1

        _push(stream)
        _push(stream)
        await asyncio.wait_for(_next_final(), 5.0)
        stream.end_input()
        events.extend([e async for e in stream])
        await stream.aclose()
        await suki_stt.aclose()

    assert [
        e.alternatives[0].text for e in events if e.type == stt.SpeechEventType.FINAL_TRANSCRIPT
    ] == ["first segment", "second segment"]
    assert _types(events).count(stt.SpeechEventType.START_OF_SPEECH) == 2
    assert _types(events).count(stt.SpeechEventType.END_OF_SPEECH) == 2
    usage = [
        e.recognition_usage.audio_duration
        for e in events
        if e.type == stt.SpeechEventType.RECOGNITION_USAGE and e.recognition_usage is not None
    ]
    assert usage == [pytest.approx(0.1), pytest.approx(0.2)]
