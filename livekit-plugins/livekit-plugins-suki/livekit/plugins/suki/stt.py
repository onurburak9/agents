# Copyright 2026 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Suki Dictation STT plugin for LiveKit Agents.

Each stream attempt runs one Dictation session:

    POST /api/v1/auth/login                          -> suki_token (valid ~1h)
    POST /api/v1/transcription/session/create        -> transcription_session_id
    GET  /ws/transcribe                              (headers: sdp_suki_token,
                                                      transcription_session_id)
    POST /api/v1/transcription/session/{id}/end      (best effort)

Wire contract on /ws/transcribe (JSON text frames only):

    client -> server  {"type": "AUDIO", "audioData": <base64 PCM_S16LE>}
                      {"type": "EVENT", "event": "AUDIO_END"}
    server -> client  {"transcript": {"transcript": ..., "words": [...]},
                       "is_final": bool, "transcript_id": ..., "lang_id": ...}
                      {"transcript": {"transcript": "EOF", "words": []}}

Interim frames carry the draft of the current segment and replace each other; a
final commits that segment and is never revised. The EOF frame is the only one
without `is_final`, which is how it is told apart from a dictated word "EOF".

Transcripts are clinical dictation, so neither transcript text nor credentials
are ever logged, and aiohttp errors (whose request info carries the auth
headers) are never chained onto raised exceptions.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import time
from typing import Any

import aiohttp
import numpy as np

from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APIStatusError,
    APITimeoutError,
    stt,
    utils,
)
from livekit.agents.language import LanguageCode
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

from .log import logger

DEFAULT_BASE_URL = "https://sdp.suki.ai"
SAMPLE_RATE = 16000
# ~100 ms of 16 kHz mono PCM16 per AUDIO frame, as the wire format guide recommends
_SAMPLES_PER_CHUNK = SAMPLE_RATE // 10

# login does not return an expiry; Suki documents a one-hour lifetime
_TOKEN_TTL_SECONDS = 3600.0
_TOKEN_REFRESH_MARGIN_SECONDS = 300.0

_EOF_TIMEOUT_SECONDS = 10.0
_END_SESSION_TIMEOUT_SECONDS = 5.0

_EOF_TEXT = "EOF"


class STT(stt.STT):
    def __init__(
        self,
        *,
        partner_id: NotGivenOr[str] = NOT_GIVEN,
        partner_token: NotGivenOr[str] = NOT_GIVEN,
        provider_id: NotGivenOr[str] = NOT_GIVEN,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        language: str = "English",
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """Create a new instance of Suki Dictation STT.

        Credentials fall back to the ``SUKI_PARTNER_ID``, ``SUKI_PARTNER_TOKEN``,
        ``SUKI_PROVIDER_ID`` and ``SUKI_BASE_URL`` environment variables.

        Args:
            partner_id: Partner ID issued by Suki.
            partner_token: Partner JWT used to log in. It must not be expired.
            provider_id: Provider ID, required for Bearer and Single Auth Token partners.
            base_url: REST base URL, e.g. ``https://sdp.suki-stage.com``. The
                WebSocket uses the same host. Defaults to production.
            language: ``audio_language`` for the Dictation session. Suki documents
                ``English`` and ``Multilingual``.
            http_session: Optional aiohttp session to reuse.
        """
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=True,
                aligned_transcript=False,
                offline_recognize=False,
            ),
        )
        suki_partner_id = (
            partner_id if is_given(partner_id) else os.environ.get("SUKI_PARTNER_ID", "")
        )
        suki_partner_token = (
            partner_token if is_given(partner_token) else os.environ.get("SUKI_PARTNER_TOKEN", "")
        )
        if not suki_partner_id or not suki_partner_token:
            raise ValueError(
                "Suki partner_id and partner_token are required, either as arguments or by "
                "setting the SUKI_PARTNER_ID and SUKI_PARTNER_TOKEN environment variables",
            )
        self._partner_id = suki_partner_id
        self._partner_token = suki_partner_token
        self._provider_id = (
            provider_id if is_given(provider_id) else os.environ.get("SUKI_PROVIDER_ID") or None
        )
        self._base_url = (
            base_url if is_given(base_url) else os.environ.get("SUKI_BASE_URL", DEFAULT_BASE_URL)
        ).rstrip("/")
        self._language = language
        self._session = http_session

        self._token: str | None = None
        self._token_expires_at = 0.0
        self._token_lock = asyncio.Lock()
        self._end_session_tasks: set[asyncio.Task[None]] = set()

    @property
    def model(self) -> str:
        return "dictation"

    @property
    def provider(self) -> str:
        return "suki"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    def _ws_url(self) -> str:
        if self._base_url.startswith("https://"):
            return "wss://" + self._base_url[len("https://") :] + "/ws/transcribe"
        if self._base_url.startswith("http://"):
            return "ws://" + self._base_url[len("http://") :] + "/ws/transcribe"
        return self._base_url + "/ws/transcribe"

    def _auth_headers(self, token: str) -> dict[str, str]:
        headers = {"sdp_suki_token": token}
        if self._provider_id:
            headers["sdp_provider_id"] = self._provider_id
        return headers

    async def _get_token(self, *, timeout: float) -> str:
        async with self._token_lock:
            if self._token and time.monotonic() < self._token_expires_at:
                return self._token
            token = await self._login(timeout=timeout)
            self._token = token
            self._token_expires_at = (
                time.monotonic() + _TOKEN_TTL_SECONDS - _TOKEN_REFRESH_MARGIN_SECONDS
            )
            return token

    def _invalidate_token(self, token: str) -> None:
        # only drop the token that was rejected; a concurrent stream may already
        # have refreshed it
        if self._token == token:
            self._token = None
            self._token_expires_at = 0.0

    async def _login(self, *, timeout: float) -> str:
        body: dict[str, str] = {
            "partner_id": self._partner_id,
            "partner_token": self._partner_token,
        }
        if self._provider_id:
            body["provider_id"] = self._provider_id

        status, payload = await self._post_json(
            "/api/v1/auth/login", body=body, headers={}, timeout=timeout, what="login"
        )
        if status != 200:
            raise _status_error("suki login failed", status)
        token = payload.get("suki_token") if isinstance(payload, dict) else None
        if not isinstance(token, str) or not token:
            raise APIConnectionError("suki login returned no token", retryable=False)
        return token

    async def _authed_post(
        self, path: str, *, body: dict[str, Any] | None, timeout: float, what: str
    ) -> tuple[int, Any]:
        """POST with the cached Suki token, logging in again once on a 401."""
        for attempt in range(2):
            token = await self._get_token(timeout=timeout)
            status, payload = await self._post_json(
                path, body=body, headers=self._auth_headers(token), timeout=timeout, what=what
            )
            if status == 401 and attempt == 0:
                self._invalidate_token(token)
                continue
            return status, payload
        raise AssertionError("unreachable")

    async def _post_json(
        self,
        path: str,
        *,
        body: dict[str, Any] | None,
        headers: dict[str, str],
        timeout: float,
        what: str,
    ) -> tuple[int, Any]:
        try:
            async with self._ensure_session().post(
                self._base_url + path,
                json=body,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as resp:
                try:
                    payload = await resp.json(content_type=None)
                except (json.JSONDecodeError, aiohttp.ContentTypeError, ValueError):
                    payload = None
                return resp.status, payload
        except asyncio.TimeoutError:
            raise APITimeoutError(f"suki {what} timed out") from None
        except aiohttp.ClientError as e:
            # never chain: aiohttp errors carry the request headers (the token)
            raise APIConnectionError(f"suki {what} failed ({type(e).__name__})") from None

    async def _create_session(self, *, language: str, timeout: float) -> str:
        status, payload = await self._authed_post(
            "/api/v1/transcription/session/create",
            body={
                "audio_config": {
                    "audio_encoding": "LINEAR16",
                    "audio_language": language,
                    "sample_rate_hertz": SAMPLE_RATE,
                }
            },
            timeout=timeout,
            what="session create",
        )
        if status not in (200, 201):
            raise _status_error("suki session create failed", status)
        session_id = payload.get("transcription_session_id") if isinstance(payload, dict) else None
        if not isinstance(session_id, str) or not session_id:
            raise APIConnectionError("suki session create returned no session id")
        return session_id

    async def _end_session(self, session_id: str) -> None:
        try:
            status, _ = await asyncio.wait_for(
                self._authed_post(
                    f"/api/v1/transcription/session/{session_id}/end",
                    body=None,
                    timeout=_END_SESSION_TIMEOUT_SECONDS,
                    what="session end",
                ),
                _END_SESSION_TIMEOUT_SECONDS * 2,
            )
            if status != 200:
                logger.warning("suki session end returned status %d", status)
        except Exception as e:
            logger.warning("suki session end failed (%s)", type(e).__name__)

    def _schedule_end_session(self, session_id: str) -> None:
        # run detached so a cancelled stream (aclose) still releases the session
        task = asyncio.create_task(self._end_session(session_id))
        self._end_session_tasks.add(task)
        task.add_done_callback(self._end_session_tasks.discard)

    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        raise NotImplementedError("suki STT does not support single-shot recognition")

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStream:
        return SpeechStream(
            stt_instance=self,
            conn_options=conn_options,
            language=language if is_given(language) else self._language,
        )

    async def aclose(self) -> None:
        if self._end_session_tasks:
            await asyncio.gather(*self._end_session_tasks, return_exceptions=True)
        await super().aclose()


def _mono_pcm(frame: rtc.AudioFrame) -> bytes:
    """Suki expects mono; the base stream resamples but keeps the channel count."""
    if frame.num_channels == 1:
        return frame.data.tobytes()
    samples = np.frombuffer(frame.data, dtype=np.int16).reshape(-1, frame.num_channels)
    return samples.mean(axis=1).astype(np.int16).tobytes()


def _status_error(message: str, status: int) -> APIStatusError:
    return APIStatusError(
        f"{message} (status {status})",
        status_code=status,
        request_id=None,
        body=None,
        retryable=status == 429 or status >= 500,
    )


class SpeechStream(stt.SpeechStream):
    def __init__(
        self,
        *,
        stt_instance: STT,
        conn_options: APIConnectOptions,
        language: str,
    ) -> None:
        super().__init__(stt=stt_instance, conn_options=conn_options, sample_rate=SAMPLE_RATE)
        self._stt: STT = stt_instance
        self._session_language = language
        self._language = LanguageCode(language)
        self._session_id = ""
        self._reset_state()

    def _reset_state(self) -> None:
        self._input_done = False
        self._is_speaking = False
        self._audio_sent = 0.0
        self._reported_usage = 0.0
        self._segment_start = 0.0

    async def _run(self) -> None:
        self._reset_state()
        timeout = self._conn_options.timeout
        # every attempt (including retries) gets a fresh Dictation session, so a
        # reconnect never races the previous session's FailedPrecondition window
        self._session_id = await self._stt._create_session(
            language=self._session_language, timeout=timeout
        )
        try:
            ws = await self._connect_ws(timeout=timeout)
            try:
                await self._stream(ws)
            finally:
                await ws.close()
        finally:
            self._finalize()
            self._stt._schedule_end_session(self._session_id)

    async def _connect_ws(self, *, timeout: float) -> aiohttp.ClientWebSocketResponse:
        for attempt in range(2):
            token = await self._stt._get_token(timeout=timeout)
            headers = self._stt._auth_headers(token)
            headers["transcription_session_id"] = self._session_id
            try:
                return await asyncio.wait_for(
                    self._stt._ensure_session().ws_connect(self._stt._ws_url(), headers=headers),
                    timeout,
                )
            except aiohttp.WSServerHandshakeError as e:
                if e.status == 401 and attempt == 0:
                    self._stt._invalidate_token(token)
                    continue
                raise _status_error("suki websocket handshake failed", e.status) from None
            except asyncio.TimeoutError:
                raise APITimeoutError("suki websocket connect timed out") from None
            except aiohttp.ClientError as e:
                raise APIConnectionError(
                    f"suki websocket connect failed ({type(e).__name__})"
                ) from None
        raise AssertionError("unreachable")

    async def _stream(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        send_task = asyncio.create_task(self._send_task(ws))
        recv_task = asyncio.create_task(self._recv_task(ws))
        try:
            done, _ = await asyncio.wait(
                (send_task, recv_task), return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                task.result()

            if recv_task in done:
                # the server finished before we sent AUDIO_END
                raise APIConnectionError("suki STT ended the stream early", retryable=True)

            try:
                await asyncio.wait_for(asyncio.shield(recv_task), _EOF_TIMEOUT_SECONDS)
            except asyncio.TimeoutError:
                logger.warning("suki STT did not send EOF after AUDIO_END")
        finally:
            await utils.aio.gracefully_cancel(send_task, recv_task)

    async def _send_task(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        audio_bstream = utils.audio.AudioByteStream(
            sample_rate=SAMPLE_RATE, num_channels=1, samples_per_channel=_SAMPLES_PER_CHUNK
        )
        try:
            async for data in self._input_ch:
                if isinstance(data, self._FlushSentinel):
                    frames = audio_bstream.flush()
                else:
                    frames = audio_bstream.write(_mono_pcm(data))
                for frame in frames:
                    await self._send_audio(ws, frame)

            for frame in audio_bstream.flush():
                await self._send_audio(ws, frame)

            self._input_done = True
            await ws.send_str(json.dumps({"type": "EVENT", "event": "AUDIO_END"}))
        except Exception as e:
            raise APIConnectionError(
                f"suki STT send failed ({type(e).__name__})", retryable=self._audio_sent == 0.0
            ) from None

    async def _send_audio(self, ws: aiohttp.ClientWebSocketResponse, frame: rtc.AudioFrame) -> None:
        audio = base64.b64encode(frame.data.tobytes()).decode("ascii")
        await ws.send_str(json.dumps({"type": "AUDIO", "audioData": audio}))
        self._audio_sent += frame.samples_per_channel / frame.sample_rate

    async def _recv_task(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        while True:
            try:
                msg = await ws.receive()
            except Exception as e:
                raise APIConnectionError(f"suki STT receive failed ({type(e).__name__})") from None

            if msg.type in (
                aiohttp.WSMsgType.CLOSED,
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSING,
            ):
                if self._input_done:
                    return
                raise APIConnectionError("suki STT closed unexpectedly")
            if msg.type == aiohttp.WSMsgType.ERROR:
                raise APIConnectionError("suki STT websocket failed")
            if msg.type != aiohttp.WSMsgType.TEXT:
                continue

            try:
                if self._process_message(json.loads(msg.data)):
                    return
            except APIError:
                raise
            except Exception:
                # never attach the exception or payload: it may contain transcript text
                logger.warning("suki STT returned malformed data")

    def _process_message(self, data: Any) -> bool:
        """Handle one inbound frame. Returns True on the terminal EOF frame."""
        if not isinstance(data, dict):
            raise ValueError("suki frame must be a JSON object")

        transcript = data.get("transcript")
        if not isinstance(transcript, dict):
            # error frames are not documented; log the shape, never the values
            logger.warning(
                "suki STT sent an unrecognized frame", extra={"keys": sorted(data.keys())}
            )
            return False

        text = transcript.get("transcript")
        if not isinstance(text, str):
            raise ValueError("suki transcript text must be a string")

        if "is_final" not in data:
            if text == _EOF_TEXT:
                return True
            raise ValueError("suki transcript frame is missing is_final")

        is_final = data["is_final"]
        if not isinstance(is_final, bool):
            raise ValueError("suki is_final must be a boolean")

        text = text.strip()
        if not text:
            return False

        lang_id = data.get("lang_id")
        language = LanguageCode(lang_id) if isinstance(lang_id, str) and lang_id else self._language

        if not self._is_speaking:
            self._is_speaking = True
            self._event_ch.send_nowait(
                stt.SpeechEvent(
                    type=stt.SpeechEventType.START_OF_SPEECH,
                    request_id=self._session_id,
                    speech_start_time=self.start_time + self._segment_start,
                )
            )

        start_time = self._segment_start + self.start_time_offset
        end_time = self._audio_sent + self.start_time_offset
        speech_data = stt.SpeechData(
            language=language, text=text, start_time=start_time, end_time=end_time
        )

        if not is_final:
            self._event_ch.send_nowait(
                stt.SpeechEvent(
                    type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                    request_id=self._session_id,
                    alternatives=[speech_data],
                )
            )
            return False

        self._event_ch.send_nowait(
            stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                request_id=self._session_id,
                alternatives=[speech_data],
            )
        )
        self._emit_usage()
        self._emit_end_of_speech()
        self._segment_start = self._audio_sent
        return False

    def _emit_usage(self) -> None:
        duration = self._audio_sent - self._reported_usage
        if duration <= 0:
            return
        self._event_ch.send_nowait(
            stt.SpeechEvent(
                type=stt.SpeechEventType.RECOGNITION_USAGE,
                request_id=self._session_id,
                recognition_usage=stt.RecognitionUsage(audio_duration=duration),
            )
        )
        self._reported_usage = self._audio_sent

    def _emit_end_of_speech(self) -> None:
        if not self._is_speaking:
            return
        self._is_speaking = False
        self._event_ch.send_nowait(
            stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH, request_id=self._session_id)
        )

    def _finalize(self) -> None:
        """Close out the attempt: report remaining usage, release an open segment."""
        self._emit_usage()
        self._emit_end_of_speech()
