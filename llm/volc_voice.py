#!/usr/bin/env python3
"""
火山引擎（豆包）语音协议：大模型流式 ASR + seed-tts-2.0 双向流式 TTS。

二进制 WebSocket 协议（4 字节头 + 可选事件/序列 + payload），整型大端。
改写自生产级参考实现，精简到两个 async 生成器：

  asr_stream(audio_aiter)  -> 产出 {text, is_final, utterances}
  tts_stream(text, ...)    -> 产出原始音频字节块（PCM/MP3）

文档：
  ASR  https://www.volcengine.com/docs/6561/1354869
  TTS  https://www.volcengine.com/docs/6561/1329505
"""

from __future__ import annotations

import asyncio
import contextlib
import gzip
import io
import json
import logging
import struct
import uuid
from dataclasses import dataclass, field
from enum import IntEnum
from typing import AsyncIterator, Optional, Union

import websockets
from websockets.exceptions import WebSocketException

import appconfig

logger = logging.getLogger("volc_voice")

# ── 凭证与端点（来自 config.yaml，可用环境变量覆盖）──────────────────────────
DEFAULT_APP_ID       = appconfig.get("volcengine.app_id",       env="VOLC_APP_ID")
DEFAULT_ACCESS_TOKEN = appconfig.get("volcengine.access_token", env="VOLC_ACCESS_TOKEN")

# 账号开通的是「豆包流式语音识别模型 2.0」(seedasr，仅 bigmodel_async / nostream 接受该资源)
ASR_URL = appconfig.get("volcengine.asr_url", env="VOLC_ASR_URL")
TTS_URL = appconfig.get("volcengine.tts_url", env="VOLC_TTS_URL")
ASR_RESOURCE_ID = appconfig.get("volcengine.asr_resource_id", env="VOLC_ASR_RESOURCE_ID")
TTS_RESOURCE_ID = appconfig.get("volcengine.tts_resource_id", env="VOLC_TTS_RESOURCE_ID")


# ── 异常 ──────────────────────────────────────────────────────────────────────
class VolcVoiceError(Exception):
    def __init__(self, message, code=None, payload=None):
        super().__init__(message)
        self.code = code
        self.payload = payload

    def __str__(self):
        base = super().__str__()
        return f"[{self.code}] {base}" if self.code is not None else base


class VolcAuthError(VolcVoiceError):    pass
class VolcParamError(VolcVoiceError):   pass
class VolcServerError(VolcVoiceError):  pass
class VolcTimeoutError(VolcVoiceError): pass


def _raise_for_code(code: int, message: str, payload=None) -> None:
    if code == 45000001 or code in (401, 403):
        raise VolcAuthError(message, code=code, payload=payload)
    if code == 45000002 or 45000000 <= code < 46000000:
        raise VolcParamError(message, code=code, payload=payload)
    raise VolcServerError(message, code=code, payload=payload)


# ── 协议枚举 ──────────────────────────────────────────────────────────────────
class MsgType(IntEnum):
    Invalid            = 0
    FullClientRequest  = 0b0001
    AudioOnlyClient    = 0b0010
    FullServerResponse = 0b1001
    AudioOnlyServer    = 0b1011
    FrontEndResultServer = 0b1100
    Error              = 0b1111


class Flags(IntEnum):
    NoSeq       = 0b0000
    PositiveSeq = 0b0001
    LastNoSeq   = 0b0010
    NegativeSeq = 0b0011
    WithEvent   = 0b0100


class Serialization(IntEnum):
    Raw  = 0
    JSON = 0b0001


class Compression(IntEnum):
    None_ = 0
    Gzip  = 0b0001


class Event(IntEnum):
    None_ = 0
    StartConnection   = 1
    FinishConnection  = 2
    ConnectionStarted = 50
    ConnectionFailed  = 51
    ConnectionFinished = 52
    StartSession  = 100
    CancelSession = 101
    FinishSession = 102
    SessionStarted  = 150
    SessionCanceled = 151
    SessionFinished = 152
    SessionFailed   = 153
    TaskRequest  = 200
    UpdateConfig = 201
    TTSSentenceStart = 350
    TTSSentenceEnd   = 351
    TTSResponse      = 352
    TTSEnded         = 359
    ASRInfo     = 450
    ASRResponse = 451
    ASREnded    = 459


# 不带 session_id 的连接级事件
_CONNECTION_EVENTS = frozenset({
    Event.StartConnection, Event.FinishConnection,
    Event.ConnectionStarted, Event.ConnectionFailed, Event.ConnectionFinished,
})


@dataclass
class Message:
    """TTS / ASR 共用的线缆消息。marshal/unmarshal 按 (type, flag) 决定可选字段。"""

    type: MsgType = MsgType.Invalid
    flag: Flags = Flags.NoSeq
    serialization: Serialization = Serialization.JSON
    compression: Compression = Compression.None_

    event: Event = Event.None_
    session_id: str = ""
    connect_id: str = ""
    sequence: int = 0
    error_code: int = 0

    payload: bytes = field(default=b"")

    def json_payload(self) -> Optional[dict]:
        if not self.payload:
            return None
        raw = self.payload
        try:
            if self.compression == Compression.Gzip:
                raw = gzip.decompress(raw)
            return json.loads(raw.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return None

    def marshal(self) -> bytes:
        buf = io.BytesIO()
        buf.write(bytes([
            (1 << 4) | 1,                                          # version=1, header_size=1 (=4 字节)
            (int(self.type) << 4) | int(self.flag),
            (int(self.serialization) << 4) | int(self.compression),
            0x00,
        ]))
        if self.flag == Flags.WithEvent:
            buf.write(struct.pack(">i", int(self.event)))
            if self.event not in _CONNECTION_EVENTS:
                sid = self.session_id.encode("utf-8")
                buf.write(struct.pack(">I", len(sid)))
                if sid:
                    buf.write(sid)

        if self.type == MsgType.Error:
            buf.write(struct.pack(">I", self.error_code & 0xFFFFFFFF))
        elif self.flag in (Flags.PositiveSeq, Flags.NegativeSeq) and self.type in (
            MsgType.FullClientRequest, MsgType.FullServerResponse,
            MsgType.FrontEndResultServer, MsgType.AudioOnlyClient, MsgType.AudioOnlyServer,
        ):
            buf.write(struct.pack(">i", self.sequence))

        buf.write(struct.pack(">I", len(self.payload)))
        if self.payload:
            buf.write(self.payload)
        return buf.getvalue()

    @classmethod
    def unmarshal(cls, data: bytes) -> "Message":
        if len(data) < 4:
            raise ValueError(f"frame too short: {len(data)} bytes")
        header_size = (data[0] & 0x0F) * 4
        if header_size < 4:
            raise ValueError(f"invalid header_size {header_size}")

        msg = cls(
            type=MsgType(data[1] >> 4),
            flag=Flags(data[1] & 0x0F),
            serialization=Serialization(data[2] >> 4),
            compression=Compression(data[2] & 0x0F),
        )
        buf = io.BytesIO(data[header_size:])

        if msg.flag == Flags.WithEvent:
            msg.event = Event(struct.unpack(">i", buf.read(4))[0])
            if msg.event not in _CONNECTION_EVENTS:
                sid_size = struct.unpack(">I", buf.read(4))[0]
                if sid_size:
                    msg.session_id = buf.read(sid_size).decode("utf-8", errors="replace")
            elif msg.type == MsgType.FullServerResponse:
                cid_size_bytes = buf.read(4)
                if cid_size_bytes:
                    cid_size = struct.unpack(">I", cid_size_bytes)[0]
                    if cid_size:
                        msg.connect_id = buf.read(cid_size).decode("utf-8", errors="replace")

        if msg.type == MsgType.Error:
            msg.error_code = struct.unpack(">I", buf.read(4))[0]
        elif msg.flag in (Flags.PositiveSeq, Flags.NegativeSeq) and msg.type in (
            MsgType.FullClientRequest, MsgType.FullServerResponse,
            MsgType.FrontEndResultServer, MsgType.AudioOnlyClient, MsgType.AudioOnlyServer,
        ):
            msg.sequence = struct.unpack(">i", buf.read(4))[0]

        size_bytes = buf.read(4)
        if size_bytes:
            size = struct.unpack(">I", size_bytes)[0]
            if size:
                msg.payload = buf.read(size)
        return msg


# ── headers / send / recv ────────────────────────────────────────────────────
def _build_headers(resource_id: str, app_id: str, access_token: str) -> dict:
    return {
        "X-Api-App-Id":     app_id,
        "X-Api-App-Key":    app_id,        # 某些端点用 App-Key，两个都给以兼容
        "X-Api-Access-Key": access_token,
        "X-Api-Resource-Id": resource_id,
        "X-Api-Request-Id": str(uuid.uuid4()),
        "X-Api-Connect-Id": str(uuid.uuid4()),
    }


async def _recv(ws, timeout: float) -> Message:
    try:
        data = await asyncio.wait_for(ws.recv(), timeout=timeout)
    except asyncio.TimeoutError as e:
        raise VolcTimeoutError(f"等待服务端帧超时 ({timeout}s)") from e
    except WebSocketException as e:
        raise VolcVoiceError(f"等待服务端帧时连接关闭: {e}") from e
    if isinstance(data, str):
        raise VolcVoiceError(f"意外的文本帧: {data[:200]!r}")
    if not data:
        raise VolcVoiceError("空的服务端帧")
    try:
        msg = Message.unmarshal(data)
    except (ValueError, struct.error) as e:
        raise VolcVoiceError(f"服务端帧格式错误: {e}") from e
    if msg.type == MsgType.Error:
        body = msg.payload.decode("utf-8", errors="replace") if msg.payload else ""
        _raise_for_code(msg.error_code, f"服务端错误: {body}", payload=body)
    if msg.event in (Event.ConnectionFailed, Event.SessionFailed):
        body = msg.json_payload() or msg.payload
        raise VolcServerError(f"{msg.event.name}: {body!r}", code=msg.error_code or None, payload=body)
    return msg


async def _send(ws, msg: Message) -> None:
    await ws.send(msg.marshal())


async def _connect(url: str, headers: dict, timeout: float):
    try:
        return await asyncio.wait_for(
            websockets.connect(
                url,
                additional_headers=headers,
                max_size=16 * 1024 * 1024,
                open_timeout=timeout,
                ping_interval=20,
                ping_timeout=20,
                proxy=None,   # 关闭 websockets>=14 的系统代理自动探测
            ),
            timeout=timeout,
        )
    except asyncio.TimeoutError as e:
        raise VolcTimeoutError(f"连接超时 ({timeout}s): {url}") from e
    except WebSocketException as e:
        t = str(e)
        if "401" in t or "403" in t or "auth" in t.lower():
            raise VolcAuthError(f"握手被拒绝: {e}") from e
        raise VolcVoiceError(f"握手失败: {e}") from e
    except OSError as e:
        raise VolcVoiceError(f"连接失败: {e}") from e


# ── TTS 双向流式 ─────────────────────────────────────────────────────────────
async def tts_stream(
    text: str,
    *,
    speaker: str = "zh_female_vv_uranus_bigtts",
    audio_format: str = "pcm",      # pcm | mp3 | ogg_opus
    sample_rate: int = 24000,
    speech_rate: int = 0,           # 整型百分比增量 [-50, 100]
    emotion: Optional[str] = None,  # happy|sad|angry|surprised|... (seed-tts-2.0 emo)
    emotion_scale: Optional[int] = None,  # 1..5
    app_id: str = DEFAULT_APP_ID,
    access_token: str = DEFAULT_ACCESS_TOKEN,
    resource_id: str = TTS_RESOURCE_ID,
    timeout: float = 30.0,
) -> AsyncIterator[bytes]:
    """流式合成，产出原始音频字节块（默认 PCM16 单声道 24k）。

    事件流：StartConnection→ConnectionStarted→StartSession→SessionStarted
            →TaskRequest→(AudioOnlyServer...)→SessionFinished/TTSEnded。
    """
    headers = _build_headers(resource_id, app_id, access_token)

    audio_params = {
        "format": audio_format,
        "sample_rate": sample_rate,
        "speech_rate": max(-50, min(100, int(speech_rate))),
    }
    if emotion:
        audio_params["emotion"] = emotion
    if emotion_scale is not None:
        audio_params["emotion_scale"] = int(emotion_scale)

    req_params = {"speaker": speaker, "audio_params": audio_params}
    base_payload = {
        "user": {"uid": headers["X-Api-Request-Id"]},
        "namespace": "BidirectionalTTS",
        "req_params": req_params,
    }
    session_id = str(uuid.uuid4())

    ws = await _connect(TTS_URL, headers, timeout)
    try:
        # 1. StartConnection
        await _send(ws, Message(type=MsgType.FullClientRequest, flag=Flags.WithEvent,
                                event=Event.StartConnection, payload=b"{}"))
        msg = await _recv(ws, timeout)
        if msg.event != Event.ConnectionStarted:
            raise VolcServerError(f"期望 ConnectionStarted，收到 {msg.event.name}",
                                  payload=msg.json_payload() or msg.payload)

        # 2. StartSession
        start_payload = dict(base_payload, event=int(Event.StartSession))
        await _send(ws, Message(type=MsgType.FullClientRequest, flag=Flags.WithEvent,
                                event=Event.StartSession, session_id=session_id,
                                payload=json.dumps(start_payload, ensure_ascii=False).encode("utf-8")))
        msg = await _recv(ws, timeout)
        if msg.event != Event.SessionStarted:
            raise VolcServerError(f"期望 SessionStarted，收到 {msg.event.name}",
                                  payload=msg.json_payload() or msg.payload)

        # 3. TaskRequest（携带文本）
        task_payload = dict(base_payload, event=int(Event.TaskRequest))
        task_payload["req_params"] = dict(req_params, text=text)
        await _send(ws, Message(type=MsgType.FullClientRequest, flag=Flags.WithEvent,
                                event=Event.TaskRequest, session_id=session_id,
                                payload=json.dumps(task_payload, ensure_ascii=False).encode("utf-8")))

        # 3b. 立刻 FinishSession，告知不再有文本，服务端可收尾
        await _send(ws, Message(type=MsgType.FullClientRequest, flag=Flags.WithEvent,
                                event=Event.FinishSession, session_id=session_id, payload=b"{}"))

        # 4. 收音频，直到 TTSEnded / SessionFinished
        session_done = False
        while True:
            msg = await _recv(ws, timeout)
            if msg.type == MsgType.AudioOnlyServer:
                if msg.payload:
                    yield msg.payload
            elif msg.type == MsgType.FullServerResponse:
                if msg.event in (Event.TTSEnded, Event.SessionFinished):
                    session_done = (msg.event == Event.SessionFinished)
                    break
                # TTSResponse/SentenceStart/SentenceEnd 为进度帧，忽略

        # 5. 收尾
        if not session_done:
            await _send(ws, Message(type=MsgType.FullClientRequest, flag=Flags.WithEvent,
                                    event=Event.FinishSession, session_id=session_id, payload=b"{}"))
            with contextlib.suppress(VolcTimeoutError):
                await _recv(ws, timeout)
        await _send(ws, Message(type=MsgType.FullClientRequest, flag=Flags.WithEvent,
                                event=Event.FinishConnection, payload=b"{}"))
        with contextlib.suppress(VolcTimeoutError):
            await _recv(ws, timeout)
    finally:
        with contextlib.suppress(WebSocketException):
            await ws.close()


class TTSSession:
    """复用一条 WS 连接连续合成多句（每句一个 session）。
    相比每句新建连接，省掉每句的 TCP/TLS + StartConnection（~0.4s），句间几乎无空隙。
    用法：await open() →（多次）async for a in say(text, ...) → await close()。"""

    def __init__(self, *, speaker="zh_female_vv_uranus_bigtts", sample_rate=24000,
                 app_id=DEFAULT_APP_ID, access_token=DEFAULT_ACCESS_TOKEN,
                 resource_id=TTS_RESOURCE_ID, timeout=30.0):
        self.speaker = speaker
        self.sample_rate = sample_rate
        self.headers = _build_headers(resource_id, app_id, access_token)
        self.timeout = timeout
        self.ws = None
        self._open_lock = asyncio.Lock()

    async def open(self):
        async with self._open_lock:
            if self.ws is not None:
                return
            ws = await _connect(TTS_URL, self.headers, self.timeout)
            await _send(ws, Message(type=MsgType.FullClientRequest, flag=Flags.WithEvent,
                                    event=Event.StartConnection, payload=b"{}"))
            msg = await _recv(ws, self.timeout)
            if msg.event != Event.ConnectionStarted:
                with contextlib.suppress(WebSocketException):
                    await ws.close()
                raise VolcServerError(f"期望 ConnectionStarted，收到 {msg.event.name}",
                                      payload=msg.json_payload() or msg.payload)
            self.ws = ws

    async def say(self, text, *, emotion=None, emotion_scale=None, speech_rate=0):
        """在已建连接上起一个新 session 合成 text，逐块产出 PCM。"""
        if self.ws is None:
            await self.open()
        session_id = str(uuid.uuid4())
        audio_params = {"format": "pcm", "sample_rate": self.sample_rate,
                        "speech_rate": max(-50, min(100, int(speech_rate)))}
        if emotion:
            audio_params["emotion"] = emotion
        if emotion_scale is not None:
            audio_params["emotion_scale"] = int(emotion_scale)
        req_params = {"speaker": self.speaker, "audio_params": audio_params}
        base = {"user": {"uid": self.headers["X-Api-Request-Id"]},
                "namespace": "BidirectionalTTS", "req_params": req_params}

        start_payload = dict(base, event=int(Event.StartSession))
        await _send(self.ws, Message(type=MsgType.FullClientRequest, flag=Flags.WithEvent,
                                     event=Event.StartSession, session_id=session_id,
                                     payload=json.dumps(start_payload, ensure_ascii=False).encode("utf-8")))
        msg = await _recv(self.ws, self.timeout)
        if msg.event != Event.SessionStarted:
            raise VolcServerError(f"期望 SessionStarted，收到 {msg.event.name}",
                                  payload=msg.json_payload() or msg.payload)

        task_payload = dict(base, event=int(Event.TaskRequest))
        task_payload["req_params"] = dict(req_params, text=text)
        await _send(self.ws, Message(type=MsgType.FullClientRequest, flag=Flags.WithEvent,
                                     event=Event.TaskRequest, session_id=session_id,
                                     payload=json.dumps(task_payload, ensure_ascii=False).encode("utf-8")))
        await _send(self.ws, Message(type=MsgType.FullClientRequest, flag=Flags.WithEvent,
                                     event=Event.FinishSession, session_id=session_id, payload=b"{}"))
        while True:
            msg = await _recv(self.ws, self.timeout)
            if msg.type == MsgType.AudioOnlyServer:
                if msg.payload:
                    yield msg.payload
            elif msg.type == MsgType.FullServerResponse:
                if msg.event in (Event.TTSEnded, Event.SessionFinished):
                    break

    async def close(self):
        if self.ws is None:
            return
        ws, self.ws = self.ws, None
        with contextlib.suppress(Exception):
            await _send(ws, Message(type=MsgType.FullClientRequest, flag=Flags.WithEvent,
                                    event=Event.FinishConnection, payload=b"{}"))
            with contextlib.suppress(VolcTimeoutError):
                await _recv(ws, self.timeout)
        with contextlib.suppress(WebSocketException):
            await ws.close()


# ── ASR 大模型流式 ───────────────────────────────────────────────────────────
async def asr_stream(
    audio_aiter: AsyncIterator[bytes],
    *,
    sample_rate: int = 16000,
    bits: int = 16,
    channel: int = 1,
    audio_format: str = "pcm",
    codec: str = "raw",
    model_name: str = "bigmodel",
    enable_itn: bool = True,
    enable_punc: bool = True,
    enable_ddc: bool = True,
    show_utterances: bool = True,
    app_id: str = DEFAULT_APP_ID,
    access_token: str = DEFAULT_ACCESS_TOKEN,
    resource_id: str = ASR_RESOURCE_ID,
    timeout: float = 30.0,
) -> AsyncIterator[dict]:
    """流式识别。逐帧产出 {text, is_final, utterances}。

    ``audio_aiter`` 持续产出 PCM16 字节块（约 200ms/块）。其耗尽 = 用户说完，
    此时发送负序列号末包让服务端定稿。sender / receiver 两协程并发，
    消费方可在服务端首次返回时就拿到部分结果。
    """
    headers = _build_headers(resource_id, app_id, access_token)

    init_payload = {
        "user": {"uid": headers["X-Api-Request-Id"]},
        "audio": {"format": audio_format, "codec": codec,
                  "rate": sample_rate, "bits": bits, "channel": channel},
        "request": {
            "model_name": model_name,
            "enable_itn": enable_itn,
            "enable_punc": enable_punc,
            "enable_ddc": enable_ddc,
            "show_utterances": show_utterances,
            "enable_nonstream": False,
        },
    }

    ws = await _connect(ASR_URL, headers, timeout)
    recv_queue: "asyncio.Queue" = asyncio.Queue()

    async def receiver():
        try:
            while True:
                msg = await _recv(ws, timeout)
                payload = msg.json_payload()
                if msg.type == MsgType.FullServerResponse and isinstance(payload, dict):
                    result = payload.get("result")
                    last = bool(msg.flag & Flags.LastNoSeq) or msg.flag == Flags.NegativeSeq
                    if isinstance(result, dict):
                        out = {
                            "text": result.get("text", ""),
                            "is_final": bool(result.get("is_final", False)) or last,
                            "utterances": result.get("utterances", []) or [],
                        }
                        await recv_queue.put(out)
                        if out["is_final"]:
                            break
                    elif last:
                        break
        except Exception as e:
            await recv_queue.put(e if isinstance(e, Exception) else VolcVoiceError(str(e)))
        finally:
            await recv_queue.put(None)

    async def sender():
        seq = 1
        try:
            await _send(ws, Message(type=MsgType.FullClientRequest, flag=Flags.PositiveSeq,
                                    serialization=Serialization.JSON, compression=Compression.Gzip,
                                    sequence=seq,
                                    payload=gzip.compress(json.dumps(init_payload, ensure_ascii=False).encode("utf-8"))))
            seq += 1
            prev = None
            async for chunk in audio_aiter:
                if not chunk:
                    continue
                if prev is not None:
                    await _send(ws, Message(type=MsgType.AudioOnlyClient, flag=Flags.PositiveSeq,
                                            serialization=Serialization.JSON, compression=Compression.Gzip,
                                            sequence=seq, payload=gzip.compress(prev)))
                    seq += 1
                prev = chunk
            final = prev if prev is not None else b""
            await _send(ws, Message(type=MsgType.AudioOnlyClient, flag=Flags.NegativeSeq,
                                    serialization=Serialization.JSON, compression=Compression.Gzip,
                                    sequence=-seq, payload=gzip.compress(final)))
        except Exception as e:
            await recv_queue.put(e if isinstance(e, Exception) else VolcVoiceError(str(e)))

    recv_task = asyncio.create_task(receiver(), name="volc-asr-recv")
    send_task = asyncio.create_task(sender(), name="volc-asr-send")
    try:
        while True:
            item = await recv_queue.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            yield item
            if item.get("is_final"):
                break
    finally:
        for t in (send_task, recv_task):
            if not t.done():
                t.cancel()
        for t in (send_task, recv_task):
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await t
        with contextlib.suppress(WebSocketException):
            await ws.close()
