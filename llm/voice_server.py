#!/usr/bin/env python3
"""
语音服务（asyncio WebSocket，端口 5001，path /ws/voice）。

每条连接编排「双流式」语音对话：
  浏览器麦克风 PCM ─▶ ASR 流 ─▶ 定稿文本 ─▶ LLM 流 ─▶ 句级切分 ─▶ TTS 流 ─▶ 浏览器播放

句级流水线是「双流式」的核心：第 k 句在 TTS 合成时，LLM 仍在生成第 k+1 句。
与文字聊天共享 chat_core 里的 sessions/personas，因此走同一段记忆。

浏览器 → 服务端：
  {"type":"start", session_id, persona_id, model, mode:"ptt"|"hands_free"}
  binary 帧 = PCM16 16k 单声道音频块
  {"type":"stop"}      PTT 松手，当前句定稿
  {"type":"barge_in"}  免提打断，停掉正在播放的 TTS

服务端 → 浏览器：
  {"type":"asr_partial"|"asr_final", text}
  {"type":"llm_chunk", text}
  {"type":"tts_start", sample_rate} → 若干 binary 音频帧 → {"type":"tts_end"}
  {"type":"meta", emotion, turn_count, memory_updated, profile_updated, has_memory, chain_length, model}
  {"type":"stop_audio"}  令前端清空播放队列（打断）
  {"type":"error", message}
"""

import asyncio
import contextlib
import json
import logging
import re

import websockets

import chat_core as core
import volc_voice as vv

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("voice_server")

TTS_SAMPLE_RATE = 24000          # seed-tts 输出 PCM16 24k
SENTENCE_ENDERS = "。！？!?…\n"    # 句末标点
SOFT_BREAKERS   = "，、,;；"       # 长句软切分点
SOFT_LIMIT      = 60             # 缓冲超过此长度时在软切分点提前出句
FIRST_FLUSH_MIN = 8             # 一轮里的「第一句」：攒够这么多字就在软切分点抢着出声，尽早开口
MIN_SEG         = 22            # 第一句之后，每段至少攒这么多字再出（合并短句，播放更连贯）
# 情绪转折处垫的静音（换气感）：~200ms @ 24k PCM16 单声道 = 24000*0.2*2 字节
EMO_SHIFT_PAUSE = b"\x00" * int(TTS_SAMPLE_RATE * 0.20) * 2

_STOP = object()   # PTT 松手：当前句音频结束
_EOF  = object()   # 连接关闭：音频流彻底结束


def pick_model(req_model, persona_id):
    if req_model in core.MODEL_IDS:
        return req_model
    m = core.personas.get(persona_id, {}).get("preferred_model", core.DEFAULT_MODEL)
    return m if m in core.MODEL_IDS else core.DEFAULT_MODEL


class VoiceConn:
    """单条浏览器连接的会话编排。"""

    def __init__(self, ws):
        self.ws = ws
        self.send_lock = asyncio.Lock()      # websockets 不允许并发 send
        self.audio_q: asyncio.Queue = asyncio.Queue()
        self.turn_q: asyncio.Queue = asyncio.Queue()
        self.mode = "ptt"
        self.sid = ""
        self.persona_id = ""
        self.persona = {}
        self.model = core.DEFAULT_MODEL
        self.sess = None
        self.tts_task: asyncio.Task = None   # 当前 TTS 消费协程（用于打断）
        self.closing = False

    # ── 出站（统一加锁）──
    async def send_json(self, obj):
        async with self.send_lock:
            await self.ws.send(json.dumps(obj, ensure_ascii=False))

    async def send_bytes(self, data: bytes):
        async with self.send_lock:
            await self.ws.send(data)

    # ── 主循环：收浏览器帧 ──
    async def reader(self):
        async for frame in self.ws:
            if isinstance(frame, (bytes, bytearray)):
                await self.audio_q.put(bytes(frame))
                continue
            try:
                msg = json.loads(frame)
            except json.JSONDecodeError:
                continue
            t = msg.get("type")
            if t == "stop":
                await self.audio_q.put(_STOP)
            elif t == "barge_in":
                await self._cancel_tts()
                await self.send_json({"type": "stop_audio"})
            elif t == "end":
                break
        self.closing = True
        await self.audio_q.put(_EOF)
        await self.turn_q.put(_EOF)

    async def _cancel_tts(self):
        if self.tts_task and not self.tts_task.done():
            self.tts_task.cancel()
            try:
                await self.tts_task
            except (asyncio.CancelledError, Exception):
                pass

    # ── 当前句音频迭代器（PTT：到 _STOP 结束；任何模式到 _EOF 都结束）──
    async def _utterance_audio(self, first: bytes):
        # 先吐出已收到的首块（开连接的触发块），再继续读后续音频。
        yield first
        while True:
            item = await self.audio_q.get()
            if item is _STOP or item is _EOF:
                if item is _EOF:
                    await self.audio_q.put(_EOF)  # 让后续也能读到
                return
            yield item

    # ── 整段音频迭代器（免提：仅到 _EOF 结束，忽略 _STOP）──
    async def _continuous_audio(self):
        while True:
            item = await self.audio_q.get()
            if item is _EOF:
                return
            if item is _STOP:
                continue
            yield item

    # ── ASR 循环 ──
    async def asr_loop(self):
        try:
            if self.mode == "hands_free":
                await self._asr_continuous()
            else:
                await self._asr_ptt()
        except vv.VolcVoiceError as e:
            await self.send_json({"type": "error", "message": f"ASR: {e}"})
        finally:
            await self.turn_q.put(_EOF)

    async def _asr_ptt(self):
        """按住说话：一次按压 = 一个 asr_stream，定稿即一轮。

        关键：必须等真正的音频帧到了再开 ASR 连接。火山 ASR 在连接建立后
        若 8s 收不到音频包就报 [45000081] 超时——用户松手到下次开口之间的静默
        会触发它，进而拖垮整条语音管道。所以这里先阻塞等首块音频。
        """
        while not self.closing:
            first = await self.audio_q.get()
            if first is _EOF:
                await self.audio_q.put(_EOF)  # 让其它读者也能收到
                break
            if first is _STOP:
                continue  # 误触/空按一下，没有音频，忽略

            final_text = ""
            async for r in vv.asr_stream(self._utterance_audio(first)):
                txt = r.get("text", "")
                if r.get("is_final"):
                    final_text = txt or final_text
                elif txt:
                    await self.send_json({"type": "asr_partial", "text": txt})
            final_text = final_text.strip()
            if final_text:
                log.info("asr_final: %r", final_text[:60])
                await self.send_json({"type": "asr_final", "text": final_text})
                await self.turn_q.put(final_text)
            else:
                log.info("asr_final: <空>（没识别到语音）")

    async def _asr_continuous(self):
        """免提连续：单条 ASR 连接，按 utterance.definite 切轮。"""
        seen_definite = 0
        async for r in vv.asr_stream(self._continuous_audio()):
            utts = r.get("utterances", []) or []
            definite = [u for u in utts if u.get("definite")]
            if len(definite) > seen_definite:
                for u in definite[seen_definite:]:
                    txt = (u.get("text") or "").strip()
                    if txt:
                        await self.send_json({"type": "asr_final", "text": txt})
                        await self.turn_q.put(txt)
                seen_definite = len(definite)
            # 未定稿的尾巴作为实时预览
            tail = next((u.get("text", "") for u in utts if not u.get("definite")), r.get("text", ""))
            if tail:
                await self.send_json({"type": "asr_partial", "text": tail})

    # ── 轮次 worker：串行处理每一句定稿用户话 ──
    async def turn_worker(self):
        while True:
            item = await self.turn_q.get()
            if item is _EOF:
                break
            try:
                await self.handle_turn(item)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.exception("turn failed")
                await self.send_json({"type": "error", "message": str(e)})

    # ── 一轮：LLM 流 → 句级切分 → TTS 流（双流式核心）──
    async def handle_turn(self, user_text: str):
        sess = self.sess
        sess["messages"].append({"role": "user", "content": user_text})
        sess["turn_count"] += 1

        # 压缩用同步 LLM，放线程里跑，别阻塞事件循环
        memory_updated, profile_updated = await asyncio.to_thread(
            core.maybe_compress, sess, self.persona, self.model
        )

        system_content = core.assemble_system(
            self.persona_id, sess["user_profile"], sess["memory_summary"], sess["memory_chain"]
        )
        full_msgs = [{"role": "system", "content": system_content}] + sess["messages"]

        # 句队列 + TTS 消费协程（边生成边合成边播）
        sentence_q: asyncio.Queue = asyncio.Queue()
        emotion_box = {"emotion": "平静"}    # 徽章情绪（整轮一个，给前端头像）
        cur = {"emotion": "平静"}            # 句级滚动情绪：每遇括号神态就更新，逐句驱动语气

        # 整轮复用一条 TTS 连接（消除句间重连空隙），并提前预热连接（与 LLM 生成并行）
        voice = self.persona.get("tts_voice", core.DEFAULT_VOICE)
        tts_sess = vv.TTSSession(speaker=voice, sample_rate=TTS_SAMPLE_RATE)
        warm = asyncio.create_task(tts_sess.open())   # 预热：连接在 LLM 出字时就建好

        async def tts_consumer():
            prev_emo = None
            try:
                with contextlib.suppress(Exception):
                    await warm                        # 等预热完（已建好则立即返回）
                while True:
                    item = await sentence_q.get()
                    if item is _EOF:
                        break
                    s, sent_emo = item                # 每段带自己的情绪
                    s = core.strip_stage(s)          # 剥掉舞台动作/旁白，别念出来
                    if not s:
                        continue                      # 整句只剩动作 → 跳过，不合成
                    tts_emo = core.TTS_EMOTION.get(sent_emo, "neutral")
                    rate = core.speech_rate_for(sent_emo, self.persona)   # 人设基础 + 情绪增量
                    log.info("tts 段: 情绪=%s 语速=%+d 文本=%r", sent_emo, rate, s[:20])
                    await self.send_json({"type": "tts_start", "sample_rate": TTS_SAMPLE_RATE})
                    # 情绪发生转折时，先垫一小段静音当「换气/停顿」，让转折更自然不突兀
                    if prev_emo is not None and sent_emo != prev_emo:
                        await self.send_bytes(EMO_SHIFT_PAUSE)
                    try:
                        async for audio in tts_sess.say(
                            s, emotion=tts_emo if tts_emo != "neutral" else None,
                            emotion_scale=5, speech_rate=rate,
                        ):
                            await self.send_bytes(audio)
                    except vv.VolcVoiceError as e:
                        await self.send_json({"type": "error", "message": f"TTS: {e}"})
                    await self.send_json({"type": "tts_end"})
                    prev_emo = sent_emo
            finally:
                await tts_sess.close()

        self.tts_task = asyncio.create_task(tts_consumer())

        full_reply = ""
        buf = ""
        tag_done = False
        head = ""   # 标签解析前的缓冲
        self._first_seg = True   # 本轮第一句尽早出声（降低首声延迟）
        cleaner = core.StageStreamer()   # 流式剥离括号神态 + 据此细化语气

        try:
            stream = await core.aclient_for(self.model).chat.completions.create(
                model=self.model, messages=full_msgs,
                temperature=0.85, max_tokens=1024, stream=True,
                **core.llm_extra(self.model),
            )
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content or ""
                if not delta:
                    continue

                # 攒到「情绪声明已完整」再一次性健壮解析（别在半截标签处下刀）
                if not tag_done:
                    head += delta
                    if not core.lead_ready(head):
                        continue   # 声明还没收完，继续缓冲
                    emo, delta = core.parse_lead_emotion(head)
                    emotion_box["emotion"] = emo
                    cur["emotion"] = emo          # 句级滚动情绪起点
                    tag_done = True
                    head = ""

                if not delta:
                    continue
                # 流式剥离括号神态 + 清残留标签碎片；括号神态逐句驱动语气（句级情绪切换）
                show, stage_emo = cleaner.feed(delta)
                show = core.TAG_JUNK_RE.sub("", show)
                if stage_emo:
                    cur["emotion"] = stage_emo                 # 每个神态都更新当前语气
                    if emotion_box["emotion"] == "平静":         # 徽章用首个明确情绪
                        emotion_box["emotion"] = stage_emo
                if not show:
                    continue
                full_reply += show
                await self.send_json({"type": "llm_chunk", "text": show})

                # 句级切分 → 推入 TTS 队列（带上当前这段的情绪）
                buf += show
                buf = await self._flush_sentences(buf, sentence_q, cur["emotion"])

            # 收尾：剩余缓冲 + 没解析到标签的极短回复
            if not tag_done and head:
                emo, clean = core.extract_emotion(head)
                emotion_box["emotion"] = emo
                cur["emotion"] = emo
                show, stage_emo = cleaner.feed(clean)
                if stage_emo:
                    cur["emotion"] = stage_emo
                    if emotion_box["emotion"] == "平静":
                        emotion_box["emotion"] = stage_emo
                if show:
                    full_reply += show
                    await self.send_json({"type": "llm_chunk", "text": show})
                    buf += show
            if buf.strip():
                await sentence_q.put((buf.strip(), cur["emotion"]))
                buf = ""
        except asyncio.CancelledError:
            raise
        except Exception as e:
            sess["messages"].pop()
            sess["turn_count"] -= 1
            await sentence_q.put(_EOF)
            await self._cancel_tts()
            await self.send_json({"type": "error", "message": str(e)})
            return

        # 等 TTS 把所有句子播完
        await sentence_q.put(_EOF)
        if self.tts_task:
            try:
                await self.tts_task
            except asyncio.CancelledError:
                pass  # 被打断

        sess["messages"].append({"role": "assistant", "content": full_reply})
        await self.send_json({
            "type": "meta",
            "emotion": emotion_box["emotion"],
            "session_id": self.sid,
            "turn_count": sess["turn_count"],
            "memory_updated": memory_updated,
            "profile_updated": profile_updated,
            "has_memory": bool(sess["memory_summary"]),
            "chain_length": len(sess["memory_chain"]),
            "model": self.model,
        })

    async def _flush_sentences(self, buf: str, sentence_q: asyncio.Queue, emotion: str) -> str:
        """把 buf 里完整的句子（连同当前情绪）推入队列，返回剩余未成句部分。
        第一段尽早出声（降首声）；之后的段攒够 MIN_SEG 再出（合并短句，减少句间空隙、更连贯）。"""
        while True:
            # 本轮第一句：攒够 FIRST_FLUSH_MIN 就在最早的软切分点/句末抢着出声，尽快开口
            if self._first_seg and len(buf) >= FIRST_FLUSH_MIN:
                cut = next((i for i, ch in enumerate(buf)
                            if ch in SENTENCE_ENDERS or ch in SOFT_BREAKERS), -1)
                if cut >= 0:
                    sentence = buf[:cut + 1].strip()
                    buf = buf[cut + 1:]
                    if sentence:
                        await sentence_q.put((sentence, emotion))
                        self._first_seg = False
                    continue
            # 之后：在句末切，但要求这一段已攒够 MIN_SEG（短句合并，播放更连贯）
            if not self._first_seg:
                idx = next((i for i, ch in enumerate(buf)
                            if ch in SENTENCE_ENDERS and i + 1 >= MIN_SEG), -1)
                if idx >= 0:
                    sentence = buf[:idx + 1].strip()
                    buf = buf[idx + 1:]
                    if sentence:
                        await sentence_q.put((sentence, emotion))
                    continue
            # 太长则在软切分点提前出句，降低长句延迟
            if len(buf) >= SOFT_LIMIT:
                cut = max((buf.rfind(c) for c in SOFT_BREAKERS), default=-1)
                if cut > 0:
                    sentence = buf[:cut + 1].strip()
                    buf = buf[cut + 1:]
                    if sentence:
                        await sentence_q.put((sentence, emotion))
                        self._first_seg = False
                    continue
            break
        return buf

    # ── 连接入口 ──
    async def run(self):
        # 第一帧必须是 start
        try:
            first = await asyncio.wait_for(self.ws.recv(), timeout=30)
        except (asyncio.TimeoutError, websockets.WebSocketException):
            return
        try:
            cfg = json.loads(first) if isinstance(first, str) else {}
        except json.JSONDecodeError:
            cfg = {}
        if cfg.get("type") != "start":
            await self.send_json({"type": "error", "message": "首帧必须是 start"})
            return

        self.sid = cfg.get("session_id") or ""
        self.persona_id = cfg.get("persona_id") or (next(iter(core.personas)) if core.personas else "")
        if self.persona_id not in core.personas:
            self.persona_id = next(iter(core.personas))
        self.persona = core.personas[self.persona_id]
        self.model = pick_model(cfg.get("model", ""), self.persona_id)
        self.mode = "hands_free" if cfg.get("mode") == "hands_free" else "ptt"
        if not self.sid:
            import uuid
            self.sid = str(uuid.uuid4())
        self.sess = core.get_session(self.sid, self.persona_id)

        await self.send_json({"type": "ready", "session_id": self.sid, "mode": self.mode})
        log.info("voice start sid=%s persona=%s model=%s mode=%s",
                 self.sid, self.persona_id, self.model, self.mode)

        reader_t = asyncio.create_task(self.reader())
        asr_t    = asyncio.create_task(self.asr_loop())
        worker_t = asyncio.create_task(self.turn_worker())
        try:
            await asyncio.gather(reader_t, asr_t, worker_t)
        finally:
            for t in (reader_t, asr_t, worker_t):
                if not t.done():
                    t.cancel()
            await self._cancel_tts()


async def _handler(ws):
    path = getattr(getattr(ws, "request", None), "path", "") or ""
    if path and not path.startswith("/ws/voice"):
        await ws.close(code=1008, reason="unknown path")
        return
    conn = VoiceConn(ws)
    try:
        await conn.run()
    except websockets.WebSocketException:
        pass
    except Exception:
        log.exception("voice connection crashed")


async def start_voice_server(host="0.0.0.0", port=5001):
    log.info("🎙️  语音服务启动 ws://%s:%d/ws/voice", host, port)
    async with websockets.serve(_handler, host, port, max_size=16 * 1024 * 1024):
        await asyncio.Future()   # run forever


if __name__ == "__main__":
    core.load_personas()
    asyncio.run(start_voice_server())
