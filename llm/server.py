#!/usr/bin/env python3
"""
AI 对话系统 — HTTP 层（Flask + SSE 文字流）
  ① 三层 system prompt（稳定层 / 用户档案层 / 易变层）— 缓存友好
  ② 双轨记忆压缩（对话摘要 + 用户档案分开维护）
  ③ 记忆链（会话分裂归档，支持"你还记得…"的历史感）
  ④ 情绪标签（AI 输出 <emotion> 标签，服务端剥离后返回前端）
  ⑤ 语音（ASR/TTS 双流式）— 见 voice_server.py（asyncio，端口 5001）

对话核心（人设/prompt/记忆/压缩/情绪）抽到 chat_core.py，与语音服务共享，
因此语音轮次和文字轮次走同一段记忆。
"""

import os, uuid, json, threading, asyncio
from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context

import chat_core as core
from chat_core import (
    client_for, llm_extra, DEFAULT_MODEL, AVAILABLE_MODELS, MODEL_IDS,
    personas, load_personas, assemble_system,
    sessions, get_session, maybe_compress,
)
from voice_server import start_voice_server

app = Flask(__name__, static_folder=".")


# ── 路由 ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/showcase")
@app.route("/showcase.html")
def showcase():
    return send_from_directory(".", "showcase.html")


@app.route("/favicon.ico")
def favicon():
    return ("", 204)   # 没有图标，回 204 避免控制台 404 红字


# ── 机器人查看器（同源托管，便于情绪→动作 BroadcastChannel 直连）────────────────
_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))      # 项目根
_VIEWER = os.path.join(_ROOT, "emotion_motion_pipeline", "web_viewer")     # 复刻的查看器

@app.route("/robot/")
def robot_index():
    return send_from_directory(_VIEWER, "index.html")

@app.route("/robot/<path:p>")
def robot_static(p):
    return send_from_directory(_VIEWER, p)

@app.route("/urdf/<path:p>")
def urdf_static(p):
    return send_from_directory(os.path.join(_ROOT, "urdf"), p)

@app.route("/meshes/<path:p>")
def meshes_static(p):
    return send_from_directory(os.path.join(_ROOT, "meshes"), p)


@app.route("/models")
def list_models():
    return jsonify(AVAILABLE_MODELS)


@app.route("/personas")
def list_personas():
    return jsonify([{
        "id":              p["id"],
        "display_name":    p["display_name"],
        "alias":           p.get("alias", ""),
        "avatar":          p.get("avatar", "👤"),
        "tagline":         p.get("tagline", ""),
        "theme_start":     p.get("theme_start", "#888"),
        "theme_end":       p.get("theme_end", "#aaa"),
        "preferred_model": p.get("preferred_model", ""),
    } for p in personas.values()])


@app.route("/profile")
def get_profile():
    """返回指定 session 的用户档案（用于前端展示）。"""
    sid        = request.args.get("session_id", "")
    persona_id = request.args.get("persona_id", "")
    key = f"{sid}_{persona_id}"
    sess = sessions.get(key)
    if not sess:
        return jsonify({"profile": "", "summary": "", "chain": []})
    return jsonify({
        "profile": sess["user_profile"],
        "summary": sess["memory_summary"],
        "chain":   sess["memory_chain"],
    })


@app.route("/chat", methods=["POST"])
def chat():
    data       = request.get_json()
    sid        = data.get("session_id", "")
    persona_id = data.get("persona_id", "")
    message    = data.get("message", "").strip()
    req_model  = data.get("model", "")
    if req_model in MODEL_IDS:
        model = req_model
    else:
        # 优先用人设绑定的模型，再回退全局默认
        model = personas.get(persona_id, {}).get("preferred_model", DEFAULT_MODEL)
        if model not in MODEL_IDS:
            model = DEFAULT_MODEL

    if not message:
        return jsonify({"error": "empty message"}), 400
    if persona_id not in personas:
        persona_id = next(iter(personas))
    persona = personas[persona_id]

    if not sid:
        sid = str(uuid.uuid4())

    sess = get_session(sid, persona_id)
    sess["messages"].append({"role": "user", "content": message})
    sess["turn_count"] += 1

    # ── 触发压缩（文字/语音共用 chat_core.maybe_compress）──
    memory_updated, profile_updated = maybe_compress(sess, persona, model)

    # ── 组装三层 Prompt ──
    system_content = assemble_system(
        persona_id,
        sess["user_profile"],
        sess["memory_summary"],
        sess["memory_chain"],
    )
    full_msgs = [{"role": "system", "content": system_content}] + sess["messages"]

    def generate():
        buf         = ""      # 缓冲区：用于检测 <emotion> 标签
        full_reply  = ""      # 完整回复，流结束后写入 session
        emotion     = "平静"
        tag_done    = False   # 是否已处理完情绪标签
        cleaner     = core.StageStreamer()   # 流式剥离括号神态 + 据此细化情绪

        def emit(text):
            """流式去括号 + 据括号细化情绪 + 清残留标签碎片，返回要发的干净文本。"""
            nonlocal emotion
            show, se = cleaner.feed(text)
            if se and emotion == "平静":
                emotion = se
            return core.TAG_JUNK_RE.sub("", show)

        def sse(obj):
            return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"

        try:
            stream = client_for(model).chat.completions.create(
                model=model,
                messages=full_msgs,
                temperature=0.85,
                max_tokens=1024,
                stream=True,
                **llm_extra(model),
            )

            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content or ""
                if not delta:
                    continue

                if not tag_done:
                    # 攒到「情绪声明已完整」再一次性健壮解析（别在半截标签处下刀）
                    buf += delta
                    if core.lead_ready(buf):
                        tag_done = True
                        emotion, rest = core.parse_lead_emotion(buf)
                        show = emit(rest)
                        if show:
                            full_reply += show
                            yield sse({"type": "chunk", "text": show})
                    # else: 继续缓冲标签
                else:
                    show = emit(delta)
                    if show:
                        full_reply += show
                        yield sse({"type": "chunk", "text": show})

            # 流结束但标签还没处理（极短回复）
            if not tag_done and buf:
                emotion, clean = core.parse_lead_emotion(buf)
                show = emit(clean)
                full_reply = show
                if show:
                    yield sse({"type": "chunk", "text": show})

        except Exception as e:
            sess["messages"].pop()
            sess["turn_count"] -= 1
            yield sse({"type": "error", "message": str(e)})
            yield "data: [DONE]\n\n"
            return

        # 写入 session
        sess["messages"].append({"role": "assistant", "content": full_reply})

        yield sse({
            "type":           "meta",
            "emotion":        emotion,
            "session_id":     sid,
            "turn_count":     sess["turn_count"],
            "memory_updated": memory_updated,
            "profile_updated":profile_updated,
            "has_memory":     bool(sess["memory_summary"]),
            "chain_length":   len(sess["memory_chain"]),
            "model":          model,
        })
        yield "data: [DONE]\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _run_flask():
    # Werkzeug 开发服务器跑在后台线程；语音服务（asyncio）占主线程
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False, threaded=True)


def _prewarm():
    """后台预热：LLM 连接池 + 火山 TTS 主机的 DNS/TLS，让首条消息少握手一次。"""
    import contextlib, volc_voice as vv
    core.prewarm_llm()
    async def _warm_tts():
        with contextlib.suppress(Exception):
            async for _ in vv.tts_stream("你好", sample_rate=24000):
                break   # 拿到首帧即可，目的是把到 openspeech 的连接打通
    with contextlib.suppress(Exception):
        asyncio.run(_warm_tts())
    print("🔥 预热完成（LLM 连接池 + TTS 主机）")


if __name__ == "__main__":
    load_personas()
    print(f"💕 AI 对话系统 — {len(personas)} 个人设")
    print(f"   记忆压缩：每 {core.COMPRESS_EVERY} 轮 | 记忆链：保留全部 | 情绪标签：已启用")
    print("🌐 HTTP   http://localhost:5000")
    print("🤖 机器人 http://localhost:5000/robot/  (情绪→动作，同源直连)")
    print("🎙️  语音   ws://localhost:5001/ws/voice")

    threading.Thread(target=_run_flask, daemon=True).start()
    threading.Thread(target=_prewarm, daemon=True).start()
    try:
        asyncio.run(start_voice_server(port=5001))
    except KeyboardInterrupt:
        print("\n再见～")
