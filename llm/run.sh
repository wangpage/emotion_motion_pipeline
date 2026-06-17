#!/usr/bin/env bash
# 启动 AI 一直在 服务（HTTP :5000 + 语音 :5001）。
# 固定用系统 Python 3.9：依赖装在这里，miniconda 3.13 缺 markupsafe 会起不来。
set -euo pipefail

cd "$(dirname "$0")"

PYBIN="/Library/Developer/CommandLineTools/usr/bin/python3"
LOG="/tmp/aiyizhizai_server.log"

# 停掉已在跑的实例，避免端口占用
pkill -f "[s]erver.py" 2>/dev/null && { echo "已停止旧实例"; sleep 1; } || true

case "${1:-}" in
  -d|--daemon)
    nohup "$PYBIN" server.py > "$LOG" 2>&1 &
    echo "后台启动 pid $!  ·  日志：$LOG"
    ;;
  *)
    exec "$PYBIN" server.py
    ;;
esac
