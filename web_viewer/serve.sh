#!/usr/bin/env bash
# 在项目根目录起 HTTP 服务（web_viewer 在 emotion_motion_pipeline/ 下，上溯两级到根）
cd "$(dirname "$0")/../.." || exit 1
echo "Serving project root at http://localhost:8000"
echo "打开： http://localhost:8000/emotion_motion_pipeline/web_viewer/"
python3 -m http.server 8000
