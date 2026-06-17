#!/usr/bin/env python3
"""
配置加载：从 config.yaml 读取密钥与接入参数（启动时一次性加载）。
每项支持同名大写环境变量覆盖，便于线上注入而不动文件。
"""

import os
import yaml

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
_cfg = None


def _load() -> dict:
    global _cfg
    if _cfg is None:
        try:
            with open(_CONFIG_PATH, encoding="utf-8") as f:
                _cfg = yaml.safe_load(f) or {}
        except FileNotFoundError:
            raise SystemExit(
                f"[配置] 缺少 {_CONFIG_PATH}；请按 config.yaml 模板填入密钥后再启动。"
            )
    return _cfg


def get(path: str, default=None, env: str = None):
    """按点路径取配置，如 get('llm.api_key', env='LLM_API_KEY')。环境变量优先。"""
    if env and os.environ.get(env):
        return os.environ[env]
    cur = _load()
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur
