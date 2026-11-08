#!/usr/bin/env python3
"""transcribe.py — 一次性耳朵（冷路径）：stdin 一行 JSON 进，stdout 一行 JSON 出。

每次调用都要重新加载模型（medium 约 5.6 秒）。常驻路径见 worker.py。
判定逻辑在 ears_core.py，两边共用——绝不复制一份判定，否则迟早漂移。

契约：
  in : {"audio": "...", "model": "...", "language": "zh", "device": "auto",
        "compute": "auto", "prompt": "", "maxSeconds": 600,
        "thresholds": {...}}
  out: {"ok": true, "verdict": "heard|unclear|no_speech|silent|empty", "text": "...",
        "partialText": "", "reason": "...", "language": "zh", "duration": 5.8,
        "elapsed": 7.4, "loadSeconds": 5.6, "decodeSeconds": 1.6, "device": "cuda",
        "compute": "int8_float16", "segments": [...]}
  err: {"ok": false, "error": "...", "hint": "..."}  → exit 1
"""
from __future__ import annotations

import json
import sys
import time

from ears_core import EarsFailure, configure_streams, run_once


def emit(obj: dict, code: int = 0) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()
    raise SystemExit(code)


def main() -> None:
    configure_streams()
    started = time.perf_counter()
    raw = sys.stdin.read()
    if not raw.strip():
        emit({"ok": False, "error": "没有收到请求", "hint": "stdin 需要一行 JSON，形如 {\"audio\": \"a.wav\"}"}, 1)
    try:
        req = json.loads(raw)
    except json.JSONDecodeError as exc:
        emit({"ok": False, "error": f"请求不是合法 JSON：{exc}",
              "hint": "检查引号与反斜杠转义，Windows 路径里 \\\\ 要写成双斜杠"}, 1)

    try:
        result = run_once(req)
    except EarsFailure as exc:
        emit({"ok": False, "error": exc.error, "hint": exc.hint}, 1)

    result["elapsed"] = round(time.perf_counter() - started, 2)
    emit(result)


if __name__ == "__main__":
    main()
