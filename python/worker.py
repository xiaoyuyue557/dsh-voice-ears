#!/usr/bin/env python3
"""worker.py — 常驻耳朵（热路径）：模型只加载一次，之后每条语音只解码。

为什么值得多一个文件：medium 的加载是 5.6 秒，解码只要 1–2 秒。
QQ 聊天的节奏下，每条语音等 7 秒和等 1.5 秒是两种产品。

协议（一行一条 JSON，stdout 只走协议内容，日志一律走 stderr）：
  → {"type":"init","model":"medium","device":"auto","compute":"auto"}
  ← {"type":"ready","device":"cuda","compute":"int8_float16","loadSeconds":5.6}
  ← {"type":"error","error":"...","hint":"..."}      # init 失败，进程随即退出

  → {"id":1,"audio":"...","language":"zh",...}
  ← {"id":1,"ok":true,"verdict":"heard",...}
  ← {"id":1,"ok":false,"error":"...","hint":"..."}   # 单条失败不杀进程

  → {"type":"ping"}
  ← {"type":"pong"}

设计取舍：
  - 单线程串行。whisper/ctranslate2 并发解码会抢显存，队列化比并行更稳。
  - 单条失败只回错误，不退出：模型已经花 5.6 秒装好了，不能因为一条坏音频丢掉它。
  - 空闲退出由 Node 侧决定（杀掉进程），worker 自己不猜策略。
"""
from __future__ import annotations

import json
import sys
import time

from ears_core import Ears, EarsFailure, configure_streams


def send(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def main() -> None:
    configure_streams()

    first = sys.stdin.readline()
    if not first.strip():
        send({"type": "error", "error": "worker 启动时没有收到 init", "hint": "内部错误，请重试"})
        return
    try:
        init = json.loads(first)
    except json.JSONDecodeError as exc:
        send({"type": "error", "error": f"init 不是合法 JSON：{exc}", "hint": "内部错误"})
        return

    ears = Ears(init.get("model") or "medium", init.get("device") or "auto", init.get("compute") or "auto")
    try:
        ears.ensure()
    except EarsFailure as exc:
        send({"type": "error", "error": exc.error, "hint": exc.hint})
        return

    send({
        "type": "ready",
        "device": ears.used[0],
        "compute": ears.used[1],
        "model": ears.model_ref,
        "loadSeconds": ears.load_seconds,
        "pid": __import__("os").getpid(),
    })

    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as exc:
            send({"ok": False, "error": f"请求不是合法 JSON：{exc}", "hint": ""})
            continue

        if req.get("type") == "ping":
            send({"type": "pong", "busy": False})
            continue
        if req.get("type") == "shutdown":
            send({"type": "bye"})
            return

        rid = req.get("id")
        started = time.perf_counter()
        try:
            result = ears.transcribe(req)
            result["ok"] = True
            result["id"] = rid
            result["elapsed"] = round(time.perf_counter() - started, 2)
            send(result)
        except EarsFailure as exc:
            send({"id": rid, "ok": False, "error": exc.error, "hint": exc.hint,
                  "elapsed": round(time.perf_counter() - started, 2)})
        except Exception as exc:  # noqa: BLE001
            # 模型可能已经进入坏状态（例如显存被别的进程抢走），丢掉它，下一条会重新加载。
            ears.drop()
            send({"id": rid, "ok": False, "error": f"转写失败：{exc}",
                  "hint": "已丢弃当前模型，下一条语音会重新加载；如果反复失败，看 README 的『报错对照』",
                  "elapsed": round(time.perf_counter() - started, 2)})


if __name__ == "__main__":
    try:
        main()
    except (BrokenPipeError, KeyboardInterrupt):
        pass
