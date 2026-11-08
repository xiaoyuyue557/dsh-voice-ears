#!/usr/bin/env python3
"""ears_core.py — 耳朵的核心：加载模型、转写、判定。

被两个入口共用：
  - transcribe.py  一次性调用（每次起进程，模型每次重载 —— 冷路径）
  - worker.py      常驻进程（模型只加载一次 —— 热路径）

拆出来的原因很实际：0.1 的判定逻辑散在 CLI 脚本里，加常驻服务就得复制一份，
两份判定迟早会漂移——而这个插件最不能容忍的就是判定漂移。

术语：verdict 只有 heard 可信。其余（unclear / no_speech / silent / empty）都不该被当成"对方说了这句话"。
"""
from __future__ import annotations

import os
import site
import sys
import time
import wave
from pathlib import Path

DEFAULT_THRESHOLDS = {
    "no_speech_prob_max": 0.6,
    "avg_logprob_min": -1.0,
    "lang_prob_min": 0.7,
    # no_speech_prob 单独不能否决：带背景音的正常语音能到 0.67。
    # 只有"不像人说话"且"字音也含糊"同时成立才判没听清。
    "no_speech_logprob_floor": -0.6,
}

# 中文/英文语种判定放宽一档：短语音的 language_probability 天然偏低，
# 用 0.7 卡会把正常的普通话短句判成"没把握"。
LANG_RELAXED = {"zh": 0.5, "en": 0.5, "yue": 0.5}

# whisper 在静音/纯音乐/噪声上最爱编的几句。命中就是"不是在听人说话"。
# 简繁两套都要写：中文输出里繁体相当常见，只写简体等于漏一半。
# 只放**整段签名**，不放单词：单放「订阅」会把「帮我订阅一下」这种正常话误杀。
HALLUCINATION_PATTERNS = (
    ("请不吝点赞订阅转发打赏", "請不吝點贊訂閱轉發打賞"),
    ("支持明镜与点点栏目", "支持明鏡與點點欄目"),
    ("明镜与点点", "明鏡與點點"),
    ("词曲", "詞曲"),          # 静音上编出来的音乐字幕：「词曲 李宗盛」
    ("谢谢观看", "謝謝觀看"),
    ("感谢观看", "感謝觀看"),
    ("下期再见", "下期再見"),
    ("字幕由", "字幕由"),
    ("thanksforwatching",),
    ("likeandsubscribe",),
    ("amaraorg",),
    ("subscribe",),
)

# 中文短语音的默认提示词。实测（medium，2–8 秒 QQ 语音条）它同时改善三件事：
#   1. 输出从繁体变简体（whisper 对短普通话经常吐繁体）
#   2. 补上标点（否则一整句连在一起，agent 读起来费劲）
#   3. 专有名词更准（「小飞鱼」→「小肥鱼」）
# 只在明确指定 zh 时使用：language=auto 时给中文提示会干扰语种判断。
ZH_DEFAULT_PROMPT = "以下是普通话的句子。"

# 频谱平坦度：越高越像噪声。实测（16k 单声道，30ms 窗）：
#   真人语音 0.17 · 褐噪 0.44 · 粉噪 0.69 · 白噪 0.84 · 纯正弦 0.001
# 取 0.6 是保守值：只拦白噪/粉噪，褐噪会漏过去（由重复检测和黑名单兜）。
NOISE_FLATNESS_MAX = 0.6


class EarsFailure(Exception):
    """带人话提示的失败。error 面向"发生了什么"，hint 面向"该怎么办"。"""

    def __init__(self, error: str, hint: str = ""):
        super().__init__(error)
        self.error = error
        self.hint = hint


def configure_streams() -> None:
    """强制 UTF-8。

    中文会从 stdin 进来（路径、提示词）。Windows 上 Python 默认按系统代码页（cp936）解 stdin，
    一旦调用方没有设 PYTHONIOENCODING，提示词就会变成乱码送进模型，表现是"转出空的"。
    与其依赖环境变量，不如在入口处强制 UTF-8——少一个用户要配的东西。
    """
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


# ---------------------------------------------------------------- DLL 预加载
# os.add_dll_directory 返回的是**句柄**：句柄被 GC 回收时，它加进去的搜索路径立刻失效。
# 这里必须把句柄存住，否则会出现「DLL 明明在 site-packages 里、却报 not found」的鬼故事。
_DLL_HANDLES: list[object] = []


def add_dll_dirs() -> list[str]:
    """把 ctranslate2 需要的 CUDA 运行时目录挂进搜索路径。

    Windows 上 `Library cublas64_12.dll is not found` 是这里的第一高频故障。
    DLL 可能来自三处：pip 的 nvidia/*/bin、torch/lib（如果机器上正好有）、PATH 里已有的 CUDA。
    """
    added: list[str] = []
    if not hasattr(os, "add_dll_directory"):
        return added

    roots: list[Path] = []
    try:
        roots.extend(Path(p) for p in site.getsitepackages())
    except Exception:
        pass
    try:
        roots.append(Path(site.getusersitepackages()))
    except Exception:
        pass
    roots.append(Path(sys.prefix) / "Lib" / "site-packages")

    for root in roots:
        nvidia = root / "nvidia"
        if nvidia.is_dir():
            for pkg in nvidia.iterdir():
                for candidate in (pkg / "bin", pkg / "lib"):
                    if candidate.is_dir():
                        try:
                            _DLL_HANDLES.append(os.add_dll_directory(str(candidate)))
                            added.append(str(candidate))
                        except OSError:
                            pass
        torch_lib = root / "torch" / "lib"
        if torch_lib.is_dir():
            try:
                _DLL_HANDLES.append(os.add_dll_directory(str(torch_lib)))
                added.append(str(torch_lib))
            except OSError:
                pass

    # 光有 add_dll_directory 不够：ctranslate2 解析 cublas64_12.dll 时不走那条搜索路径，
    # 实测只有把目录塞进 PATH 才会被找到。两条路都铺上，代价几行代码，换来用户不必配环境变量。
    if added:
        os.environ["PATH"] = ";".join(added) + ";" + os.environ.get("PATH", "")
    return added


# ---------------------------------------------------------------- 音频探测
def probe_wav(path: str) -> dict:
    """轻量读 wav：时长、峰值、频谱平坦度、300–3400 Hz 频带占比。

    为什么值得算这几个数：whisper 不会说"这是噪声"，它会把噪声编成一句人话
    （实测白噪 → 「請不吝點贊訂閱轉發打賞」。）所以要有一个**不看模型**的信号
    来判断"这段到底是不是人在说话"，从音频本身拿。
    """
    info = {"duration": None, "peak": None, "flatness": None, "band": None}
    try:
        with wave.open(path, "rb") as f:
            frames, rate, width = f.getnframes(), f.getframerate() or 1, f.getsampwidth()
            info["duration"] = round(frames / rate, 3)
            raw = f.readframes(min(frames, rate * 60))   # 最多分析 60 秒，够判了
    except Exception:
        return info

    if not raw:
        return info
    try:
        import numpy as np
    except Exception:
        return info

    if width != 2:
        return info
    audio = np.frombuffer(raw[: len(raw) - (len(raw) % 2)], dtype="<i2").astype(np.float32) / 32768.0
    if audio.size == 0:
        return info
    info["peak"] = round(float(np.abs(audio).max()), 4)

    win = max(1, int(rate * 0.032))
    if audio.size < win * 2:
        return info
    frames2d = np.lib.stride_tricks.sliding_window_view(audio, win)[:: win // 2]
    mag = np.abs(np.fft.rfft(frames2d * np.hanning(win), axis=1)) + 1e-10
    flatness = np.exp(np.mean(np.log(mag), axis=1)) / np.mean(mag, axis=1)
    freqs = np.fft.rfftfreq(win, 1 / rate)
    voice_band = (freqs >= 300) & (freqs <= 3400)
    band = mag[:, voice_band].sum(axis=1) / mag.sum(axis=1)
    energy = np.sqrt((frames2d ** 2).mean(axis=1))
    active = energy > (energy.max() * 0.05)
    if active.any():
        info["flatness"] = round(float(flatness[active].mean()), 3)
        info["band"] = round(float(band[active].mean()), 3)
    return info


# ---------------------------------------------------------------- 幻觉判定
def artifact_hit(text: str) -> str | None:
    """命中"模型在瞎编"的套话签名。

    规则：命中的片段必须占全文一半以上。whisper 的幻觉套话是**整段**复读，
    所以这个比例不会误伤正常句子；而「帮我订阅一下」这类正常短句因为比例低会被放过。
    """
    import re

    norm = re.sub(r"[\s\W_]+", "", text.lower())
    if not norm:
        return None
    for group in HALLUCINATION_PATTERNS:
        for pattern in group:
            key = re.sub(r"[\s\W_]+", "", pattern.lower())
            if key and key in norm and len(key) / len(norm) >= 0.5:
                return pattern
    return None


def prompt_echo(text: str, prompt: str) -> bool:
    """模型把提示词本身复读出来了吗。

    这是 initial_prompt 的经典副作用：音频里没什么内容时，whisper 会接着提示词往下写。
    实测两种形态都要拦：
      1. 整段照抄——「以下是普通话的句子。」
      2. **提示词碎片拼出来**——给了一串人名/术语后，听不清的音频上会吐出
         「大肥鱼，本鱼，粤语，粤语，粤语…」（顺序还被它打乱，所以不能用"包含"判断）
    第 2 种用二元组覆盖率判：文本里的相邻两字，八成以上都能在提示词里找到 → 就是在复读。
    """
    if not prompt or not text:
        return False
    import re

    norm = lambda s: re.sub(r"[\s\W_]+", "", s.lower())  # noqa: E731
    t, p = norm(text), norm(prompt)
    if not t or not p:
        return False
    if t in p or (len(p) >= 6 and p in t and len(t) <= len(p) * 1.5):
        return True
    if len(t) < 6:
        return False
    prompt_bigrams = {p[i:i + 2] for i in range(len(p) - 1)}
    grams = [t[i:i + 2] for i in range(len(t) - 1)]
    if not grams:
        return False
    covered = sum(1 for g in grams if g in prompt_bigrams) / len(grams)
    return covered >= 0.8


def nonsense_repetition(text: str) -> tuple[str, int] | None:
    """检出「单音节/小片段复读」——把人声的乱叫当成话的最主要来源。

    真实数据（本机 21 条 QQ 语音条）里撞到的三个例子：
      「哒」×42        —— 用户故意乱叫，5.1 秒
      「鸟鸟鸟鸟」      —— 2.2 秒
      「叭噶 叭噶」     —— 2.4 秒
    这些音频的频谱特征**和真语音一模一样**（人在发声），置信度也一模一样
    （avg_logprob -0.17，比正常语音的 -0.24 还高）。换句话说：任何基于
    「模型有多确定」的判据都拦不住它们，只能看**文字的形状**。

    判据：二元组（两个连续字）重复占据全文一半以上。中文正常句子几乎不会这样；
    「谢谢谢谢」「好的好的」这类短的重复礼貌语因为长度不足 10 字会被放过（宁可漏，不可误杀）。

    这个判据同时兜住了另一类事故：给了提示词时，模型在听不清的音频上会把提示词本身
    复读成一长串（实测「大肥鱼，本鱼，粤语，粤语，粤语…」），也属于同一种形状。
    """
    import re

    norm = re.sub(r"[\s\W_]+", "", text)
    if len(norm) < 10:
        return None
    bigrams: dict[str, int] = {}
    for i in range(len(norm) - 1):
        gram = norm[i:i + 2]
        bigrams[gram] = bigrams.get(gram, 0) + 1
    top, count = max(bigrams.items(), key=lambda kv: kv[1])
    if count * 2 / len(norm) >= 0.5:
        return top, count
    return None


# 真人真的会说的短重复（道谢、应和、笑）。短复读判据必须放过这些，
# 否则「谢谢谢谢」「对对对」会被当成乱叫——这比漏一条乱叫更让人恼火。
COMMON_SHORT_REPEATS = (
    "谢谢", "好的", "对对", "是是", "行行", "嗯嗯", "哈哈", "嘿嘿", "拜拜", "好好",
    "收到", "知道", "可以", "没事", "加油", "晚安", "早", "嗯",
)


def short_repeat_gibberish(text: str) -> str | None:
    """短句里的单字/双字复读——「鸟鸟鸟鸟」「叭噶 叭噶」这种。

    为什么需要单独一条：长复读（≥10 字）由 nonsense_repetition 处理；但真实数据里
    乱叫经常只有 3–5 个字，长度不够，只能靠「字太少、重复太多、又不是常用礼貌语」来判断。
    代价是「快快快快」这类冷门重复会被误判——比把乱叫当事实转述出去要好。
    """
    import re

    norm = re.sub(r"[\s\W_]+", "", text)
    if not (3 <= len(norm) <= 7):
        return None
    if len(set(norm)) / len(norm) > 0.6:
        return None
    if any(word in norm for word in COMMON_SHORT_REPEATS):
        return None
    return norm


def repeated_phrase(segments: list[dict]) -> tuple[str, int] | None:
    """检出「同一句复读」型幻觉。返回 (那句话, 重复次数) 或 None。"""
    texts = [s["text"].strip() for s in segments if s["text"].strip()]
    if len(texts) < 3:
        return None
    counts: dict[str, int] = {}
    for t in texts:
        counts[t] = counts.get(t, 0) + 1
    top, count = max(counts.items(), key=lambda kv: kv[1])
    if count >= 3 and count / len(texts) >= 0.5:
        return top, count
    return None


def decide(*, text: str, segments: list[dict], duration: float, peak, flatness,
           lang_prob: float, language: str, avg_logprob: float, no_speech: float,
           thresholds: dict, prompt: str = "") -> tuple[str, str]:
    """把几个数折成一个敢写进日志的结论。

    原则：宁可说"没听清"，也不把猜出来的话当事实交出去。
    whisper 不会说"我不懂"，它一定会凑一句通顺的——这是它最危险的地方。

    判定不是简单比阈值，因为三条信号各有各的坑（都是实测出来的）：

      no_speech_prob  静音上很高（0.87）✓，但**带背景音的正常语音也能到 0.67**，
                      单独否决会误杀真话。→ 只在 avg_logprob 也差时才采信。
      avg_logprob     字音可信度。糊掉（-1.2）时可靠；幻觉时它也高（-0.3），靠不住。
      language_prob   短句天然偏低，zh/en/yue 放宽到 0.5。
      重复检测        静音幻觉会出现「同一句复读 10 次」，这是概率拦不住的，
                      而且在正常语音里几乎不会发生。→ 最可靠的幻觉信号。
    """
    ns_max = float(thresholds["no_speech_prob_max"])
    al_min = float(thresholds["avg_logprob_min"])
    ns_floor = float(thresholds.get("no_speech_logprob_floor", -0.6))
    lp_min = float(thresholds.get("lang_prob_min") or 0.7)
    if lang_prob:
        lp_min = LANG_RELAXED.get(str(language), lp_min)

    # 「有没有声音」必须在「有没有字」之前判。反过来就会出这种事：
    # 数字静音（峰值 0）上 whisper 编出「词曲 李宗盛」，因为 text 非空，
    # 前面的空字符串分支被跳过，最后按"文字"去比阈值——那当然比得过。
    if peak is not None and peak < 0.002:
        return "silent", "这段几乎完全没有声音（峰值接近 0），像是空录音或麦克风没开。"

    if not text:
        if duration and duration < 0.4:
            return "empty", "音频太短（不到 0.4 秒），没有可识别的内容。"
        return "empty", "没转出任何文字。可能是纯音乐、环境音，或者音轨本身就是空的。"

    # 先问"这是不是人在说话"（看音频），再问"模型听到了什么"（看文字）。
    # 顺序反了就会被幻觉牵着走：噪声先被编成一句话，再拿这句话去比阈值，怎么比都像真的。
    if flatness is not None and flatness >= NOISE_FLATNESS_MAX:
        return "no_speech", (f"这段听起来是噪声（频谱平坦度 {flatness:.2f} ≥ {NOISE_FLATNESS_MAX}），"
                             f"不像人在说话，内容不可信。")

    pattern = artifact_hit(text)
    if pattern:
        return "no_speech", f"模型给的是套话（命中「{pattern}」），这类内容通常出自静音段，不算真听清。"

    if prompt_echo(text, prompt):
        return "no_speech", "模型只是把提示词复读了一遍——这段音频里没有它能抓住的话。"

    nonsense = nonsense_repetition(text)
    if nonsense:
        return "no_speech", (f"同一小段「{nonsense[0]}」在 {len(text)} 个字里重复了 {nonsense[1]} 次——"
                             f"像单音节或语气词，不是能当事实转述的句子。")

    short = short_repeat_gibberish(text)
    if short:
        return "no_speech", (f"「{short}」的用字太单一（{len(set(short))} 种字 / {len(short)} 个字）——"
                             f"更像是顺手发出的声音，而不是一句话。")

    loop = repeated_phrase(segments)
    if loop:
        return "no_speech", (f"同一句「{loop[0]}」重复了 {loop[1]} 次——这是静音段上的经典幻觉，"
                             f"不是真的听到了什么。")

    # 语速上限：中文正常说话约 4–6 字/秒，超过 12 基本只能是幻觉或时间轴错乱。
    if duration >= 1.0 and len(text) / duration > 12:
        return "unclear", (f"文字密度不合理（{len(text)/duration:.0f} 字/秒），"
                           f"时间轴对不上，转出来的内容不可信。")

    if no_speech > ns_max and avg_logprob < ns_floor:
        return "no_speech", (f"既不像人说话（no_speech_prob {no_speech:.2f} 超过 {ns_max:.2f}），"
                             f"字音也含糊（avg_logprob {avg_logprob:.2f} 低于 {ns_floor:.2f}）。")

    if avg_logprob < al_min:
        return "unclear", (f"字音对不上（avg_logprob {avg_logprob:.2f} 低于 {al_min:.2f}），"
                           f"这条大概糊了——离话筒远、有风、或者语速太快。")

    if str(language) not in ("zh", "en", "yue") and lang_prob and lang_prob < lp_min:
        return "unclear", f"语种没把握（{lang_prob:.2f} < {lp_min:.2f}），转出来的字不可信。"

    hints = []
    if peak is not None and peak < 0.02:
        hints.append("音量很低，建议贴近话筒")
    if no_speech > ns_max:
        hints.append("背景音偏重，但字音可信，已按听得清处理")
    tail = f"（{'；'.join(hints)}）" if hints else ""
    return "heard", f"听清了{tail}"


def _looks_latin(text: str) -> bool:
    if not text:
        return False
    return sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff") == 0


def _import_whisper():
    try:
        from faster_whisper import WhisperModel
        return WhisperModel
    except Exception as exc:  # noqa: BLE001
        raise EarsFailure(
            f"faster-whisper 加载失败：{exc}",
            "这个 Python 里没装 faster-whisper。指向一个装好的解释器，"
            "或在它上面 `pip install -r python/requirements.txt`。",
        ) from exc


def _attempts(device: str, compute: str) -> list[tuple[str, str]]:
    """决定先试哪套 (device, compute)。

    这里有一个必须收口的细节：CPU 后端不支持 int8_float16/float16。
    用户把 device 写成 cpu、而 compute 还留着 GPU 默认值 int8_float16 时，
    ctranslate2 会直接抛 "Requested int8_float16 compute type, but the target device ..."，
    表现为"改成 CPU 反而加载不了"。所以按设备把精度夹到合法值。
    """

    def clamp_ok(dev: str, comp: str) -> str:
        if dev == "cpu" and comp in ("int8_float16", "float16", "bfloat16"):
            return "int8"
        return comp

    if device == "auto":
        cuda_comp = "int8_float16" if compute == "auto" else compute
        return [("cuda", clamp_ok("cuda", cuda_comp)), ("cpu", "int8")]
    comp = compute if compute != "auto" else ("int8_float16" if device == "cuda" else "int8")
    return [(device, clamp_ok(device, comp))]


def _friendly_model_error(joined: str) -> str:
    low = joined.lower()
    if "cublas" in low or "cudnn" in low:
        return ("CUDA 运行库缺失（cublas64_12.dll / cudnn_ops64_9.dll）。"
                "装 `nvidia-cublas-cu12` + `nvidia-cudnn-cu12`，或把 device 改成 cpu。")
    if "no such file" in low or "not found" in low:
        return "模型目录不对。medium 需要 model.bin + config.json + tokenizer.json + vocabulary.txt 四个文件。"
    if "out of memory" in low:
        return "显存不够。把 compute 换成 int8，或者换 small 模型。"
    return "见 README 的『报错对照』。"


class Ears:
    """一个进程内只该有一个实例：模型加载贵（medium 约 5.6 秒），重复加载纯属浪费。"""

    def __init__(self, model_ref: str, device: str = "auto", compute: str = "auto"):
        self.model_ref = model_ref
        self.device = (device or "auto").lower()
        self.compute = (compute or "auto").lower()
        self.model = None
        self.used = ("", "")
        self.load_seconds = 0.0
        self.dll_dirs: list[str] = []

    def ensure(self) -> None:
        if self.model is not None:
            return
        started = time.perf_counter()
        self.dll_dirs = add_dll_dirs()
        WhisperModel = _import_whisper()
        errors = []
        for dev, comp in _attempts(self.device, self.compute):
            try:
                self.model = WhisperModel(self.model_ref, device=dev, compute_type=comp)
                self.used = (dev, comp)
                self.load_seconds = round(time.perf_counter() - started, 2)
                return
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{dev}/{comp}: {exc}")
        joined = "\n".join(errors)
        raise EarsFailure("模型加载失败\n" + joined, _friendly_model_error(joined))

    def drop(self) -> None:
        self.model = None
        self.used = ("", "")

    def transcribe(self, req: dict) -> dict:
        """一次转写。req 字段见 README；返回结果 dict（不含 ok 字段的约定由调用方补）。"""
        audio = req.get("audio")
        if not audio:
            raise EarsFailure("缺少 audio 字段", "传入音频文件的绝对路径")
        if not Path(audio).is_file():
            raise EarsFailure(f"文件不存在：{audio}",
                              "确认路径存在；相对路径会按 DSH 工作目录解析，建议用绝对路径")

        thresholds = dict(DEFAULT_THRESHOLDS)
        thresholds.update({k: v for k, v in (req.get("thresholds") or {}).items() if v is not None})

        language = req.get("language") or "auto"
        wav_info = probe_wav(audio)
        max_seconds = float(req.get("maxSeconds") or 600)
        if wav_info["duration"] and wav_info["duration"] > max_seconds:
            raise EarsFailure(
                f"音频太长：{wav_info['duration']:.0f} 秒，超过上限 {max_seconds:.0f} 秒",
                "调大配置里的 maxSeconds，或者先把长音频切成几段再转。",
            )

        kwargs = dict(
            language=None if language == "auto" else language,
            beam_size=int(req.get("beamSize") or 5),
            vad_filter=False,           # 短语音开 VAD 会把整条判成非语音，几乎拿不到词
            condition_on_previous_text=False,
            word_timestamps=False,
            temperature=0.0,
        )
        if req.get("prompt"):
            kwargs["initial_prompt"] = str(req["prompt"])
            used_prompt = str(req["prompt"])
        elif language == "zh":
            kwargs["initial_prompt"] = ZH_DEFAULT_PROMPT
            used_prompt = ZH_DEFAULT_PROMPT
        else:
            used_prompt = ""

        self.ensure()
        started = time.perf_counter()
        segments_iter, info = self._run(audio, kwargs)

        segments = []
        for seg in segments_iter:      # 生成器，真正的解码发生在这里
            text = (seg.text or "").strip()
            if not text:
                continue
            segments.append({
                "start": round(float(seg.start), 2),
                "end": round(float(seg.end), 2),
                "text": text,
                "avg_logprob": round(float(seg.avg_logprob), 3),
                "no_speech_prob": round(float(seg.no_speech_prob), 3),
            })
        # 计时必须包到消费完为止：faster-whisper 返回的是生成器，调用那一刻几乎不花时间，
        # 把计时停在调用处会得出「解码 0.02 秒」这种假的漂亮数字。
        decode_seconds = round(time.perf_counter() - started, 2)

        text = "".join(s["text"] for s in segments).strip()
        if _looks_latin(text):
            text = " ".join(text.split())

        duration = float(getattr(info, "duration", 0.0) or wav_info["duration"] or 0.0)
        lang_prob = float(getattr(info, "language_probability", 0.0) or 0.0)
        avg_logprob = sum(s["avg_logprob"] for s in segments) / len(segments) if segments else -99.0
        no_speech = max((s["no_speech_prob"] for s in segments), default=1.0)

        verdict, reason = decide(
            text=text, segments=segments, duration=duration, peak=wav_info["peak"],
            flatness=wav_info["flatness"], lang_prob=lang_prob,
            language=getattr(info, "language", None) or language,
            avg_logprob=avg_logprob, no_speech=no_speech, thresholds=thresholds,
            prompt=used_prompt,
        )

        return {
            "verdict": verdict,
            "reason": reason,
            "text": text if verdict == "heard" else "",
            # 只有 unclear 才给原文：其它失败态下那段字是噪声上编出来的，
            # 交出去只会误导下一个读到它的人（或 agent）。
            "partialText": text if verdict == "unclear" else "",
            "language": getattr(info, "language", None) or language,
            "language_probability": round(lang_prob, 3),
            "avg_logprob": round(avg_logprob, 3),
            "no_speech_prob": round(no_speech, 3),
            "duration": round(duration, 2),
            "flatness": wav_info["flatness"],
            "band": wav_info["band"],
            "decodeSeconds": decode_seconds,
            "loadSeconds": self.load_seconds,
            "device": self.used[0],
            "compute": self.used[1],
            "model": str(self.model_ref),
            "prompt": used_prompt,
            "thresholds": thresholds,
            "segments": segments,
        }

    def _run(self, audio: str, kwargs: dict):
        """调 transcribe；CUDA 在解码阶段才炸时（cublas 缺失的典型表现）自动退 CPU 重试一次。"""
        try:
            return self.model.transcribe(audio, **kwargs)
        except Exception as exc:  # noqa: BLE001
            low = str(exc).lower()
            if self.used[0] == "cuda" and ("cublas" in low or "cudnn" in low or "cuda" in low):
                self.drop()
                self.device, self.compute = "cpu", "int8"
                self.ensure()
                return self.model.transcribe(audio, **kwargs)
            raise


def run_once(req: dict) -> dict:
    """一次性调用：加载 + 转写，返回完整结果。"""
    ears = Ears(req.get("model") or "medium", req.get("device") or "auto", req.get("compute") or "auto")
    result = ears.transcribe(req)
    result["ok"] = True
    result["audio"] = str(req.get("audio"))
    result["dllDirs"] = len(ears.dll_dirs)
    return result
