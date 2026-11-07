# 更新日志

## 0.1.0 — 2026-11-08

第一版。范围刻意压到最小：**一段音频进，一句可信的话出**——包括 QQ 的 SILK 语音条。

### 有

- `ears_transcribe`：本地转写，返回 `verdict` + 文字 + 三个可信度 + 时长/耗时/设备
- `ears_status`：环境自检（解释器、依赖版本、模型位置、ffmpeg、配置路径）
- **QQ / 微信 SILK 语音条直读**：`silk-wasm`（MIT，纯 WASM）解码，免编译
- 五档判定：`heard` / `unclear` / `no_speech` / `silent` / `empty`
- 五层反幻觉：**峰值静音检测**、**频谱平坦度**（拦噪声）、**复读检测**（拦静音幻觉）、
  **提示词回声检测**（拦"复读提示词"）、**套话签名**（简繁双语）
- 输入格式按**魔数**识别，不信扩展名（QQ 的 `.amr` 里装的其实是 SILK）
- 非 wav 自动 ffmpeg 转 16k 单声道，带 `highpass=80` + `dynaudnorm`（救回糊掉的手机录音）
- 中文默认提示词：简体化 + 补标点 + 专名更准（实测「小飞鱼」→「小肥鱼」）
- CUDA 不可用自动退 CPU，并把"为什么退"写进结果
- 不需要 torch（约 1.5 GB 依赖，而不是 7 GB）
- 中英双语文档、本地化标题、原创图标

### 没有（故意的）

- 无常驻服务。每次调用重新加载模型，实测多花 5.6 秒 → 0.2.0
- 无设置页。配置改 `~/.dsh/voice-ears.json` → 0.3.0
- 无依赖自举。要手动 `pip install` → 0.3.0
- 无框架适配层（SnowLuma / onebot 自动丢语音条）→ 0.4.0
- 只验证了 Windows

### 实测数据（本机 RTX 4060 / medium / int8_float16）

| 输入 | 判定 | 说明 |
|---|---|---|
| 5.5 s 合成语音（wav / mp3 / **silk**） | `heard` | 三条路结果一致 |
| 5 s 纯静音（数字零） | `silent` | whisper 编出「词曲 李宗盛」，被峰值检测拦住 |
| 5 s 白噪 | `no_speech` | whisper 编出「請不吝點贊…」，被频谱平坦度（0.84）拦住 |
| 5 s 粉噪 | `no_speech` | — |
| 440 Hz 纯音 | `unclear` | 落到"猜"，但没有硬说听清 |
| 真 QQ 语音条 ×5 | 4 条 `heard` + 1 条哼音 `no_speech` | 转完即删，不入仓库 |

### 踩过的坑（都写进代码注释了）

- `os.add_dll_directory` 返回的是**句柄**，被 GC 回收后搜索路径失效 → 必须存住句柄
- 光有 `add_dll_directory` 不够，ctranslate2 只在 `PATH` 里找 `cublas64_12.dll` → 两条路都得铺
- CUDA 模型**加载成功**、`transcribe` 时才因 cublas 炸 → 加载和解码要一起兜底重试
- `no_speech_prob` 单独当否决条件会误杀正常语音（实测 0.56–0.67）→ 必须和 `avg_logprob` 一起看
- 中文黑名单只写简体，whisper 吐的是繁体 → 简繁两套都要写
- 短语音开 VAD 会把整条判成非语音 → 默认关
- Windows 上 Python 按系统代码页解 stdin，中文提示词变乱码 → 主动 `reconfigure(utf-8)`
- 「有没有声音」的判定必须排在「有没有文字」之前，否则数字静音会被编出的字带偏
- `initial_prompt` 会让模型在静音上**复读提示词** → 专门加了一条回声检测
- `pilk`（另一个 SILK 解码库）是 GPLv3 + C 扩展，和 MIT 冲突且要 MSVC → 换 `silk-wasm`
