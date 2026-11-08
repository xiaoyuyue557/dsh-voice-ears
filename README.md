# dsh-voice-ears · 语音耳朵

[![npm](https://img.shields.io/npm/v/dsh-voice-ears.svg)](https://www.npmjs.com/package/dsh-voice-ears)
[![license](https://img.shields.io/npm/l/dsh-voice-ears.svg)](LICENSE)

> 给 DSH 里的 agent 装一双**本地**耳朵：把别人发来的语音转成文字，**没听清的时候敢说没听清**。

定位很窄，就一件事：**听得懂别人发的语音条**。

- 它**不是**麦克风听写（那是 [`dsh-ears`](https://www.npmjs.com/package/dsh-ears) 的活）
- 它**不是**音乐分析工具箱（那是 [`dsh-soundtry`](https://github.com/CixiSVesper/dsh-soundtry) 的活）
- 它面向 **QQ / 微信 bot 玩家**：对话另一方发来一条语音，你的 agent 需要知道那条语音说了什么

全离线，不上传，不需要 torch，默认模型 whisper **medium**。

**当前版本 0.2.0**：转写 + 五档判定 + **QQ 语音条（SILK）直读** + **常驻 worker**（模型只加载一次，之后每条 1.6 秒）。
设置页和依赖自举还没做（见[路线图](#路线图)）。

---

## 为什么不是又一个 whisper 包装

因为 whisper **不会说"我听不懂"**。它一定会凑出一句通顺的话。这些都是我在这台机器上实测的，不是猜的：

| 输入 | whisper 的输出 | 覆盖它的手段 |
|---|---|---|
| 5 秒白噪声 | 「請不吝點贊訂閱轉發打賞支持明鏡與點點欄目」 | 频谱平坦度 0.84 → 判噪声 |
| 5 秒数字静音 | 「词曲 李宗盛」 | 峰值 ≈ 0 → 判静音 |
| 5 秒纯静音 | 「一切都在你身上」× 14 遍 | 同一句复读 ≥ 50% → 判幻觉 |
| 8 秒真人语音（带背景音） | 转对了，但 `no_speech_prob` = 0.56–0.67 | 单独看这个数会**误杀真话**，所以不单独用它 |

所以这个插件交出来的不是一段文字，而是 **文字 + 一个判定**：

| verdict | 含义 | 你该怎么做 |
|---|---|---|
| `heard` | 听清了 | `text` 就是对方说的，可以引用 |
| `unclear` | 糊了 | `text` 是空的；`partialText` 只是猜的，**别当原话**。说没听清、请对方再说一遍 |
| `no_speech` | 这不是人声 | 噪声/音乐/静音。`text` 是空的 |
| `silent` | 几乎没有声音 | 空录音、麦克风没开 |
| `empty` | 没转出任何字 | 纯音乐、空轨、太短 |

判定顺序有讲究：**先问"这是不是人在说话"（看音频），再问"模型听到了什么"（看文字）**。反过来就会被幻觉牵着走——噪声先被编成一句话，再拿这句话去比阈值，怎么比都像真的。

---

## 能吃什么

| 输入 | 支持 | 说明 |
|---|---|---|
| `.wav` | ✅ | 直接读，不经过 ffmpeg |
| `.mp3` / `.m4a` / `.ogg` / `.amr` / `.flac` | ✅ | 内部用 ffmpeg 转 16k 单声道 |
| **`.silk`（QQ / 微信语音条）** | ✅ | `silk-wasm`（MIT，纯 WASM）解码，不需要编译 |
| 扩展名说谎的文件 | ✅ | 一律按**魔数**认格式 |

最后一条不是洁癖：**QQ 存下来的语音文件扩展名是 `.amr`，内容是 SILK**。按扩展名判断会把它交给 ffmpeg，然后得到一句看不懂的错。按魔数认就不存在这个问题。

---

## 安装

### 前提

1. **DeepSeek Harness**（桌面版或 CLI 都行）
2. **Python 3.10+**（3.12 验证过）
3. **ffmpeg**（非 wav 输入需要；QQ 语音条需要）

### 第一步：装 Python 依赖

```bash
python -m pip install -r python/requirements.txt
```

国内慢就换镜像：

```bash
python -m pip install -r python/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> **不需要 torch。** faster-whisper 背后是 CTranslate2，它直接吃 pip 装下来的 cublas/cudnn 运行时 DLL。
> 整套约 1.5 GB，比带 torch 的方案（约 7 GB）轻一个量级。

### 第二步：装插件

```bash
dsh plugin install dsh-voice-ears     # 从 npm
```

或者从 GitHub（国内直连 GitHub 常超时，用代理前缀）：

```bash
dsh plugin install https://gh-proxy.com/https://github.com/xiaoyuyue557/dsh-voice-ears/archive/refs/heads/main.tar.gz
```

> 从源码链接安装时，插件目录要单独跑一次 `npm install`（SILK 解码器是 npm 依赖）。

### 第三步：确认

重启 DSH 后让 agent 调 `ears_status`，应该看到：

```
耳朵可用 ✓
解释器：C:\...\python.exe
  python 3.12.16 · faster-whisper 1.2.1 · ctranslate2 4.8.5
模型：medium（首次调用会下载）
设备：auto / int8_float16
```

---

## 配置

`~/.dsh/voice-ears.json`，不存在就用默认值。全部字段：

| 字段 | 默认 | 说明 |
|---|---|---|
| `python` | `""` | 空 = 自动找（`python` / `python3` / `py`）。**指到装好 faster-whisper 的那个解释器** |
| `model` | `"medium"` | 模型名，或 CT2 模型目录 |
| `modelDir` | `""` | 指向本地模型目录时填这里，优先于 `model` |
| `language` | `"zh"` | 默认中文；`auto` 让模型自己判断 |
| `device` | `"auto"` | `auto` / `cuda` / `cpu`。`auto` 先试 GPU，失败自动退 CPU |
| `compute` | `"auto"` | `int8_float16`（GPU 最省显存）/ `float16` / `int8` / `float32` |
| `ffmpeg` | `""` | 空 = 用 PATH 里的 ffmpeg |
| `prompt` | `""` | 提示词：人名、昵称、黑话。**不填时中文会自动用一条默认提示词**（见下） |
| `resident` | `true` | 常驻 worker：模型只加载一次。设 `false` 则每条语音起一个进程（慢，但完全无状态） |
| `idleSeconds` | `600` | 常驻 worker 闲置多久自己退出（秒）。退出后显存/内存归还 |
| `maxSeconds` | `600` | 超过就不转（防止有人丢进来两小时播客） |
| `timeoutMs` | `300000` | 单次调用上限 |
| `extraPath` | `[]` | 额外塞进子进程 `PATH` 的目录（例如 `torch\lib`） |
| `thresholds` | 见下 | 判定阈值 |

```json
{
  "python": "C:\\Python312\\python.exe",
  "model": "medium",
  "language": "zh",
  "device": "auto",
  "compute": "int8_float16",
  "prompt": "大肥鱼，本鱼，小鱼干"
}
```

### 中文默认提示词

`language=zh` 且没填 `prompt` 时，会自动加上「以下是普通话的句子。」。实测（medium，2–8 秒语音条）它同时改善三件事：

1. 输出从**繁体变简体**（whisper 对短普通话经常吐繁体）
2. **补上标点**（否则一整句连读，agent 读起来费劲）
3. **专有名词更准**（「小飞鱼」→「小肥鱼」）

副作用是静音上模型可能把提示词原样复读出来，所以判定里有一条专门拦它。

### 阈值

```json
{
  "thresholds": {
    "no_speech_prob_max": 0.6,
    "avg_logprob_min": -1.0,
    "lang_prob_min": 0.7,
    "no_speech_logprob_floor": -0.6
  }
}
```

`no_speech_prob_max` 和 `no_speech_logprob_floor` 是**一起**用的：只有"不像人说话"**且**"字音也含糊"才判没听清。单看前者会误杀带背景音的正常语音（实测 0.56–0.67）。

---

## 工具

### `ears_transcribe`

```
input       必填。音频路径（wav/mp3/m4a/ogg/amr/flac/silk）
language    可选。zh / en / ja / auto
model       可选。本次调用覆盖模型
device      可选。auto / cuda / cpu
prompt      可选。本次调用的提示词
```

返回 `verdict` + `text` + 三个可信度数字 + 时长/耗时/设备。

### `ears_status`

不带参数。报告解释器、依赖版本、模型位置、ffmpeg、配置路径。装完先跑这个。

---

## 实测

本机：RTX 4060 8 GB / Python 3.12 / faster-whisper 1.2.1 / ctranslate2 4.8.5 / medium / `int8_float16`

| 输入 | 判定 | 冷（首条，含加载） | 热（常驻后） |
|---|---|---|---|
| 5.5 秒合成语音（wav） | `heard`「这是一条测试语音,请把这句话转成文字。」 | 8.4 s | **1.5 s** |
| 同上转成 mp3 | `heard`，结果一致 | — | 1.7 s |
| 同上编码成 **SILK** | `heard`，结果一致 | — | 1.7 s |
| 5 秒静音 | `silent` | — | 1.1 s |
| 5 秒白噪 | `no_speech` | — | 1.9 s |
| 5 秒粉噪 | `no_speech` | — | 1.1 s |
| 5 秒 440 Hz 纯音 | `unclear` | — | 1.1 s |

那 5.6 秒的模型加载**只付一次**：worker 常驻，闲置 10 分钟自己退出。

> 0.1 的时候每条语音都要重载模型（7.4 秒/条）。QQ 聊天的节奏下那样是没法用的，0.2 把它变成 1.5 秒。

三个必须对的行为都实测过：**闲置超时会自己退**（显存归还）、**被强杀后下一条自动重启**、**换模型/换设备自动换进程**。

真实 QQ 语音条验证（把本机 QQ 落盘的 5 条语音条逐一转写，转完即删，未留在仓库里）：

| 内容 | 结果 |
|---|---|
| 「测试刚下载的新插件」 | ✅ 一字不差 |
| 「语音测试小飞鱼听到赶紧回话」 | ✅（默认提示词把「小飞鱼」纠正成「小肥鱼」） |
| 「那現在可以聽清楚了嗎」 | ✅ 且自动转成简体「那现在可以听清楚了吗?」 |
| 一段哼音（嘟——） | ✅ 判 `no_speech`，没有硬编成一句话 |

---

## 已知限制

诚实列出来，都是实测撞到的：

- **2–3 秒的哼音/气声**可能落到 `no_speech`。它对"这不是说话"很敏感，代价是极短的语气词也会被拒。
- **纯音和褐噪**会落到 `unclear`（而不是 `no_speech`）：频谱平坦度取 0.6 是为了不误伤真话，代价是褐噪（实测 0.44）漏过去，最后由黑名单兜住一部分。
- **纯音乐**没有可靠的廉价判据。模型可能把歌词转出来，也可能编。
- **中文可能吐繁体**（默认提示词能大幅改善，但不是 100%）。
- 只验证了 **Windows**。
- 常驻 worker 会**占着显存**（medium / `int8_float16` 实测约 1.08 GB）直到闲置超时。显存紧张就把 `idleSeconds` 调小。
- **CPU 模式慢**：同一段 5.5 秒语音，GPU 1.5 秒、CPU 约 22 秒（int8）。CPU 能跑，但不是给人用的速度。
- 依赖还得手动 `pip install`。0.3 解决。

---

## 报错对照

| 你看到的 | 原因 | 怎么办 |
|---|---|---|
| `找不到能用的 Python：没有一个解释器能 import faster_whisper` | 依赖没装，或装到了别的 Python | 在报错列出的解释器上 `pip install -r requirements.txt`，或把 `python` 配成那个解释器 |
| `CUDA 运行库缺失（cublas64_12.dll / cudnn_ops64_9.dll）` | nvidia wheel 没装 / DLL 找不到 | `pip install nvidia-cublas-cu12 nvidia-cudnn-cu12`；或把 `device` 改成 `cpu` |
| `SILK 解码器没装（silk-wasm）` | npm 依赖缺失 | 在插件目录跑 `npm install` |
| `SILK 解码失败` | 文件不完整 / 不是真 SILK | 换一份文件；或者让框架先转 wav |
| `ffmpeg 没找到` | 没装 ffmpeg 或不在 PATH | 装一个，或配置里填 `ffmpeg` 绝对路径 |
| `音频太长：xxx 秒，超过上限 600 秒` | 触发了 `maxSeconds` | 切段，或调大 `maxSeconds` |
| `转写超时` | 音频太长 / CPU 太慢 | 调大 `timeoutMs`，换 `small` 模型，或用 GPU |
| 第一次调用卡很久 | 在下载模型（medium ≈ 1.5 GB） | 正常。国内可以 `set HF_ENDPOINT=https://hf-mirror.com` 再试 |

---

## 隐私

- 音频**不出本机**。没有网络请求（除了第一次下载模型）。
- 不写转写日志到磁盘。临时 wav 用完即删。
- 常驻服务（v0.2 起）只绑 `127.0.0.1`，token 写进 `~/.dsh/voice-ears/endpoint.json`（0600）。

---

## 路线图

| 版本 | 内容 | 状态 |
|---|---|---|
| **0.1.0** | 转写 + 五档判定 + SILK 直读 | ✅ 已完成 |
| **0.2.0** | 常驻 worker（消掉每条 5.6 秒加载）+ 闲置退出 + 崩溃自愈 | ✅ 已完成 |
| **0.3.0** | 设置页 + 依赖自举（一条命令装完依赖、下模型、带镜像） | 计划中 |
| **0.4.0** | 框架适配层（SnowLuma / onebot 直接把语音条丢过来） | 计划中 |

---

## 开发

```bash
npm install        # 拉 silk-wasm
npm test           # 端到端：真语音要听清（wav/mp3/silk）/ 静音白噪粉噪要拒 / 热路径要比冷路径快
node test/edge.mjs # 报错路径：坏 SILK / 路径错 / 非音频文件
```

`test/fixtures/` 里的音频都是自己合成的（Windows SAPI 中文语音 + ffmpeg 生成的噪声），没有第三方版权内容。

改动判定逻辑后**必须**跑一遍 `npm test`——这个插件最容易坏的地方不是代码写错，而是判定悄悄变松，而单看输出很像正常。

### 代码结构

```
python/ears_core.py   判定 + 模型加载（两条路共用，绝不允许复制第二份判定）
python/transcribe.py  一次性 CLI（冷路径）
python/worker.py      常驻 worker（热路径，一行一条 JSON）
src/worker.js         worker 生命周期：启动、串行队列、闲置退出、崩溃重启
src/runner.js         选路：常驻优先，worker 起不来才回退冷路径
src/media.js          格式认别（按魔数）、SILK 解码、ffmpeg 归一化
src/tools.js          两个工具的描述——verdict 怎么读，全写在这里
```

### 为什么不用 pilk

`pilk` 是另一个 Silk 解码库，但它是 **GPLv3 + C 扩展**：既和 MIT 冲突，用户机器上还得有 MSVC 才能编译。`silk-wasm` 是 MIT、纯 WASM、免编译，所以选它。

---

## 关于这个名字

npm 上的 `dsh-ears` 已经被另一个插件占了，而且它做的是**麦克风听写**（你对着话筒说话，往输入框里写字）。
那个功能面向"人用语音打字"，这个插件面向"agent 听懂别人发来的语音"——方向相反，所以名字也不该撞。
`dsh-voice-ears` 就是为这个区别取的。

## License

MIT
