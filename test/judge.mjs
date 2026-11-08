/**
 * judge.mjs — 判定层的单元测试（不需要模型，跑得很快）。
 *
 * 为什么单独有这个文件：判定逻辑是这个插件的全部价值，而它最容易被"顺手调一下阈值"改坏。
 * smoke.mjs 要加载模型（几十秒），judge.mjs 只喂数字给它（1 秒），所以改阈值前先跑这个。
 *
 * 用例全部来自真实数据：本机 21 条 QQ 语音条里撞到的乱叫、提示词复读，以及正常语句的对照。
 */
import { spawn } from 'node:child_process';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { load, pickPython } from '../src/config.js';

const HERE = dirname(fileURLToPath(import.meta.url));
const PY = `
import json, sys
sys.path.insert(0, r"${join(HERE, '..', 'python')}")
import ears_core
from ears_core import decide, nonsense_repetition, prompt_echo, short_repeat_gibberish

fails = []
def check(name, cond, detail=""):
    if cond:
        print("ok   " + name)
    else:
        fails.append(name)
        print("FAIL " + name + ("  " + detail if detail else ""))

# ---- 复读形状：真实数据里的三条乱叫
check("哒×42 判为复读", nonsense_repetition("哒" * 42) is not None)
check("鸟鸟鸟鸟 太短不判", nonsense_repetition("鸟鸟鸟鸟") is None)
check("叭噶叭噶 太短不判", nonsense_repetition("叭噶 叭噶") is None)
# ---- 提示词复读（实测事故）
check("提示词复读被认出", nonsense_repetition("大肥鱼，本鱼，" + "粤语，" * 8) is not None)
check("prompt_echo 命中碎片拼法", prompt_echo("大肥鱼，本鱼，粤语", "大肥鱼，本鱼，鲸鱼，粤语，小鱼干"))
check("prompt_echo 不误杀正常句", not prompt_echo("大肥鱼你吃饭了吗", "大肥鱼，本鱼，鲸鱼，粤语"))
# ---- 短复读（真实数据：鸟鸟鸟鸟 / 叭噶 叭噶）
check("鸟鸟鸟鸟 判为乱叫", short_repeat_gibberish("鸟鸟鸟鸟") is not None)
check("叭噶 叭噶 判为乱叫", short_repeat_gibberish("叭噶 叭噶") is not None)
for word in ("谢谢谢谢", "对对对", "哈哈哈", "好的好的"):
    check("短复读不误杀：" + word, short_repeat_gibberish(word) is None)
check("正常短句不误杀", short_repeat_gibberish("我不去了") is None)
# ---- 不许误杀的对照（真人会说这些）
for text in ("谢谢谢谢", "好的，好的，好的。", "对对对",
             "这条离远一点说了，我不知道你有没有听到。",
             "明明是中文，你怎么说？是粤语啊？你骗我吧？"):
    check("不误杀：" + text[:12], nonsense_repetition(text) is None)

\ndef d(**kw):
    base = dict(text="", segments=[], duration=5.0, peak=0.5, flatness=0.15,
                lang_prob=1.0, language="zh", avg_logprob=-0.2, no_speech=0.1,
                thresholds=dict(ears_core.DEFAULT_THRESHOLDS), prompt="")
    base.update(kw)
    return decide(**base)[0]

check("数字静音 → silent", d(text="词曲 李宗盛", peak=0.0) == "silent")
check("白噪 → no_speech", d(text="請不吝點贊訂閱", flatness=0.84) == "no_speech")
check("正常语音 → heard", d(text="这是一条测试语音", segments=[{"text": "这是一条测试语音", "avg_logprob": -0.2, "no_speech_prob": 0.1}]) == "heard")
check("乱叫（42 字复读）→ no_speech", d(text="哒" * 42, segments=[{"text": "哒" * 42, "avg_logprob": -0.17, "no_speech_prob": 0.48}]) == "no_speech")
check("糊掉 → unclear", d(text="哪里? 给你一本面的词啊。", avg_logprob=-1.2, segments=[{"text": "x", "avg_logprob": -1.2, "no_speech_prob": 0.2}]) == "unclear")
check("背景音重但字音可信 → heard", d(text="这条离远一点说了", no_speech=0.67, segments=[{"text": "这条离远一点说了", "avg_logprob": -0.21, "no_speech_prob": 0.67}]) == "heard")

print(json.dumps({"failed": fails}))
sys.exit(1 if fails else 0)
`;

const cfg = load();
const picked = await pickPython(cfg);
if (!picked.python) {
  console.log('找不到能用的 Python，跳过判定层测试');
  process.exit(0);
}

const code = await new Promise((resolve) => {
  const child = spawn(picked.python, ['-c', PY], { stdio: ['ignore', 'pipe', 'pipe'], windowsHide: true });
  const out = [];
  const err = [];
  child.stdout.on('data', (b) => out.push(b));
  child.stderr.on('data', (b) => err.push(b));
  child.on('close', (c) => resolve({ code: c, out: Buffer.concat(out).toString('utf8'), err: Buffer.concat(err).toString('utf8') }));
});

process.stdout.write(code.out);
if (code.code !== 0) {
  if (code.err.trim()) console.error(code.err.trim());
  console.log('\n判定层测试未通过');
  process.exit(1);
}
console.log('\n判定层全部通过');
