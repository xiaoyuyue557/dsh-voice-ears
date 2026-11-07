/**
 * config.js — 配置读写与 Python 解释器选择。
 *
 * 配置放在 `~/.dsh/voice-ears.json`（和 DSH 其它插件同一层目录），不存在就用默认值，
 * 第一次显式保存时才落盘。这样用户"什么都没配"也能跑，而不是被一个空文件卡住。
 */
import { spawn } from 'node:child_process';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { homedir } from 'node:os';
import { dirname, isAbsolute, join } from 'node:path';

export const CONFIG_PATH = join(homedir(), '.dsh', 'voice-ears.json');

export const DEFAULTS = {
  python: '',        // 空 = 自动找（见 resolveInterpreters）
  model: 'medium',   // faster-whisper 模型名，或一个 CTranslate2 模型的目录
  modelDir: '',      // 想指向本地模型目录时填这里，优先于 model
  language: 'zh',    // 默认按中文；传 auto 让模型自己判断
  device: 'auto',    // auto | cuda | cpu
  compute: 'auto',   // auto | int8_float16 | float16 | int8 | float32
  ffmpeg: '',        // 空 = 用 PATH 里的 ffmpeg
  prompt: '',        // 给识别器的提示词（人名、黑话）
  maxSeconds: 600,   // 超过就不转（防止有人丢进来两小时播客）
  timeoutMs: 300000, // 单次调用上限
  extraPath: [],     // 额外塞进子进程 PATH 的目录（例如 torch\lib）
  thresholds: {
    no_speech_prob_max: 0.6,
    avg_logprob_min: -1.0,
    lang_prob_min: 0.7,
    no_speech_logprob_floor: -0.6,
  },
};

export function normalize(raw = {}) {
  const merged = {
    ...DEFAULTS,
    ...Object.fromEntries(Object.entries(raw).filter(([, v]) => v !== undefined && v !== null)),
    thresholds: { ...DEFAULTS.thresholds, ...(raw.thresholds ?? {}) },
  };
  merged.extraPath = Array.isArray(merged.extraPath) ? merged.extraPath.filter(Boolean) : [];
  merged.timeoutMs = clamp(Number(merged.timeoutMs) || DEFAULTS.timeoutMs, 5000, 3600000);
  merged.maxSeconds = clamp(Number(merged.maxSeconds) || DEFAULTS.maxSeconds, 1, 7200);
  return merged;
}

export function load(env = process.env) {
  if (env.DSH_VOICE_EARS_CONFIG && existsSync(env.DSH_VOICE_EARS_CONFIG)) {
    try { return normalize(JSON.parse(readFileSync(env.DSH_VOICE_EARS_CONFIG, 'utf8'))); } catch { /* 落到默认 */ }
  }
  try {
    return normalize(JSON.parse(readFileSync(CONFIG_PATH, 'utf8')));
  } catch {
    return normalize({});
  }
}

export function save(patch = {}) {
  const next = normalize({ ...load(), ...patch });
  mkdirSync(dirname(CONFIG_PATH), { recursive: true });
  writeFileSync(CONFIG_PATH, `${JSON.stringify(next, null, 2)}\n`, 'utf8');
  return next;
}

/** 模型引用：给了 modelDir 就用它，否则用模型名（faster-whisper 会自己去 HF 缓存里找/下载）。 */
export function modelRef(cfg) {
  if (cfg.modelDir) return cfg.modelDir;
  if (cfg.model && (isAbsolute(cfg.model) || cfg.model.startsWith('.'))) return cfg.model;
  return cfg.model || 'medium';
}

function clamp(n, lo, hi) {
  return Math.min(hi, Math.max(lo, n));
}

/**
 * 按优先级列解释器候选。
 *
 * 这里刻意不做"扫全盘找能用的 python"：那会让第一次调用变得不可预测（在别人的机器上
 * 扫盘几秒钟却什么也没找到）。只用明确的几处，找不到就报错让用户配 `python`。
 */
export function resolveInterpreters(cfg, env = process.env) {
  const list = [
    cfg.python,
    env.DSH_VOICE_EARS_PYTHON,
    env.VOICE_EARS_PYTHON,
  ].filter(Boolean);
  // Windows 上 python 常常只是应用商店的占位符，py 启动器更可靠，所以两个都试。
  list.push('python', 'python3');
  if (process.platform === 'win32') list.push('py');
  return [...new Set(list)];
}

const probeCache = new Map();

/**
 * 探一次解释器：能不能 import faster_whisper、什么版本。
 * 结果按解释器缓存，避免每次转写都多花 1–2 秒。
 */
export function probePython(python, { env = process.env, timeoutMs = 20000 } = {}) {
  if (probeCache.has(python)) return probeCache.get(python);
  const promise = new Promise((resolve) => {
    let child;
    try {
      child = spawn(python, [
        '-c',
        'import sys,json;\n'
        + 'try:\n'
        + '    import faster_whisper, ctranslate2\n'
        + '    print(json.dumps({"ok":True,"python":sys.version.split()[0],'
        + '"faster_whisper":getattr(faster_whisper,"__version__","?"),'
        + '"ctranslate2":getattr(ctranslate2,"__version__","?")}))\n'
        + 'except Exception as e:\n'
        + '    print(json.dumps({"ok":False,"error":str(e),"python":sys.version.split()[0]}))\n',
      ], { stdio: ['ignore', 'pipe', 'pipe'], windowsHide: true, env });
    } catch (error) {
      resolve({ ok: false, error: String(error?.message ?? error), python, interpreter: python });
      return;
    }
    const out = [];
    const err = [];
    const timer = setTimeout(() => { try { child.kill(); } catch { /* ignore */ } }, timeoutMs);
    timer.unref?.();
    child.stdout.on('data', (b) => out.push(b));
    child.stderr.on('data', (b) => err.push(b));
    child.on('error', (e) => {
      clearTimeout(timer);
      resolve({ ok: false, error: String(e?.message ?? e), python, interpreter: python });
    });
    child.on('close', (code) => {
      clearTimeout(timer);
      const text = Buffer.concat(out).toString('utf8').trim();
      try {
        const parsed = JSON.parse(text.split('\n').filter(Boolean).pop() ?? '{}');
        resolve({ ...parsed, interpreter: python, exitCode: code });
      } catch {
        resolve({
          ok: false,
          error: Buffer.concat(err).toString('utf8').trim() || `退出码 ${code}`,
          interpreter: python,
        });
      }
    });
  });
  probeCache.set(python, promise);
  return promise;
}

/** 找到第一个能 import faster_whisper 的解释器；都没有就返回最后一次的错误。 */
export async function pickPython(cfg, env = process.env) {
  const tried = [];
  for (const candidate of resolveInterpreters(cfg, env)) {
    const result = await probePython(candidate, { env });
    if (result.ok) return { python: candidate, probe: result, tried };
    tried.push({ python: candidate, error: result.error });
  }
  return { python: null, tried };
}

export function clearProbeCache() {
  probeCache.clear();
}
