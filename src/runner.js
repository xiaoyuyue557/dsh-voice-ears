/**
 * runner.js — 起一个 Python 进程，把音频喂给它，拿回「文字 + 判定」。
 *
 * 0.0.1 是每次调用起一次进程：简单、不会泄漏、不占常驻内存。
 * 代价是每次都要重新加载模型（实测 medium 在 4060 上约 5.6 秒）。
 * v0.1 会换成常驻服务，那一步的必要性就是这 5.6 秒。
 *
 * 通信协议见 python/transcribe.py 顶部注释：stdin 一行 JSON，stdout 一行 JSON。
 */
import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { load, modelRef, pickPython } from './config.js';
import { EarsError, toWav } from './media.js';

export const PLUGIN_ROOT = dirname(dirname(fileURLToPath(import.meta.url)));
export const SCRIPT = join(PLUGIN_ROOT, 'python', 'transcribe.py');

function childEnv(cfg, env) {
  const next = { ...env, PYTHONIOENCODING: 'utf-8', PYTHONUTF8: '1', KMP_DUPLICATE_LIB_OK: 'TRUE' };
  const extras = (cfg.extraPath ?? []).filter(Boolean);
  if (extras.length) next.PATH = `${extras.join(';')};${env.PATH ?? ''}`;
  return next;
}

function runPython(python, payload, { cfg, env, signal, timeoutMs }) {
  return new Promise((resolvePromise, reject) => {
    let child;
    try {
      child = spawn(python, [SCRIPT], {
        stdio: ['pipe', 'pipe', 'pipe'],
        windowsHide: true,
        env: childEnv(cfg, env),
      });
    } catch (error) {
      reject(new EarsError('Python 进程起不来', String(error?.message ?? error), 'spawn_failed'));
      return;
    }

    const out = [];
    const err = [];
    let settled = false;
    let timer = null;

    const finish = (fn, value) => {
      if (settled) return;
      settled = true;
      if (timer) clearTimeout(timer);
      fn(value);
    };

    timer = setTimeout(() => {
      try { child.kill(); } catch { /* ignore */ }
      finish(reject, new EarsError(
        `转写超时（超过 ${Math.round(timeoutMs / 1000)} 秒）`,
        '长音频会慢；可以在配置里调大 timeoutMs，或把音频切成几段',
        'timeout',
      ));
    }, timeoutMs);
    timer.unref?.();

    if (signal) {
      const onAbort = () => {
        try { child.kill(); } catch { /* ignore */ }
        finish(reject, new EarsError('已取消', '', 'aborted'));
      };
      if (signal.aborted) onAbort();
      else signal.addEventListener('abort', onAbort, { once: true });
    }

    child.stdout.on('data', (b) => out.push(b));
    child.stderr.on('data', (b) => err.push(b));
    child.on('error', (e) => finish(reject, new EarsError(
      `Python 启动失败：${e?.message ?? e}`,
      '配置里的 python 路径不对，或者这个解释器被删了',
      'spawn_failed',
    )));
    child.on('close', (code) => {
      const stdout = Buffer.concat(out).toString('utf8');
      const stderr = Buffer.concat(err).toString('utf8');
      const line = stdout.split('\n').filter((l) => l.trim()).pop();
      let parsed = null;
      if (line) {
        try { parsed = JSON.parse(line); } catch { /* 下面按失败处理 */ }
      }
      if (!parsed) {
        const tail = (stderr || stdout).trim().split('\n').slice(-6).join('\n');
        finish(reject, new EarsError(
          `转写脚本没有返回结果（退出码 ${code}）`,
          tail || '没有输出，也没有报错——大概率是解释器不对',
          'no_output',
        ));
        return;
      }
      if (!parsed.ok) {
        finish(reject, new EarsError(parsed.error ?? '未知错误', parsed.hint ?? '', 'transcribe_failed'));
        return;
      }
      finish(resolvePromise, parsed);
    });

    child.stdin.on('error', () => { /* 子进程早退时写 stdin 会 EPIPE，忽略即可 */ });
    child.stdin.end(`${JSON.stringify(payload)}\n`);
  });
}

/**
 * 转写一个音频文件。
 *
 * @param {string} input 音频路径（wav/mp3/m4a/ogg/amr/flac…）
 * @param {{language?: string, model?: string, device?: string, prompt?: string,
 *          signal?: AbortSignal, env?: object}} [opts]
 */
export async function transcribe(input, opts = {}) {
  const env = opts.env ?? process.env;
  const cfg = load(env);

  const picked = await pickPython(cfg, env);
  if (!picked.python) {
    const tried = picked.tried.map((t) => `  ${t.python} → ${t.error}`).join('\n');
    throw new EarsError(
      '找不到能用的 Python：没有一个解释器能 import faster_whisper',
      `装一份依赖：\n  pip install -r "${join(PLUGIN_ROOT, 'python', 'requirements.txt')}"\n`
      + `或者在 ~/.dsh/voice-ears.json 里把 "python" 指到一个已经装好的解释器。\n试过：\n${tried}`,
      'python_missing',
    );
  }

  const media = await toWav(input, cfg, { env, signal: opts.signal });
  try {
    const payload = {
      audio: media.path,
      model: opts.model || modelRef(cfg),
      language: opts.language || cfg.language || 'auto',
      device: opts.device || cfg.device,
      compute: cfg.compute,
      prompt: opts.prompt || cfg.prompt || '',
      maxSeconds: cfg.maxSeconds,
      thresholds: cfg.thresholds,
    };
    const result = await runPython(picked.python, payload, {
      cfg, env, signal: opts.signal, timeoutMs: cfg.timeoutMs,
    });
    return { ...result, source: media.kind.label, python: picked.python };
  } finally {
    media.cleanup?.();
  }
}

/** 不转写，只报"环境到底行不行"。给 ears_status 用。 */
export async function environment({ env = process.env } = {}) {
  const cfg = load(env);
  const picked = await pickPython(cfg, env);
  const ref = modelRef(cfg);
  const modelLooksLocal = existsSync(ref);
  return {
    ok: Boolean(picked.python),
    python: picked.python,
    pythonDetail: picked.python ? picked.probe : null,
    tried: picked.tried,
    model: ref,
    modelLooksLocal,
    device: cfg.device,
    compute: cfg.compute,
    language: cfg.language,
    ffmpeg: cfg.ffmpeg || process.env.DSH_FFMPEG_PATH || 'PATH 里的 ffmpeg',
    configPath: join(process.env.USERPROFILE ?? process.env.HOME ?? '', '.dsh', 'voice-ears.json'),
    script: SCRIPT,
  };
}
