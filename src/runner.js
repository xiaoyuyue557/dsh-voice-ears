/**
 * runner.js — 转写的两条路。
 *
 *   热路径（默认）：常驻 worker，模型只加载一次，之后每条 ~1.7 秒。
 *   冷路径：一次性进程，模型每次重载（~7.4 秒）。用于 worker 起不来时的兜底，
 *           以及"这次调用要换模型/换设备"的场合——那必须重载，常驻没意义。
 *
 * 两条路的判定完全一样，因为都在 python/ears_core.py 里，不是两套代码。
 */
import { spawn } from 'node:child_process';
import { join } from 'node:path';
import { existsSync } from 'node:fs';
import { load, modelRef, pickPython } from './config.js';
import { EarsError, toWav } from './media.js';
import { PLUGIN_ROOT } from './paths.js';
import { ensureWorker, stopWorker, workerStatus } from './worker.js';

export { PLUGIN_ROOT };
export const SCRIPT = join(PLUGIN_ROOT, 'python', 'transcribe.py');

/** 主动收工：给把本模块当库用的集成方（脚本、框架适配层）一个干净的退出方式。 */
export function stopEars(reason = 'manual') {
  stopWorker(reason);
}

function childEnv(cfg, env) {
  const next = { ...env, PYTHONIOENCODING: 'utf-8', PYTHONUTF8: '1', KMP_DUPLICATE_LIB_OK: 'TRUE' };
  const extras = (cfg.extraPath ?? []).filter(Boolean);
  if (extras.length) next.PATH = `${extras.join(';')};${env.PATH ?? ''}`;
  return next;
}

/** 冷路径：起一个进程，跑完就退。 */
function runOnce(python, payload, { cfg, env, signal, timeoutMs }) {
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

/** worker 起不来时可以回退到冷路径的错误码。单条转写失败（文件坏了等）不该回退。 */
const FALLBACK_CODES = new Set(['worker_spawn_failed', 'worker_timeout', 'worker_init_failed', 'worker_exited']);

/**
 * 转写一个音频文件。
 *
 * @param {string} input 音频路径（wav/mp3/m4a/ogg/amr/flac/silk…）
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
      language: opts.language || cfg.language || 'auto',
      prompt: opts.prompt || cfg.prompt || '',
      maxSeconds: cfg.maxSeconds,
      thresholds: cfg.thresholds,
    };
    const base = { source: media.kind.label, python: picked.python };

    // 换模型/换设备必须重载模型，常驻反而更慢——这种调用直接走冷路径。
    const overridesModel = Boolean(opts.model || opts.device);
    if (cfg.resident !== false && !overridesModel) {
      let fallbackReason = null;
      const wasRunning = workerStatus(cfg).running;
      try {
        const worker = await ensureWorker({ cfg, env });
        const result = await worker.transcribe(payload, { signal: opts.signal, timeoutMs: cfg.timeoutMs });
        // coldStart：这次调用顺手把模型装上了。给渲染层用——否则首条会显示 1.7 秒，
        // 用户以为一直是这么慢，其实那 5.5 秒加载被算在了别处。
        return { ...result, ...base, mode: 'resident', coldStart: !wasRunning };
      } catch (error) {
        if (!FALLBACK_CODES.has(error?.code)) throw error;
        fallbackReason = error;
      }
      const result = await runOnce(picked.python, {
        ...payload,
        model: modelRef(cfg),
        device: cfg.device,
        compute: cfg.compute,
      }, { cfg, env, signal: opts.signal, timeoutMs: cfg.timeoutMs });
      return {
        ...result,
        ...base,
        mode: 'oneshot',
        fallbackReason: fallbackReason ? fallbackReason.message : undefined,
      };
    }

    const result = await runOnce(picked.python, {
      ...payload,
      model: opts.model || modelRef(cfg),
      device: opts.device || cfg.device,
      compute: cfg.compute,
    }, { cfg, env, signal: opts.signal, timeoutMs: cfg.timeoutMs });
    return { ...result, ...base, mode: 'oneshot' };
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
    resident: cfg.resident !== false,
    idleSeconds: cfg.idleSeconds,
    worker: workerStatus(cfg),
    ffmpeg: cfg.ffmpeg || process.env.DSH_FFMPEG_PATH || 'PATH 里的 ffmpeg',
    configPath: join(process.env.USERPROFILE ?? process.env.HOME ?? '', '.dsh', 'voice-ears.json'),
    script: SCRIPT,
  };
}
