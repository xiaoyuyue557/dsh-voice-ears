/**
 * worker.js — 常驻耳朵进程的管理器。
 *
 * 0.1 的做法是每条语音起一个 Python 进程，模型每次重载 5.6 秒——转一条等 7 秒多，
 * QQ 聊天的节奏下这个体验过不去。这里改成：一个进程常驻，模型只加载一次，
 * 之后每条只解码（实测 1.7 秒）；闲置超时自己退出，不白占显存。
 *
 * 几条设计取舍：
 *  - 串行队列：whisper 并发解码会抢显存，排队比并行稳。
 *  - 单条失败不杀 worker：模型是花 5.6 秒装好的，不能因为一条坏音频丢掉它。
 *  - 崩溃后下次请求自动重启：用户不需要知道"刚才进程挂了"。
 *  - 模型/设备变了就重启：换模型必须重载，签名不同直接换进程。
 */
import { spawn } from 'node:child_process';
import { join } from 'node:path';
import { load, modelRef, pickPython } from './config.js';
import { EarsError } from './media.js';
import { PLUGIN_ROOT } from './paths.js';

export const WORKER_SCRIPT = join(PLUGIN_ROOT, 'python', 'worker.py');
const STDERR_KEEP = 40;

let current = null;
let seq = 0;

function workerEnv(cfg, env) {
  const next = { ...env, PYTHONIOENCODING: 'utf-8', PYTHONUTF8: '1', KMP_DUPLICATE_LIB_OK: 'TRUE' };
  const extras = (cfg.extraPath ?? []).filter(Boolean);
  if (extras.length) next.PATH = `${extras.join(';')};${env.PATH ?? ''}`;
  return next;
}

class Worker {
  constructor({ python, cfg, env, signature }) {
    this.python = python;
    this.cfg = cfg;
    this.env = env;
    this.signature = signature;
    this.child = null;
    this.ready = false;
    this.device = '';
    this.compute = '';
    this.loadSeconds = 0;
    this.pid = null;
    this.startedAt = 0;
    this.lastUsedAt = 0;
    this.pending = new Map();
    this.queue = Promise.resolve();
    this.stderr = [];
    this.idleTimer = null;
    this.waiters = [];
  }

  async start({ timeoutMs = 180000 } = {}) {
    if (this.ready) return this;
    this.child = spawn(this.python, [WORKER_SCRIPT], {
      stdio: ['pipe', 'pipe', 'pipe'],
      windowsHide: true,
      env: workerEnv(this.cfg, this.env),
    });

    let buffer = '';
    this.child.stdout.on('data', (chunk) => {
      buffer += chunk.toString('utf8');
      let idx;
      while ((idx = buffer.indexOf('\n')) >= 0) {
        const line = buffer.slice(0, idx).trim();
        buffer = buffer.slice(idx + 1);
        if (line) this.#onLine(line);
      }
    });
    this.child.stderr.on('data', (chunk) => {
      this.stderr.push(chunk.toString('utf8'));
      if (this.stderr.length > STDERR_KEEP) this.stderr.shift();
    });
    this.child.on('error', (error) => {
      this.#failAll(new EarsError(`耳朵 worker 启动失败：${error?.message ?? error}`,
        '配置里的 python 路径可能不对', 'worker_spawn_failed'));
      this.#teardown();
    });
    this.child.on('exit', (code, signal) => {
      const wasReady = this.ready;
      this.#failAll(new EarsError(
        `耳朵 worker 退出（code=${code}${signal ? `, signal=${signal}` : ''}）`,
        wasReady ? '下次调用会自动重启' : this.#stderrTail(),
        'worker_exited',
      ));
      this.#teardown();
    });

    this.startedAt = Date.now();
    this.child.stdin.write(`${JSON.stringify({
      type: 'init',
      model: modelRef(this.cfg),
      device: this.cfg.device,
      compute: this.cfg.compute,
    })}\n`);

    await new Promise((resolvePromise, reject) => {
      const timer = setTimeout(() => {
        this.stop('init-timeout');
        reject(new EarsError(
          `耳朵 worker ${Math.round(timeoutMs / 1000)} 秒内没准备好`,
          '首次加载模型可能很慢；也可能是模型文件缺失。用 ears_status 看看。',
          'worker_timeout',
        ));
      }, timeoutMs);
      timer.unref?.();
      this.waiters.push({
        resolve: (value) => { clearTimeout(timer); resolvePromise(value); },
        reject: (error) => { clearTimeout(timer); reject(error); },
      });
    });
    this.#armIdleTimer();
    return this;
  }

  #onLine(line) {
    let msg;
    try { msg = JSON.parse(line); } catch { return; }

    if (msg.type === 'ready') {
      this.ready = true;
      this.device = msg.device;
      this.compute = msg.compute;
      this.loadSeconds = msg.loadSeconds;
      this.pid = msg.pid;
      for (const waiter of this.waiters.splice(0)) waiter.resolve(this);
      return;
    }
    if (msg.type === 'error') {
      for (const waiter of this.waiters.splice(0)) {
        waiter.reject(new EarsError(msg.error ?? 'worker 初始化失败', msg.hint ?? '', 'worker_init_failed'));
      }
      this.stop('init-error');
      return;
    }
    if (msg.type === 'pong' || msg.type === 'bye') return;

    const waiter = this.pending.get(msg.id);
    if (!waiter) return;
    this.pending.delete(msg.id);
    if (msg.ok) waiter.resolve(msg);
    else waiter.reject(new EarsError(msg.error ?? '转写失败', msg.hint ?? '', 'transcribe_failed'));
  }

  #failAll(error) {
    for (const waiter of this.pending.values()) waiter.reject(error);
    this.pending.clear();
    for (const waiter of this.waiters.splice(0)) waiter.reject(error);
  }

  #stderrTail() {
    return this.stderr.join('').trim().split('\n').slice(-5).join('\n') || '没有输出';
  }

  #teardown() {
    this.ready = false;
    if (this.idleTimer) clearTimeout(this.idleTimer);
    this.idleTimer = null;
    this.child = null;
    this.pending.clear();
    if (current === this) current = null;
  }

  #armIdleTimer() {
    if (this.idleTimer) clearTimeout(this.idleTimer);
    const idleMs = Math.max(30, Number(this.cfg.idleSeconds) || 600) * 1000;
    this.idleTimer = setTimeout(() => this.stop('idle'), idleMs);
    this.idleTimer.unref?.();
  }

  /** 串行执行：一次只跑一条，排到的请求等前一条结束。 */
  transcribe(req, { signal, timeoutMs = 300000 } = {}) {
    const run = () => this.#send(req, { signal, timeoutMs });
    const next = this.queue.then(run, run);
    // 队列本身不希望因为一次失败而中断，所以吞掉异常，只把结果/错误传给调用者。
    this.queue = next.then(() => undefined, () => undefined);
    return next;
  }

  #send(req, { signal, timeoutMs }) {
    if (!this.ready || !this.child) {
      return Promise.reject(new EarsError('耳朵 worker 不在运行', '内部状态错误，重试一次', 'worker_stopped'));
    }
    const id = ++seq;
    return new Promise((resolvePromise, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new EarsError(`转写超时（超过 ${Math.round(timeoutMs / 1000)} 秒）`,
          '长音频会慢；可以调大 timeoutMs，或把音频切段', 'timeout'));
      }, timeoutMs);
      timer.unref?.();

      const onAbort = () => {
        clearTimeout(timer);
        this.pending.delete(id);
        reject(new EarsError('已取消', '', 'aborted'));
      };
      if (signal) {
        if (signal.aborted) { onAbort(); return; }
        signal.addEventListener('abort', onAbort, { once: true });
      }

      this.pending.set(id, {
        resolve: (value) => { clearTimeout(timer); resolvePromise(value); },
        reject: (error) => { clearTimeout(timer); reject(error); },
      });
      try {
        this.child.stdin.write(`${JSON.stringify({ id, ...req })}\n`);
      } catch (error) {
        this.pending.delete(id);
        clearTimeout(timer);
        reject(new EarsError(`发给 worker 失败：${error?.message ?? error}`, 'worker 可能刚退出', 'worker_write_failed'));
      }
      this.lastUsedAt = Date.now();
    });
  }

  stop(reason = 'manual') {
    if (!this.child) return;
    try { this.child.stdin.write('{"type":"shutdown"}\n'); } catch { /* ignore */ }
    const child = this.child;
    setTimeout(() => { try { child.kill(); } catch { /* ignore */ } }, 500).unref?.();
    this.#teardown();
    this.lastStopReason = reason;
  }

  /** 被调用过：刷新闲置计时，免得正在用的 worker 被自己踢掉。 */
  touch() {
    this.lastUsedAt = Date.now();
    if (this.ready) this.#armIdleTimer();
  }

  status() {
    const idleMs = this.idleTimer ? Math.max(30, Number(this.cfg.idleSeconds) || 600) * 1000 : 0;
    return {
      running: this.ready,
      device: this.device,
      compute: this.compute,
      model: modelRef(this.cfg),
      loadSeconds: this.loadSeconds,
      pid: this.pid,
      uptimeSeconds: this.startedAt ? Math.round((Date.now() - this.startedAt) / 1000) : 0,
      idleSecondsLeft: idleMs ? Math.max(0, Math.round((idleMs - (Date.now() - this.lastUsedAt)) / 1000)) : null,
      pending: this.pending.size,
    };
  }
}

function signatureOf(cfg) {
  return JSON.stringify([modelRef(cfg), cfg.device, cfg.compute]);
}

/**
 * 拿到一个可用的 worker：没有就起一个；配置签名变了就换一个新的。
 * @returns {Promise<Worker>}
 */
export async function ensureWorker({ cfg = load(), env = process.env, allowSpawn = true } = {}) {
  const signature = signatureOf(cfg);
  if (current && current.ready && current.signature === signature) {
    current.touch();
    return current;
  }
  if (!allowSpawn) return null;
  if (current) current.stop('config-changed');

  const picked = await pickPython(cfg, env);
  if (!picked.python) {
    const tried = picked.tried.map((t) => `  ${t.python} → ${t.error}`).join('\n');
    throw new EarsError(
      '找不到能用的 Python：没有一个解释器能 import faster_whisper',
      `装一份依赖：pip install -r python/requirements.txt，或把配置里的 python 指到装好的解释器。\n试过：\n${tried}`,
      'python_missing',
    );
  }

  const worker = new Worker({ python: picked.python, cfg, env, signature });
  current = worker;
  await worker.start({ timeoutMs: Math.max(180000, cfg.timeoutMs ?? 180000) });
  return worker;
}

export function workerStatus(cfg = load()) {
  if (!current) return { running: false, configured: signatureOf(cfg) };
  return { ...current.status(), stoppedReason: current.lastStopReason };
}

export function stopWorker(reason = 'manual') {
  if (current) current.stop(reason);
  current = null;
}
