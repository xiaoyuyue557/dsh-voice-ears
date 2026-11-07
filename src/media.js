/**
 * media.js — 进来的东西到底是什么格式，以及怎么把它变成 whisper 能吃的 16k 单声道 wav。
 *
 * 为什么按魔数认而不是按扩展名：QQ/微信落盘的语音文件名经常是随机的（`4f2a...amr`），
 * 扩展名和内容对不上的情况很常见；而魔数不会骗人。
 */
import { spawn } from 'node:child_process';
import { closeSync, existsSync, mkdtempSync, openSync, readFileSync, readSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';

export class EarsError extends Error {
  constructor(message, hint = '', code = 'ears_error') {
    super(message);
    this.name = 'EarsError';
    this.hint = hint;
    this.code = code;
  }
}

const SILK_MAGIC = [
  Buffer.from('#!SILK_V3'),
  Buffer.from('\x02#!SILK'),
];

/** 只看前 16 字节。返回 {kind, label}。 */
export function sniff(file) {
  let head;
  try {
    // 只读 16 字节：进来的文件可能是几十 MB 的整段录音，别为了认格式整个读进内存。
    const fd = openSync(file, 'r');
    try {
      const buf = Buffer.alloc(16);
      const read = readSync(fd, buf, 0, 16, 0);
      head = buf.subarray(0, read);
    } finally {
      closeSync(fd);
    }
  } catch (error) {
    throw new EarsError(`读不到文件：${file}`, String(error?.message ?? error), 'unreadable');
  }
  if (!head || head.length === 0) {
    throw new EarsError(`文件是空的：${file}`, '确认路径指向的是一段真实录音', 'empty_file');
  }
  const ascii = head.toString('latin1');
  if (ascii.startsWith('RIFF') && ascii.slice(8, 12) === 'WAVE') return { kind: 'wav', label: 'wav' };
  if (ascii.startsWith('ID3') || (head[0] === 0xff && (head[1] & 0xe0) === 0xe0)) return { kind: 'mp3', label: 'mp3' };
  if (ascii.startsWith('OggS')) return { kind: 'ogg', label: 'ogg/opus' };
  if (ascii.slice(4, 8) === 'ftyp') return { kind: 'mp4', label: 'm4a/mp4' };
  if (ascii.startsWith('#!AMR')) return { kind: 'amr', label: 'amr' };
  if (ascii.startsWith('fLaC')) return { kind: 'flac', label: 'flac' };
  if (head[0] === 0x1a && head[1] === 0x45 && head[2] === 0xdf && head[3] === 0xa3) return { kind: 'webm', label: 'webm/mkv' };
  for (const magic of SILK_MAGIC) {
    if (head.subarray(0, magic.length).equals(magic)) return { kind: 'silk', label: 'silk' };
  }
  return { kind: 'unknown', label: '未知格式' };
}

export function resolveFfmpeg(cfg, env = process.env) {
  const candidates = [cfg.ffmpeg, env.DSH_FFMPEG_PATH, env.FFMPEG_PATH].filter(Boolean);
  for (const candidate of candidates) {
    if (existsSync(candidate)) return { path: candidate, source: 'config' };
  }
  return { path: process.platform === 'win32' ? 'ffmpeg.exe' : 'ffmpeg', source: 'PATH' };
}

const FILTER = 'highpass=f=80,dynaudnorm=f=150:g=15';

/** SILK 解码出来的裸 PCM 包一层 wav 头。 */
function pcmToWav(pcm, sampleRate = 24000, channels = 1, bitsPerSample = 16) {
  const header = Buffer.alloc(44);
  header.write('RIFF', 0, 'ascii');
  header.writeUInt32LE(36 + pcm.length, 4);
  header.write('WAVE', 8, 'ascii');
  header.write('fmt ', 12, 'ascii');
  header.writeUInt32LE(16, 16);
  header.writeUInt16LE(1, 20);
  header.writeUInt16LE(channels, 22);
  header.writeUInt32LE(sampleRate, 24);
  header.writeUInt32LE((sampleRate * channels * bitsPerSample) / 8, 28);
  header.writeUInt16LE((channels * bitsPerSample) / 8, 32);
  header.writeUInt16LE(bitsPerSample, 34);
  header.write('data', 36, 'ascii');
  header.writeUInt32LE(pcm.length, 40);
  return Buffer.concat([header, pcm]);
}

/**
 * QQ / 微信语音条解码。
 *
 * 用 silk-wasm（MIT，NapCat / LLOneBot 同一个解码器）：纯 WASM，不编译、不带 GPL，
 * 也不需要在用户机器上装 VS 构建工具。
 *
 * 为什么必须做这一步：**QQ 存下来的文件扩展名是 `.amr`，内容是 SILK**。
 * 按扩展名判断会把它当 amr 交给 ffmpeg，然后 ffmpeg 报一句看不懂的错。
 * 所以入站格式一律按魔数认（见 sniff），认出来是 silk 就在这里解。
 */
async function decodeSilk(file) {
  let api;
  try {
    const mod = await import('silk-wasm');
    api = mod?.decode ? mod : mod?.default;
  } catch (error) {
    throw new EarsError(
      'SILK 解码器没装（silk-wasm）',
      '这是一条 QQ/微信语音条，需要 silk-wasm 才能解。正常安装（npm）会自动带上；'
      + '如果是从源码链接安装的，在插件目录跑一次 `npm install`。',
      'silk_decoder_missing',
    );
  }
  if (!api?.decode) {
    throw new EarsError('silk-wasm 里没有 decode 方法', '版本不兼容，试试 `npm i silk-wasm@^3`', 'silk_decoder_broken');
  }

  let decoded;
  try {
    const buf = readFileSync(file);   // 语音条只有几 KB，整个读没问题
    decoded = await api.decode(buf, 24000);
  } catch (error) {
    throw new EarsError(
      `SILK 解码失败：${error?.message ?? error}`,
      '文件可能不完整（例如只写了一半），或者不是真的 SILK',
      'silk_decode_failed',
    );
  }

  const pcm = Buffer.from(decoded.data);
  if (!pcm.length) {
    throw new EarsError('SILK 解码出来是空的', '这条语音可能是 0 秒或者损坏了', 'silk_empty');
  }
  const dir = mkdtempSync(join(tmpdir(), 'dsh-ears-silk-'));
  const wavPath = join(dir, 'silk.wav');
  writeFileSync(wavPath, pcmToWav(pcm, 24000));
  return {
    wavPath,
    durationMs: decoded.duration,
    cleanup: () => { try { rmSync(dir, { recursive: true, force: true }); } catch { /* ignore */ } },
  };
}

/**
 * 把任意音频变成 16k 单声道 wav。
 *
 * 两个细节都是量出来的，不是抄的：
 *  - highpass=f=80   去掉低频轰隆（风扇、桌面震动、空调）
 *  - dynaudnorm      把忽大忽小的手机录音拉平；QQ 语音条音量差异极大，这一步能明显救回糊掉的句子
 *  - 16k 单声道       whisper 原生采样率，省一半解码时间，不损识别
 *
 * wav 直接原样用（很多框架给的就是 wav），跳过 ffmpeg——少一个依赖点就少一类故障。
 */
export async function toWav(input, cfg, { env = process.env, signal } = {}) {
  const absolute = resolve(input);
  if (!existsSync(absolute)) {
    throw new EarsError(`文件不存在：${absolute}`, '相对路径按 DSH 工作目录解析；建议直接传绝对路径', 'not_found');
  }
  const kind = sniff(absolute);
  if (kind.kind === 'wav') return { path: absolute, cleanup: null, kind };

  // SILK 先解成 wav，再走下面同一条 ffmpeg 归一化链——这样重采样/滤波/音量拉平只维护一份。
  let source = absolute;
  let silkCleanup = null;
  if (kind.kind === 'silk') {
    const decoded = await decodeSilk(absolute);
    source = decoded.wavPath;
    silkCleanup = decoded.cleanup;
    kind.decodedFrom = 'silk';
    kind.silkDurationMs = decoded.durationMs;
  }

  const ffmpeg = resolveFfmpeg(cfg, env);
  const dir = mkdtempSync(join(tmpdir(), 'dsh-ears-'));
  const out = join(dir, 'audio16k.wav');
  const args = ['-hide_banner', '-loglevel', 'error', '-y', '-i', source, '-af', FILTER, '-ar', '16000', '-ac', '1', out];

  try {
    await runFfmpeg(ffmpeg, args, out, { kind, env, signal });
  } catch (error) {
    silkCleanup?.();
    throw error;
  }

  return {
    path: out,
    kind,
    cleanup: () => {
      try { rmSync(dir, { recursive: true, force: true }); } catch { /* ignore */ }
      silkCleanup?.();
    },
  };
}

function runFfmpeg(ffmpeg, args, out, { kind, env, signal }) {
  return new Promise((resolvePromise, reject) => {
    let child;
    try {
      child = spawn(ffmpeg.path, args, { stdio: ['ignore', 'ignore', 'pipe'], windowsHide: true, env });
    } catch (error) {
      reject(new EarsError('ffmpeg 启不来', String(error?.message ?? error), 'ffmpeg_missing'));
      return;
    }
    const err = [];
    child.stderr.on('data', (b) => err.push(b));
    child.on('error', (e) => {
      const detail = String(e?.message ?? e);
      reject(new EarsError(
        'ffmpeg 没找到',
        `装一个 ffmpeg 并放进 PATH，或在配置里填 ffmpeg 的绝对路径。原始错误：${detail}`,
        'ffmpeg_missing',
      ));
    });
    if (signal) signal.addEventListener('abort', () => { try { child.kill(); } catch { /* ignore */ } }, { once: true });
    child.on('close', (code) => {
      if (code === 0 && existsSync(out)) { resolvePromise(); return; }
      const tail = Buffer.concat(err).toString('utf8').trim().split('\n').slice(-4).join('\n');
      reject(new EarsError(
        `ffmpeg 转码失败（退出码 ${code}，${kind.label}）`,
        tail || '文件可能损坏，或者不是真的音频',
        'transcode_failed',
      ));
    });
  });
}
