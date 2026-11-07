/**
 * tools.js — 两个工具的声明。
 *
 * 描述会进系统提示，每次请求都要付 token，所以只写「什么时候用、结果怎么读」，
 * 实现细节留在 README 里。这里最要紧的一段是 verdict 的读法：
 * 它决定了 agent 会不会把一段猜出来的话当成用户真的说过的话。
 */
import { transcribe, environment } from './runner.js';
import { basename } from 'node:path';

const OUTPUT = {
  schema: {
    type: 'object',
    additionalProperties: true,
    properties: {
      verdict: {
        type: 'string',
        enum: ['heard', 'unclear', 'no_speech', 'silent', 'empty'],
        description: 'heard = 可以用;其它都表示没有可信文字。',
      },
      text: { type: 'string', description: '只在 verdict=heard 时有内容。' },
      partialText: { type: 'string', description: 'verdict=unclear 时的猜测，不可当事实引用。' },
      reason: { type: 'string', description: '人话解释。' },
      language: { type: 'string' },
      language_probability: { type: 'number' },
      avg_logprob: { type: 'number' },
      no_speech_prob: { type: 'number' },
      duration: { type: 'number' },
      elapsed: { type: 'number' },
      device: { type: 'string' },
    },
  },
  render(_args, value) {
    const v = value?.verdict;
    if (!v) return [{ type: 'text', text: value?.text ?? '(没有结果)' }];
    const lines = [];
    if (v === 'heard') {
      lines.push(`听清了：${value.text}`);
    } else if (v === 'unclear') {
      lines.push(`没听清：${value.reason}`);
      if (value.partialText) lines.push(`（这只是猜的，别当原话引用：${value.partialText}）`);
    } else {
      lines.push(`这不是人在说话（${v}）：${value.reason}`);
    }
    const bits = [];
    if (value.language) bits.push(value.language);
    if (value.device) bits.push(`${value.device}/${value.compute ?? '?'}`);
    if (value.duration) bits.push(`音频 ${value.duration}s`);
    if (value.elapsed) bits.push(`耗时 ${value.elapsed}s`);
    lines.push(`（${bits.join(' · ')}）`);
    return [{ type: 'text', text: lines.join('\n') }];
  },
};

const INPUT_DESC = 'Path to the audio file: .wav/.mp3/.m4a/.ogg/.amr/.flac. Absolute path, or relative to '
  + 'the working directory. QQ/WeChat voice notes usually land as .amr or .silk — SILK is not decoded yet, '
  + 'so convert it to wav first (OneBot: get_record with out_format=wav).';

export function transcribeSpec() {
  return {
    name: 'ears_transcribe',
    description: [
      'Hear a voice message: turn an audio file into text, locally and offline, and say how sure it is.',
      '',
      'Built for audio other people send you (QQ/WeChat voice notes, recordings), not for dictating',
      'your own speech. Costs no tokens for the audio itself; the first call on a machine may take',
      'several seconds because the model has to load.',
      '',
      'ALWAYS read the verdict before using the text:',
      '  heard      text is what the speaker said — quote it.',
      '  unclear    audio was garbled; text is empty and partialText is a guess. Do NOT present',
      '             partialText as the speaker\'s words; say you did not catch it and ask again.',
      '  no_speech  noise/music/silence that the model hallucinated words on; text is empty.',
      '  silent     almost no sound at all in the recording.',
      '  empty      nothing transcribed (music, empty track, or too short).',
      '',
      'Why the verdict exists: whisper always produces a fluent sentence, even on pure noise, so the',
      'text alone cannot tell you whether anyone actually spoke. This tool checks the audio itself',
      '(spectral flatness), loop hallucination and word confidence before it hands you text.',
    ].join('\n'),
    parameters: {
      type: 'object',
      properties: {
        input: { type: 'string', description: INPUT_DESC },
        language: { type: 'string', description: 'Language code such as zh, en, ja. Default from config (zh); use auto to detect.' },
        model: { type: 'string', description: 'Override the whisper model for this call (name or CTranslate2 model directory). Default: config, usually medium.' },
        device: { type: 'string', enum: ['auto', 'cuda', 'cpu'], description: 'auto falls back to cpu when CUDA is unusable.' },
        prompt: { type: 'string', description: 'Bias text: names, nicknames, jargon that appears in the audio.' },
      },
      required: ['input'],
      additionalProperties: false,
    },
    output: OUTPUT,
    timeoutMs: 10 * 60 * 1000,
    isConcurrencySafe: () => false,
    presentCall: (args) => ({ card: 'generic', title: 'listen to a voice message', kind: 'read', rawInput: args }),
    async execute(args, exec) {
      if (!args?.input) throw new Error('input is required');
      const result = await transcribe(args.input, {
        language: args.language,
        model: args.model,
        device: args.device,
        prompt: args.prompt,
        signal: exec?.signal,
      });
      return {
        verdict: result.verdict,
        text: result.text ?? '',
        partialText: result.partialText ?? '',
        reason: result.reason ?? '',
        language: result.language,
        language_probability: result.language_probability,
        avg_logprob: result.avg_logprob,
        no_speech_prob: result.no_speech_prob,
        duration: result.duration,
        elapsed: result.elapsed,
        device: result.device,
        compute: result.compute,
        model: result.model,
        source: result.source,
        segments: result.segments,
      };
    },
  };
}

export function statusSpec() {
  return {
    name: 'ears_status',
    description: [
      'Check whether the local ears can actually run: which Python has faster-whisper, where the model',
      'is, whether ffmpeg is reachable, and where the config lives.',
      '',
      'Use it when a transcription failed with a setup error, or when the user asks whether the ears',
      'are working / why a voice message came back empty.',
    ].join('\n'),
    parameters: { type: 'object', properties: {}, additionalProperties: false },
    output: {
      schema: { type: 'object', additionalProperties: true, properties: { ok: { type: 'boolean' } } },
      render(_args, value) {
        const lines = [
          value.ok ? '耳朵可用 ✓' : '耳朵不可用 ✗',
          `解释器：${value.python ?? '（没找到）'}`,
        ];
        if (value.pythonDetail?.ok) {
          lines.push(`  python ${value.pythonDetail.python} · faster-whisper ${value.pythonDetail.faster_whisper} · ctranslate2 ${value.pythonDetail.ctranslate2}`);
        }
        lines.push(`模型：${value.model}${value.modelLooksLocal ? '（本地目录）' : '（首次调用会下载）'}`);
        lines.push(`设备：${value.device} / ${value.compute}`);
        lines.push(`ffmpeg：${value.ffmpeg}`);
        lines.push(`配置：${value.configPath}`);
        if (!value.ok && value.tried?.length) {
          lines.push('试过的解释器：');
          for (const t of value.tried) lines.push(`  ${t.python} → ${t.error}`);
        }
        return [{ type: 'text', text: lines.join('\n') }];
      },
    },
    timeoutMs: 60 * 1000,
    isConcurrencySafe: () => true,
    presentCall: (args) => ({ card: 'generic', title: 'check ears', kind: 'read', rawInput: args }),
    async execute() {
      const env = await environment();
      return { ...env, script: basename(env.script) };
    },
  };
}
