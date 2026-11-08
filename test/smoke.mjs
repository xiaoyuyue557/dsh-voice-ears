/**
 * smoke.mjs — 端到端自检：真音频进，判定出，逐条对答案。
 *
 * 为什么要有这个文件：这个插件最容易坏的地方不是代码写错，而是**判定变松**——
 * 某天调了个阈值，静音又被编成一句话，而单看输出很像正常。所以把这几条固定下来，
 * 每次改判定就 `npm test`，错了立刻知道。
 *
 * 跑法：npm test（需要 ~/.dsh/voice-ears.json 里的 python / modelDir 可用）
 * 覆盖：真语音要听清（wav / mp3 / silk 三条路）/ 静音要拒 / 白噪要拒 / 粉噪要拒
 *      最后一条额外测"常驻是否真的省时间"（热路径应明显快于冷路径）。
 *
 * fixtures 里的音频都是自己合成的（Windows SAPI 中文语音 + ffmpeg 生成的噪声），没有第三方版权内容。
 */
import { existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { transcribe } from '../src/runner.js';
import { stopWorker, workerStatus } from '../src/worker.js';

const HERE = dirname(fileURLToPath(import.meta.url));
const FIXTURES = join(HERE, 'fixtures');

const CASES = [
  { file: 'speech.wav', expect: ['heard'], contains: '测试语音', note: '正常语音必须听清' },
  { file: 'speech.mp3', expect: ['heard'], contains: '测试语音', note: 'mp3 走 ffmpeg 转码后仍要听清' },
  { file: 'speech.silk', expect: ['heard'], contains: '测试语音', note: 'QQ 语音条格式（SILK）要能直接吃' },
  { file: 'silence.wav', expect: ['no_speech', 'silent', 'empty'], note: '静音不许编话' },
  { file: 'noise.wav', expect: ['no_speech', 'empty'], note: '白噪不许编话' },
  { file: 'pink.wav', expect: ['no_speech', 'empty'], note: '粉噪不许编话' },
  { file: 'tone.wav', expect: ['no_speech', 'empty', 'unclear'], note: '纯音不许当成话' },
];

let failed = 0;

for (const testCase of CASES) {
  const path = join(FIXTURES, testCase.file);
  if (!existsSync(path)) {
    console.log(`跳过 ${testCase.file}（fixture 不存在）`);
    continue;
  }
  const started = Date.now();
  try {
    const result = await transcribe(path);
    const wall = (Date.now() - started) / 1000;
    const verdictOk = testCase.expect.includes(result.verdict);
    const text = result.text || result.partialText || '';
    const textOk = !testCase.contains || text.includes(testCase.contains);
    const ok = verdictOk && textOk;
    if (!ok) failed += 1;
    console.log(
      `${ok ? '✓' : '✗'} ${testCase.file.padEnd(12)} ${result.verdict.padEnd(10)} `
      + `${String(result.mode ?? '-').padEnd(9)} ${wall.toFixed(2).padStart(5)}s  ${text.slice(0, 30)}`,
    );
    if (!verdictOk) console.log(`    期望 ${testCase.expect.join('/')}，实际 ${result.verdict}：${result.reason}`);
    if (!textOk) console.log(`    期望文字里含「${testCase.contains}」，实际「${text}」`);
  } catch (error) {
    failed += 1;
    console.log(`✗ ${testCase.file.padEnd(12)} 抛错：${error.message}${error.hint ? `\n    ${error.hint}` : ''}`);
  }
}

// 常驻到底有没有省时间：同一段音频再跑一次，热路径应该明显快。
try {
  const started = Date.now();
  const warm = await transcribe(join(FIXTURES, 'speech.wav'));
  const warmSeconds = (Date.now() - started) / 1000;
  const status = workerStatus();
  console.log(
    `\n热路径：${warm.verdict} ${warmSeconds.toFixed(2)}s（decode ${warm.decodeSeconds}s）`
    + ` · worker ${status.running ? '常驻中' : '未常驻'} · 模型加载 ${warm.loadSeconds}s`,
  );
  if (warm.mode === 'resident' && warmSeconds > 6) {
    failed += 1;
    console.log('✗ 常驻模式下热路径仍超过 6 秒，检查 worker 是否真的复用（可能每次都在重载模型）');
  }
} catch (error) {
  failed += 1;
  console.log(`✗ 热路径测试失败：${error.message}`);
} finally {
  stopWorker('smoke-done');
}

console.log(failed === 0 ? '\n全部通过' : `\n${failed} 条不符合预期`);
process.exit(failed === 0 ? 0 : 1);
