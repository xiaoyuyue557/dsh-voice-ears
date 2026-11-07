/**
 * smoke.mjs — 端到端自检：真音频进，判定出，逐条对答案。
 *
 * 为什么要有这个文件：这个插件最容易坏的地方不是代码写错，而是**判定变松**——
 * 某天调了个阈值，静音又被编成一句话，而单看输出很像正常。所以把这几条固定下来，
 * 每次改判定就 `npm test`，错了立刻知道。
 *
 * 跑法：npm test（需要 ~/.dsh/voice-ears.json 里的 python / modelDir 可用）
 * 覆盖：真语音要听清（wav / mp3 / silk 三条路）/ 静音要拒 / 白噪要拒 / 粉噪要拒
 *
 * fixtures 里的音频是自己合成的（Windows SAPI 中文语音），没有第三方版权内容。
 */
import { existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { transcribe } from '../src/runner.js';

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
  try {
    const result = await transcribe(path);
    const verdictOk = testCase.expect.includes(result.verdict);
    const text = result.text || result.partialText || '';
    const textOk = !testCase.contains || text.includes(testCase.contains);
    const ok = verdictOk && textOk;
    if (!ok) failed += 1;
    console.log(
      `${ok ? '✓' : '✗'} ${testCase.file.padEnd(12)} ${result.verdict.padEnd(10)} `
      + `${String(result.elapsed ?? '?').padStart(5)}s  ${text.slice(0, 34)}`,
    );
    if (!verdictOk) console.log(`    期望 ${testCase.expect.join('/')}，实际 ${result.verdict}：${result.reason}`);
    if (!textOk) console.log(`    期望文字里含「${testCase.contains}」，实际「${text}」`);
  } catch (error) {
    failed += 1;
    console.log(`✗ ${testCase.file.padEnd(12)} 抛错：${error.message}${error.hint ? `\n    ${error.hint}` : ''}`);
  }
}

console.log(failed === 0 ? '\n全部通过' : `\n${failed} 条不符合预期`);
process.exit(failed === 0 ? 0 : 1);
