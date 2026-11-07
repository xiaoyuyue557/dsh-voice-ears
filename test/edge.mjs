/**
 * edge.mjs — 报错路径自检：坏输入要给出「能照着做」的提示，而不是 stack trace。
 *
 * 这几条是用户最容易撞上的：silk 语音条、路径写错、把非音频文件丢进来。
 * 每条都断言 error.code 和 hint 非空——提示为空等于没提示。
 */
import { existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { transcribe } from '../src/runner.js';

const HERE = dirname(fileURLToPath(import.meta.url));
const FIXTURES = join(HERE, 'fixtures');

const CASES = [
  { label: 'SILK 内容坏了', file: join(FIXTURES, 'fake.silk'), code: 'silk_decode_failed' },
  { label: '文件不存在', file: join(FIXTURES, 'nope.wav'), code: 'not_found' },
  { label: '非音频文件', file: join(HERE, '..', 'package.json'), code: 'transcode_failed' },
];

let failed = 0;
for (const testCase of CASES) {
  if (!existsSync(testCase.file) && testCase.code !== 'not_found') {
    console.log(`跳过 ${testCase.label}（fixture 不存在）`);
    continue;
  }
  try {
    const result = await transcribe(testCase.file);
    failed += 1;
    console.log(`✗ ${testCase.label}：本该报 ${testCase.code}，却返回了 ${result.verdict}`);
  } catch (error) {
    const codeOk = error.code === testCase.code;
    const hintOk = Boolean(error.hint);
    if (!codeOk || !hintOk) failed += 1;
    console.log(`${codeOk && hintOk ? '✓' : '✗'} ${testCase.label.padEnd(12)} [${error.code}] ${String(error.message).split('\n')[0]}`);
    if (!hintOk) console.log('    缺少 hint（用户看到报错却不知道怎么办）');
  }
}

console.log(failed === 0 ? '\n全部通过' : `\n${failed} 条不符合预期`);
process.exit(failed === 0 ? 0 : 1);
