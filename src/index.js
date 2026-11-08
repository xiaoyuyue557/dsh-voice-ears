/**
 * index.js — dsh-voice-ears 插件入口（0.0.1）。
 *
 * 0.0.1 只做一件事并且做到底：**把一段音频变成一句可信的话**。
 * 没有常驻服务、没有设置页、没有依赖自举——那些是 v0.1/v0.2 的事。
 * 这一版要验证的是判定逻辑站不站得住，而不是功能有多少。
 *
 * 出错时不抛异常给上层，而是把「为什么 + 怎么办」当文本交给 agent：
 * 用户看到的应该是"缺 CUDA 运行库，装这个或改成 cpu"，而不是一段 stack trace。
 */
import { transcribeSpec, statusSpec } from './tools.js';
import { stopWorker } from './worker.js';

export const name = 'dsh-voice-ears';
export const inject = ['tools'];

export async function apply(ctx) {
  const disposers = [];

  const wrap = (spec) => {
    const inner = spec.execute;
    spec.execute = async (args, exec) => {
      try {
        return await inner(args, exec);
      } catch (error) {
        const hint = error?.hint ? `\n建议：${error.hint}` : '';
        return { text: `耳朵出错了：${error?.message ?? error}${hint}` };
      }
    };
    return spec;
  };

  const register = (spec) => {
    try {
      disposers.push(ctx.tools.register(wrap(spec)));
    } catch (error) {
      console.error(`[voice-ears] 注册 ${spec.name} 失败：${error?.message ?? error}`);
    }
  };

  register(transcribeSpec());
  register(statusSpec());

  return () => {
    // 插件被卸载/重载时把这双耳朵收起来：常驻的 Python worker 占着显存，
    // 不主动停就会被留成孤儿进程（用户看不到、还以为耳机还在听）。
    try { stopWorker('plugin-dispose'); } catch { /* ignore */ }
    for (const dispose of disposers.reverse()) {
      try { dispose?.(); } catch { /* ignore */ }
    }
  };
}
