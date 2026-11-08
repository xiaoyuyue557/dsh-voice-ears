/** paths.js — 插件根目录。单独放一个文件，免得 worker.js 和 runner.js 互相 import 成环。 */
import { dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

export const PLUGIN_ROOT = dirname(dirname(fileURLToPath(import.meta.url)));
