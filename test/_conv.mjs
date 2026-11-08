import { readFileSync, writeFileSync } from "node:fs";
import { toWav } from "../src/media.js";
import { load } from "../src/config.js";
const cfg = load();
const lines = readFileSync(process.argv[2], "utf8").replace(/^\uFEFF/, "").split(/\r?\n/).filter(Boolean);
const out = [];
for (const line of lines) {
  const [time, , file] = line.split("\t");
  const media = await toWav(file, cfg);
  out.push({ time, wav: media.path });
  // 保留临时文件（cleanup 不调用），下面 python 一次算完再删
}
writeFileSync(process.argv[3], JSON.stringify(out));
console.log("converted", out.length);
