#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";
import {fileURLToPath} from "node:url";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const reportRoot = path.join(repoRoot, "reports/0802");
const rendered = new Map([
  ["validation-summary", ""],
  ["coordinate-grid", ""],
]);
const document = {
  getElementById(id) {
    if (!rendered.has(id)) throw new Error(`unexpected report element: ${id}`);
    return {
      insertAdjacentHTML(_position, html) {
        rendered.set(id, rendered.get(id) + html);
      },
    };
  },
};
const context = vm.createContext({window: {}, document, console});

function run(relativePath) {
  const filename = path.join(reportRoot, relativePath);
  vm.runInContext(fs.readFileSync(filename, "utf8"), context, {filename});
}

run("data/wsd_data_loader_1b.js");
context.window.ICSL_HISTORICAL_1B_DATA = context.window.ICSL_DATA_LOADER_DATA;
run("data/wsd_batch_size_474m.js");
context.window.ICSL_HISTORICAL_474M_DATA = context.window.ICSL_REPORT_DATA;
run("data/wsd_batch_size_153m.js");
context.window.ICSL_HISTORICAL_153M_DATA = context.window.ICSL_REPORT_DATA;
run("data/wsd_checkpoint_producer_grid.js");
run("checkpoint_producer_grid_report.js");

const current = context.window.ICSL_CHECKPOINT_PRODUCER_GRID;
const expected = new Map();
for (const evaluator of current.smallEvaluators || []) {
  const result = evaluator.postDecayResult;
  if (evaluator.status !== "complete" || result?.status !== "complete") continue;
  const value = Number(result.validationExact ?? result.validation);
  if (!Number.isFinite(value)) continue;
  const key = `${evaluator.model}:${evaluator.batchSequences}:${evaluator.epoch}`;
  expected.set(key, Math.min(expected.get(key) ?? Infinity, value));
}

const columnIndex = new Map([
  ["474m:128", 6],
  ["474m:256", 7],
  ["153m:128", 10],
  ["153m:256", 11],
]);
const rows = new Map();
for (const match of rendered.get("validation-summary").matchAll(/<tr><td>E([0-9,]+)<\/td>(.*?)<\/tr>/g)) {
  const cells = [...match[2].matchAll(/<td[^>]*>(.*?)<\/td>/g)].map((cell) =>
    cell[1].replace(/<[^>]+>/g, "").trim(),
  );
  rows.set(Number(match[1].replaceAll(",", "")), cells);
}

for (const [key, value] of expected) {
  const [model, batchText, epochText] = key.split(":");
  const index = columnIndex.get(`${model}:${batchText}`);
  if (index === undefined) throw new Error(`no summary column for ${key}`);
  const actual = rows.get(Number(epochText))?.[index];
  const wanted = value.toFixed(3);
  if (actual !== wanted) {
    throw new Error(`rendered ${key} is ${actual ?? "missing"}; expected ${wanted}`);
  }
}

const grid = rendered.get("coordinate-grid");
for (const evaluator of current.smallEvaluators || []) {
  if (evaluator.status !== "complete") continue;
  const producer = current.producers.find((item) => item.id === evaluator.producerId);
  if (!producer) throw new Error(`missing producer for ${evaluator.id}`);
  const label = `${producer.model === "474m" ? "474M" : "153M"} · Pool-3B · BS${producer.batchSequences}`;
  const rowPattern = new RegExp(
    `<tr><td>E${evaluator.epoch}<\\/td><td>${label.replace(/[.*+?^${}()|[\\]\\]/g, "\\$&")}<\\/td>(.*?)<\\/tr>`,
  );
  const row = grid.match(rowPattern)?.[1];
  if (!row || !row.includes(`WD ${producer.weightDecay}) · PD retained · POST complete`)) {
    throw new Error(`coordinate grid does not show completed POST for ${evaluator.id}`);
  }
}

console.log(`validated rendered checkpoint report with ${expected.size} small-model summary cells`);
