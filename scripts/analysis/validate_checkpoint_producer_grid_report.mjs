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
  const key = `${evaluator.model}:dclm3b:${evaluator.batchSequences}:${evaluator.epoch}`;
  expected.set(key, Math.min(expected.get(key) ?? Infinity, value));
}
for (const run of current.dclm333mIntegratedRuns || []) {
  for (const [epochText, result] of Object.entries(run.postDecayResults || {})) {
    if (result?.status !== "complete") continue;
    const value = Number(result.validationExact ?? result.validation);
    if (!Number.isFinite(value)) continue;
    const key = `${run.model}:dclm333m:${run.batchSequences}:${epochText}`;
    expected.set(key, Math.min(expected.get(key) ?? Infinity, value));
  }
}

const columnIndex = new Map([
  ["1b:dclm333m:32", 0],
  ["1b:dclm333m:64", 1],
  ["474m:dclm333m:32", 6],
  ["474m:dclm333m:64", 7],
  ["474m:dclm333m:128", 8],
  ["474m:dclm3b:128", 11],
  ["474m:dclm3b:256", 12],
  ["474m:dclm3b:512", 13],
  ["153m:dclm333m:32", 14],
  ["153m:dclm333m:64", 15],
  ["153m:dclm333m:128", 16],
  ["153m:dclm3b:128", 19],
  ["153m:dclm3b:256", 20],
  ["153m:dclm3b:512", 21],
]);
const rows = new Map();
for (const match of rendered.get("validation-summary").matchAll(/<tr><td>E([0-9,]+)<\/td>(.*?)<\/tr>/g)) {
  const cells = [...match[2].matchAll(/<td[^>]*>(.*?)<\/td>/g)].map((cell) =>
    cell[1].replace(/<[^>]+>/g, "").trim(),
  );
  rows.set(Number(match[1].replaceAll(",", "")), cells);
}

for (const [key, value] of expected) {
  const [model, pool, batchText, epochText] = key.split(":");
  const index = columnIndex.get(`${model}:${pool}:${batchText}`);
  if (index === undefined) throw new Error(`no summary column for ${key}`);
  const actual = rows.get(Number(epochText))?.[index];
  const wanted = value.toFixed(3);
  if (actual !== wanted) {
    throw new Error(`rendered ${key} is ${actual ?? "missing"}; expected ${wanted}`);
  }
}

const grid = rendered.get("coordinate-grid");
if (grid.includes("planned")) {
  throw new Error("coordinate grid must not render planned placeholders");
}
for (const producer of current.producers || []) {
  if (!["submitted", "scheduled", "running"].includes(producer.status)) continue;
  const label = `${producer.model === "1b" ? "1B" : producer.model === "474m" ? "474M" : "153M"} · ${producer.pool === "dclm3b" ? "Pool-3B" : producer.pool} · BS${producer.batchSequences}`;
  const rowPattern = new RegExp(
    `<tr><td>E${producer.currentEpoch}<\\/td><td>${label.replace(/[.*+?^${}()|[\\]\\]/g, "\\$&")}<\\/td>(.*?)<\\/tr>`,
  );
  const row = grid.match(rowPattern)?.[1];
  const expectedState = producer.status === "running" ? "producer running" : "producer queued";
  if (!row || !row.includes(`WD ${producer.weightDecay}) · ${expectedState}`)) {
    throw new Error(`coordinate grid does not show active producer ${producer.id}`);
  }
}
for (const evaluator of current.evaluators || []) {
  const producer = current.producers.find((item) => item.id === evaluator.producerId);
  if (!producer) throw new Error(`missing producer for ${evaluator.id}`);
  for (const additional of evaluator.additionalExperiments || []) {
    if (!["submitted", "scheduled", "running"].includes(additional.status)) continue;
    const label = `1B · Pool-3B · BS${producer.batchSequences}`;
    const rowPattern = new RegExp(
      `<tr><td>E${additional.epoch}<\\/td><td>${label.replace(/[.*+?^${}()|[\\]\\]/g, "\\$&")}<\\/td>(.*?)<\\/tr>`,
    );
    const row = grid.match(rowPattern)?.[1];
    const expectedState = additional.status === "running" ? "POST running" : "POST queued";
    if (!row || !row.includes(`WD ${producer.weightDecay}) · PD retained · ${expectedState}`)) {
      throw new Error(`coordinate grid does not show active standalone POST for ${evaluator.id} E${additional.epoch}`);
    }
  }
}
for (const evaluator of current.smallEvaluators || []) {
  if (!["submitted", "scheduled", "running"].includes(evaluator.status)) continue;
  const producer = current.producers.find((item) => item.id === evaluator.producerId);
  if (!producer) throw new Error(`missing producer for ${evaluator.id}`);
  const label = `${producer.model === "474m" ? "474M" : "153M"} · Pool-3B · BS${producer.batchSequences}`;
  const rowPattern = new RegExp(
    `<tr><td>E${evaluator.epoch}<\\/td><td>${label.replace(/[.*+?^${}()|[\\]\\]/g, "\\$&")}<\\/td>(.*?)<\\/tr>`,
  );
  const row = grid.match(rowPattern)?.[1];
  const expectedState = evaluator.status === "running" ? "POST running" : "POST queued";
  if (!row || !row.includes(`WD ${producer.weightDecay}) · PD retained · ${expectedState}`)) {
    throw new Error(`coordinate grid does not show active small-model POST for ${evaluator.id}`);
  }
}
for (const evaluator of current.smallEvaluators || []) {
  if (evaluator.status !== "complete") continue;
  const producer = current.producers.find((item) => item.id === evaluator.producerId);
  if (!producer) throw new Error(`missing producer for ${evaluator.id}`);
  const result = evaluator.postDecayResult;
  const postValue = Number(result?.validationExact ?? result?.validation);
  if (!Number.isFinite(postValue)) throw new Error(`missing completed POST value for ${evaluator.id}`);
  const label = `${producer.model === "474m" ? "474M" : "153M"} · Pool-3B · BS${producer.batchSequences}`;
  const rowPattern = new RegExp(
    `<tr><td>E${evaluator.epoch}<\\/td><td>${label.replace(/[.*+?^${}()|[\\]\\]/g, "\\$&")}<\\/td>(.*?)<\\/tr>`,
  );
  const row = grid.match(rowPattern)?.[1];
  if (!row || !row.includes(`WD ${producer.weightDecay}) · PD retained · POST ${postValue.toFixed(3)}`)) {
    throw new Error(`coordinate grid does not show completed POST value for ${evaluator.id}`);
  }
}
for (const run of current.dclm333mIntegratedRuns || []) {
  if (!["submitted", "scheduled", "running"].includes(run.status)) continue;
  const label = `${run.model === "1b" ? "1B" : run.model === "474m" ? "474M" : "153M"} · Pool-333M · BS${run.batchSequences}`;
  const rowPattern = new RegExp(
    `<tr><td>E${run.currentEpoch}<\\/td><td>${label.replace(/[.*+?^${}()|[\\]\\]/g, "\\$&")}<\\/td>(.*?)<\\/tr>`,
  );
  const row = grid.match(rowPattern)?.[1];
  const queued = ["submitted", "scheduled"].includes(run.status);
  const expectedState = run.currentPhase === "post"
    ? (queued ? "POST queued" : "POST running")
    : (queued ? "producer queued" : "producer running");
  if (!row || !row.includes(`WD ${run.weightDecay})`) || !row.includes(expectedState)) {
    throw new Error(`coordinate grid omits active Pool-333M state for ${run.id}`);
  }
}

const html = fs.readFileSync(path.join(reportRoot, "wsd_checkpoint_producer_grid.html"), "utf8");
if (!html.includes('<th colspan="6" class="model-start">1B</th>') ||
    !html.includes('<th colspan="8" class="model-start">474M</th>') ||
    !html.includes('<th colspan="8" class="model-start">153M</th>') ||
    (html.match(/Pool-333M/g) || []).length !== 3) {
  throw new Error("Pool-333M summary header topology is stale");
}

console.log(`validated rendered checkpoint report with ${expected.size} current summary cells`);
