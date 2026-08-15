import fs from 'node:fs';
import path from 'node:path';

const repo = path.resolve(import.meta.dirname, '../..');
const reportDir = path.join(repo, 'reports/0802');
const source = JSON.parse(fs.readFileSync(path.join(reportDir, 'data/wsd_unique_vs_repeated_1b.json'), 'utf8'));
const current = JSON.parse(fs.readFileSync(path.join(reportDir, 'data/wsd_batch_size_1b.json'), 'utf8'));

const specs = [
  {slug: '153m', modelKey: '153M', runsKey: 'smallRepeatedRuns'},
  {slug: '474m', modelKey: '474M', runsKey: 'mediumRepeatedRuns'},
];

const completeMetricRun = run => run.status === 'complete' && Number.isFinite(run.c4);

function build(spec) {
  const model = source.modelSizes[spec.modelKey];
  const batchSequences = model.globalBatchTokens / 4096;
  const runs = source[spec.runsKey].filter(completeMetricRun);
  const epochs = [...new Set(runs.map(run => run.epoch))].sort((a, b) => a - b);
  const unhealthyIds = new Set(runs.map(run => run.wandb).filter(id => source.healthAudit?.unhealthy?.[id]));
  const unhealthy = Object.fromEntries([...unhealthyIds].map(id => [id, source.healthAudit.unhealthy[id]]));
  const byLr = new Map();
  for (const run of runs) {
    if (!byLr.has(run.lr)) byLr.set(run.lr, []);
    byLr.get(run.lr).push(run);
  }
  const batchSweeps = [...byLr.entries()].sort((a, b) => Number(a[0]) - Number(b[0])).map(([lr, lrRuns]) => {
    const ordered = [...lrRuns].sort((a, b) => a.epoch - b.epoch);
    const results = Object.fromEntries(ordered.map(run => [String(run.epoch), {
      ...run,
      validation: run.c4,
    }]));
    return {
      batchSequences,
      globalBatchTokens: model.globalBatchTokens,
      contextLength: 4096,
      lr,
      wd: '0.033',
      warmupSteps: model.warmupSteps,
      status: 'complete',
      activeEpoch: ordered.at(-1).epoch,
      search: 'historical-pre-wd-and-batch-tuning',
      beaker: ordered.find(run => run.beaker)?.beaker,
      results,
      reason: `Imported from the original ${model.label} repeated-data WSD sweep. This trajectory used fixed WD 0.033 and global batch ${batchSequences} sequences before weight-decay and batch-size tuning.`,
    };
  });
  return {
    updated: '2026-08-09',
    title: `0802 Step 1-1 · ${model.label}`,
    setup: `${model.label}; 1B-unique-pool repeated-token regime; sequence length 4096; global batch ${batchSequences} sequences (${model.globalBatchTokens.toLocaleString()} tokens); fixed WD 0.033. These runs predate weight-decay and batch-size tuning.`,
    selection: `Historical import only: the displayed epoch/LR winners are provisional minima at the original fixed WD 0.033 and global batch ${batchSequences}. Neither weight decay nor batch size is locally resolved for this model. Red trajectories are excluded from provisional selection but retained in the coordinate grid and provenance.`,
    selectionPolicy: {allowAllCompletedCoordinates: true},
    healthAudit: {
      updated: source.healthAudit?.updated,
      criterion: source.healthAudit?.criterion,
      trajectoryPolicy: source.healthAudit?.trajectoryPolicy,
      suspicious: {},
      unhealthy,
    },
    targetEpochs: epochs,
    batchTargetEpochs: {[String(batchSequences)]: epochs},
    optimizerStepComparisons: current.optimizerStepComparisons,
    batchSweeps,
  };
}

for (const spec of specs) {
  const data = build(spec);
  const json = `${JSON.stringify(data, null, 2)}\n`;
  fs.writeFileSync(path.join(reportDir, `data/wsd_batch_size_${spec.slug}.json`), json);
  fs.writeFileSync(path.join(reportDir, `data/wsd_batch_size_${spec.slug}.js`), `window.ICSL_REPORT_DATA=${JSON.stringify(data)};\n`);
}
