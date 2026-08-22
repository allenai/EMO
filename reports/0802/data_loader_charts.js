(function () {
  "use strict";

  const batchData = window.ICSL_BATCH_BASELINE_DATA;
  const wdData = window.ICSL_WD_BASELINE_DATA;
  const study = window.ICSL_DATA_LOADER_DATA;
  if (!batchData || !wdData || !study) {
    throw new Error("Data-loader report inputs are missing");
  }

  const epochs = study.targetEpochs.map(Number);
  const baselineBatches = study.baselineBatches.map(Number);
  const baselineColumns = baselineBatches.map((batch) => ({
    key: `baseline-${batch}`,
    label: batch >= 512 ? `BS${batch} · Original` : `BS${batch}`,
    batchSequences: batch,
    color: "#64748b",
    baseline: true,
  }));
  const columnByKey = Object.fromEntries(
    baselineColumns.concat(study.columns).map((column) => [column.key, column]),
  );
  const columns = [
    columnByKey["baseline-64"],
    columnByKey.dr64,
    columnByKey.drwt64,
    columnByKey.drwtembwd64,
    columnByKey["baseline-128"],
    columnByKey.dr128,
    columnByKey.drwtembwd128,
    columnByKey["baseline-256"],
    columnByKey.dr256,
    columnByKey.drwtembwd256,
    columnByKey.fixed512,
    columnByKey["baseline-512"],
    columnByKey.dr512,
    columnByKey.drwt512,
    columnByKey.drwtembwd512,
    columnByKey.fixed1024,
    columnByKey["baseline-1024"],
    columnByKey.dr1024,
  ];

  const finite = (value) =>
    value !== null && value !== undefined && value !== "" && Number.isFinite(Number(value));
  const numeric = (value) => (finite(value) ? Number(value) : null);
  const epochKey = (value) => String(Number(value));
  const wdNumber = (value) => Number(String(value));
  const lrNumber = (value) => Number(String(value));
  const formatEpoch = (value) => Number(value).toLocaleString(undefined, {
    maximumFractionDigits: 3,
  });
  const formatMetric = (value, digits = 3) =>
    finite(value) ? Number(value).toFixed(digits) : "—";
  const validationPair = (result) => {
    if (!result || (!result.preDecay && !result.postDecay)) {
      return finite(result && result.validation) ? formatMetric(result.validation) : "—";
    }
    const post = finite(result.postDecay && result.postDecay.validation)
      ? formatMetric(result.postDecay.validation)
      : "Unknown";
    const pre = finite(result.preDecay && result.preDecay.validation)
      ? formatMetric(result.preDecay.validation)
      : "Unknown";
    return `${post} [POST] | ${pre} [PD]`;
  };
  const healthMap = Object.assign(
    {},
    batchData.healthAudit && batchData.healthAudit.unhealthy,
    wdData.healthAudit && wdData.healthAudit.unhealthy,
    study.healthAudit && study.healthAudit.unhealthy,
  );

  function admissibleBaseline(batch, sweep, result) {
    if (!result || result.status !== "complete" || !finite(result.validation)) return false;
    if (result.wandb && healthMap[result.wandb]) return false;
    const lr = lrNumber(sweep.lr);
    return lr < 2e-3 || ([256, 512].includes(batch) && lr === 2e-3 && wdNumber(sweep.wd) === 0.333);
  }

  function choose(candidates) {
    return candidates.slice().sort((a, b) =>
      Number(a.result.validation) - Number(b.result.validation) ||
      wdNumber(a.wd) - wdNumber(b.wd) ||
      lrNumber(a.lr) - lrNumber(b.lr),
    )[0] || null;
  }

  function baselineCandidate(batch, epoch) {
    const candidates = [];
    if (batch < 1024) {
      for (const sweep of batchData.batchSweeps || []) {
        if (Number(sweep.batchSequences) !== batch) continue;
        const result = sweep.results && sweep.results[epochKey(epoch)];
        if (!admissibleBaseline(batch, sweep, result)) continue;
        candidates.push({
          method: `baseline-${batch}`,
          batchSequences: batch,
          lr: sweep.lr,
          wd: sweep.wd,
          result,
        });
      }
    } else {
      for (const run of wdData.runs || []) {
        if (Number(run.epoch) !== Number(epoch)) continue;
        if (run.status !== "complete" || !finite(run.validation)) continue;
        if (run.wandb && healthMap[run.wandb]) continue;
        if (lrNumber(run.lr) >= 2e-3) continue;
        candidates.push({
          method: "baseline-1024",
          batchSequences: 1024,
          lr: run.lr,
          wd: run.wd,
          result: run,
        });
      }
    }
    return choose(candidates);
  }

  const selected = new Map();
  for (const batch of baselineBatches) {
    for (const epoch of epochs) {
      selected.set(`baseline-${batch}:${epochKey(epoch)}`, baselineCandidate(batch, epoch));
    }
  }

  for (const column of study.columns) {
    let wdFloor = -Infinity;
    for (const epoch of epochs) {
      // E1 predates either intervention. Summary tables show the ordinary baseline
      // winner; the coordinate grid below scopes its green source choice per method.
      if (epoch === 1 && !column.weightTying) {
        selected.set(`${column.key}:${epochKey(epoch)}`, baselineCandidate(column.batchSequences, epoch));
        continue;
      }
      const candidates = [];
      for (const run of study.runs || []) {
        if (run.method !== column.key) continue;
        if (wdNumber(run.wd) < wdFloor) continue;
        const result = run.results && run.results[epochKey(epoch)];
        if (!result || result.status !== "complete" || !finite(result.validation)) continue;
        if (result.wandb && healthMap[result.wandb]) continue;
        candidates.push({
          method: column.key,
          batchSequences: column.batchSequences,
          lr: run.lr,
          wd: run.wd,
          result,
        });
      }
      const winner = choose(candidates);
      selected.set(`${column.key}:${epochKey(epoch)}`, winner);
      if (winner) wdFloor = wdNumber(winner.wd);
    }
  }

  const getSelected = (column, epoch) => selected.get(`${column.key}:${epochKey(epoch)}`);
  const setText = (id, value) => {
    const element = document.getElementById(id);
    if (element) element.textContent = value;
  };

  setText("title", study.title);
  setText("setup", study.setup);
  setText("updated", `Updated ${study.updated}`);
  setText("selection", study.selection);

  const legend = document.getElementById("legend");
  for (const column of columns) {
    const label = document.createElement("label");
    label.innerHTML = `<span class="dot" style="background:${column.color}"></span>${column.label}`;
    legend.appendChild(label);
  }

  function renderChart(title, metric, digits) {
    const width = 360;
    const height = 220;
    const margin = { left: 46, right: 14, top: 28, bottom: 34 };
    const points = [];
    for (const column of columns) {
      for (const epoch of epochs) {
        const winner = getSelected(column, epoch);
        const value = winner && numeric(winner.result[metric]);
        if (value !== null) points.push({ column, epoch, value });
      }
    }
    if (!points.length) return;
    let yMin = Math.min(...points.map((point) => point.value));
    let yMax = Math.max(...points.map((point) => point.value));
    const pad = Math.max((yMax - yMin) * 0.1, 0.01);
    yMin -= pad;
    yMax += pad;
    const x = (epoch) =>
      margin.left +
      ((Math.log2(epoch) - Math.log2(epochs[0])) /
        (Math.log2(epochs[epochs.length - 1]) - Math.log2(epochs[0]))) *
        (width - margin.left - margin.right);
    const y = (value) =>
      margin.top + ((yMax - value) / (yMax - yMin)) * (height - margin.top - margin.bottom);
    const chart = document.createElement("div");
    chart.className = "chart";
    const groups = columns.map((column) => {
      const series = points.filter((point) => point.column.key === column.key);
      if (!series.length) return "";
      const path = series.map((point, index) => `${index ? "L" : "M"}${x(point.epoch)},${y(point.value)}`).join(" ");
      return `<path d="${path}" fill="none" stroke="${column.color}" stroke-width="${column.baseline ? 1.2 : 2.2}" stroke-opacity="${column.baseline ? 0.45 : 1}"/>${series.map((point) => `<circle cx="${x(point.epoch)}" cy="${y(point.value)}" r="${column.baseline ? 2 : 3}" fill="${column.color}"/>`).join("")}`;
    }).join("");
    const xTicks = epochs.map((epoch) => `<text x="${x(epoch)}" y="${height - 8}" text-anchor="middle">E${formatEpoch(epoch)}</text>`).join("");
    const yTicks = [0, 0.5, 1].map((fraction) => {
      const value = yMin + fraction * (yMax - yMin);
      return `<line x1="${margin.left}" x2="${width - margin.right}" y1="${y(value)}" y2="${y(value)}" stroke="#e2e8f0"/><text x="${margin.left - 6}" y="${y(value) + 3}" text-anchor="end">${formatMetric(value, digits)}</text>`;
    }).join("");
    chart.innerHTML = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${title}"><text x="${margin.left}" y="16" style="font-size:13px;font-weight:600;fill:#162033">${title}</text>${yTicks}${groups}${xTicks}</svg>`;
    document.getElementById("charts").appendChild(chart);
  }

  renderChart("DCLM validation CE", "validation", 3);

  function renderSummary(bodyId, cellValue, options = {}) {
    const body = document.getElementById(bodyId);
    const columnBest = new Map();
    if (options.columnBest) {
      for (const column of columns) {
        const values = epochs.map((epoch) => {
          const winner = getSelected(column, epoch);
          return winner && numeric(winner.result.validation);
        }).filter((value) => value !== null);
        if (values.length) columnBest.set(column.key, Math.min(...values));
      }
    }
    for (const epoch of epochs) {
      const winners = columns.map((column) => getSelected(column, epoch));
      const rowValues = winners.map((winner) => winner && numeric(winner.result.validation)).filter((value) => value !== null);
      const rowBest = rowValues.length ? Math.min(...rowValues) : null;
      const tr = document.createElement("tr");
      tr.innerHTML = `<td>E${formatEpoch(epoch)}</td>` + columns.map((column, index) => {
        const winner = winners[index];
        const value = winner && numeric(winner.result.validation);
        const classes = [];
        if (options.columnBest && value !== null && value === columnBest.get(column.key)) classes.push("summary-best");
        if (options.rowBest && value !== null && value === rowBest) classes.push("summary-row-best");
        return `<td class="${classes.join(" ")}">${cellValue(winner)}</td>`;
      }).join("");
      body.appendChild(tr);
    }
  }

  renderSummary("validation-summary", (winner) => winner ? formatMetric(winner.result.validation) : "—", { columnBest: true, rowBest: true });
  renderSummary("coordinate-summary", (winner) => winner ? `(${winner.lr}, ${winner.wd})` : "—");

  const downstreamTasks = [
    "arc_challenge", "arc_easy", "csqa", "hellaswag",
    "openbookqa", "piqa", "socialiqa", "winogrande",
  ];
  const completeAverage = (values) => {
    const available = values.map(numeric).filter((value) => value !== null);
    return available.length === downstreamTasks.length
      ? available.reduce((sum, value) => sum + value, 0) / available.length
      : null;
  };
  const downstreamMetric = (winner, metric) => {
    if (!winner) return null;
    const result = winner.result;
    if (metric === "hs_accuracy") return numeric(result.acc ?? result.downstream?.hellaswag);
    if (metric === "hs_bpb") return numeric(result.bpb ?? result.downstreamBpb?.hellaswag);
    if (metric === "avg8_accuracy") {
      return completeAverage(downstreamTasks.map((task) =>
        task === "hellaswag" ? (result.acc ?? result.downstream?.hellaswag) : result.downstream?.[task],
      ));
    }
    if (metric === "avg8_bpb") {
      return numeric(result.avg8Bpb) ??
        completeAverage(downstreamTasks.map((task) =>
          task === "hellaswag" ? (result.bpb ?? result.downstreamBpb?.hellaswag) : result.downstreamBpb?.[task],
        )) ?? numeric(wdData.avg8BpbByWandb?.[result.wandb]);
    }
    return null;
  };
  function renderDownstreamSummary({ bodyId, metric, higherIsBetter, digits, label }) {
    const body = document.getElementById(bodyId);
    if (!body) return;
    const columnBest = new Map();
    for (const column of columns) {
      const values = epochs.map((epoch) => downstreamMetric(getSelected(column, epoch), metric))
        .filter((value) => value !== null);
      if (values.length) columnBest.set(column.key, higherIsBetter ? Math.max(...values) : Math.min(...values));
    }
    for (const epoch of epochs) {
      const winners = columns.map((column) => getSelected(column, epoch));
      const values = winners.map((winner) => downstreamMetric(winner, metric));
      const available = values.filter((value) => value !== null);
      const rowBest = available.length
        ? (higherIsBetter ? Math.max(...available) : Math.min(...available))
        : null;
      const tr = document.createElement("tr");
      tr.innerHTML = `<td>E${formatEpoch(epoch)}</td>` + columns.map((column, index) => {
        const winner = winners[index];
        const value = values[index];
        if (value === null) return "<td>—</td>";
        const classes = [];
        if (value === columnBest.get(column.key)) classes.push("summary-best");
        if (value === rowBest) classes.push("summary-row-best");
        const coordinate = winner ? `LR ${winner.lr}; WD ${winner.wd}` : "coordinate unavailable";
        return `<td class="${classes.join(" ")}" title="${column.label}; ${coordinate}; ${label}">${value.toFixed(digits)}</td>`;
      }).join("");
      body.appendChild(tr);
    }
  }
  [
    { bodyId: "epoch-hs-accuracy-summary", metric: "hs_accuracy", higherIsBetter: true, digits: 2, label: "HellaSwag accuracy" },
    { bodyId: "epoch-hs-bpb-summary", metric: "hs_bpb", higherIsBetter: false, digits: 3, label: "HellaSwag BPB" },
    { bodyId: "epoch-avg8-accuracy-summary", metric: "avg8_accuracy", higherIsBetter: true, digits: 2, label: "8-task average accuracy" },
    { bodyId: "epoch-avg8-bpb-summary", metric: "avg8_bpb", higherIsBetter: false, digits: 3, label: "8-task average BPB" },
  ].forEach(renderDownstreamSummary);

  const coordinateColumns = [
    columnByKey["baseline-64"],
    columnByKey.dr64,
    columnByKey.drwt64,
    columnByKey.drwtembwd64,
    columnByKey["baseline-128"],
    columnByKey.dr128,
    columnByKey.drwtembwd128,
    columnByKey["baseline-256"],
    columnByKey.dr256,
    columnByKey.drwtembwd256,
    columnByKey.fixed512,
    columnByKey["baseline-512"],
    columnByKey.dr512,
    columnByKey.drwt512,
    columnByKey.drwtembwd512,
    columnByKey.fixed1024,
    columnByKey["baseline-1024"],
    columnByKey.dr1024,
  ];
  const activeStatuses = new Set(["planned", "submitted", "scheduled", "running"]);
  const failedStatuses = new Set(["failed", "canceled", "cancelled", "canceled_before_task_start"]);

  function coordinateCandidates(column, epoch) {
    const candidates = [];
    if (column.baseline && column.batchSequences < 1024) {
      for (const sweep of batchData.batchSweeps || []) {
        if (Number(sweep.batchSequences) !== column.batchSequences) continue;
        const result = sweep.results && sweep.results[epochKey(epoch)];
        if (!result) continue;
        candidates.push({ lr: sweep.lr, wd: sweep.wd, result, status: result.status });
      }
    } else if (column.baseline) {
      for (const run of wdData.runs || []) {
        if (Number(run.epoch) !== Number(epoch)) continue;
        candidates.push({ lr: run.lr, wd: run.wd, result: run, status: run.status });
      }
    } else {
      for (const run of study.runs || []) {
        if (run.method !== column.key) continue;
        const result = run.results && run.results[epochKey(epoch)];
        const attempted = Number(run.activeEpoch) === Number(epoch) ||
          (run.attemptedEpochs || []).map(Number).includes(Number(epoch));
        if (!result && !attempted) continue;
        const matchingAttempts = (run.attempts || []).filter((attempt) =>
          String(attempt.output || "").includes(`_e${epochKey(epoch)}_`),
        ).length;
        candidates.push({
          lr: run.lr,
          wd: run.wd,
          result: result || {},
          status: result
            ? result.status
            : (Number(run.activeEpoch) === Number(epoch) ? run.status : "planned"),
          attempts: matchingAttempts +
            (Number(run.activeEpoch) === Number(epoch) && run.experiment ? 1 : 0),
        });
      }
    }
    const distinct = new Map();
    for (const candidate of candidates) {
      const key = `${candidate.lr}:${candidate.wd}`;
      const current = distinct.get(key);
      const value = numeric(candidate.result.validation);
      const currentValue = current && numeric(current.result.validation);
      if (!current || (value !== null && (currentValue === null || value < currentValue)) ||
          (activeStatuses.has(candidate.status) && !activeStatuses.has(current.status))) {
        distinct.set(key, candidate);
      }
    }
    return [...distinct.values()].sort((a, b) =>
      lrNumber(a.lr) - lrNumber(b.lr) || wdNumber(a.wd) - wdNumber(b.wd),
    );
  }

  function coordinateWinner(column, epoch, candidates) {
    if (column.baseline || column.weightTying || Number(epoch) !== 1) {
      return getSelected(column, epoch);
    }
    // Fixed and DR share ordinary packing at E1, but each method has its own set of
    // registered source trajectories. Choose and highlight within that method only.
    return choose(candidates.filter((candidate) =>
      candidate.status === "complete" && finite(candidate.result.validation) &&
      !(candidate.result.wandb && healthMap[candidate.result.wandb]),
    ));
  }

  const coordinateBody = document.getElementById("coordinate-grid");
  for (const column of coordinateColumns) {
    for (const epoch of epochs) {
      const candidates = coordinateCandidates(column, epoch);
      const winner = coordinateWinner(column, epoch, candidates);
      const chips = candidates.map((candidate) => {
        const value = numeric(candidate.result.validation);
        const isSelected = winner &&
          lrNumber(winner.lr) === lrNumber(candidate.lr) &&
          wdNumber(winner.wd) === wdNumber(candidate.wd);
        const classes = ["tuple"];
        if (isSelected) classes.push("selected");
        else if (activeStatuses.has(candidate.status)) classes.push("active");
        else if (failedStatuses.has(candidate.status)) classes.push("failed");
        const attempts = candidate.attempts > 1 ? ` · ${candidate.attempts} attempts` : "";
        const paired = candidate.result.preDecay || candidate.result.postDecay;
        const metricText = paired
          ? ` · CE ${validationPair(candidate.result)}`
          : (value !== null ? ` · CE ${formatMetric(value)}` : "");
        return `<span class="${classes.join(" ")}">(LR ${candidate.lr}, WD ${candidate.wd}) · ${candidate.status || "planned"}${metricText}${attempts}</span>`;
      }).join("");
      const selection = winner
        ? `LR ${winner.lr}, WD ${winner.wd} · CE ${formatMetric(winner.result.validation)}`
        : "pending";
      const tr = document.createElement("tr");
      tr.innerHTML = `<td>E${formatEpoch(epoch)}</td><td>${column.label}</td><td><div class="tuple-list">${chips || "—"}</div></td><td>${selection}</td>`;
      coordinateBody.appendChild(tr);
    }
  }

  const optimizerBody = document.getElementById("optimizer-step-summary");
  for (const comparison of (batchData.optimizerStepComparisons || [])) {
    const winners = columns.map((column) => {
      const epoch = comparison.epochs[String(column.batchSequences)];
      return epoch === undefined ? null : getSelected(column, epoch);
    });
    const values = winners
      .map((winner) => numeric(winner?.result?.validation))
      .filter((value) => value !== null);
    const rowBest = values.length ? Math.min(...values) : null;
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${Number(comparison.optimizerSteps).toLocaleString()}</td>` + columns.map((column, index) => {
      const epoch = comparison.epochs[String(column.batchSequences)];
      const winner = winners[index];
      const value = numeric(winner?.result?.validation);
      const isBest = value !== null && value === rowBest;
      const metric = winner ? formatMetric(value) : null;
      const display = winner
        ? `E${formatEpoch(epoch)} · ${isBest ? `<strong><u>${metric}</u></strong>` : metric}`
        : "—";
      return `<td class="${isBest ? "summary-best summary-row-best" : ""}"${isBest ? ' style="font-weight:900!important"' : ""}>${display}</td>`;
    }).join("");
    optimizerBody.appendChild(tr);
  }

  const methodByKey = Object.fromEntries(study.columns.map((column) => [column.key, column]));

  const downstreamCampaignBody = document.getElementById("downstream-eval-campaigns");
  for (const campaign of study.downstreamEvaluationCampaigns || []) {
    const tasks = campaign.tasks || [];
    const count = (status) => tasks.filter((task) => task.status === status).length;
    const experiment = campaign.experiment;
    const tr = document.createElement("tr");
    if (["submitted", "scheduled", "running"].includes(campaign.status)) tr.className = "run-active";
    if (campaign.status === "failed") tr.className = "run-failed";
    tr.innerHTML = `<td>${campaign.status}</td><td>${campaign.taskCount ?? tasks.length}</td><td>${campaign.checkpointCount ?? tasks.length}</td><td>${count("complete")}</td><td>${count("running")}</td><td>${count("scheduled") + count("submitted")}</td><td>${count("unavailable")}</td><td>${count("failed")}</td><td>${campaign.gpuPerTask ?? 1}</td><td>${campaign.revision || "—"}</td><td>${experiment ? `<a href="https://beaker.org/orgs/ai2/workspaces/flex2/work/${experiment}">${experiment}</a>` : "—"}</td>`;
    downstreamCampaignBody.appendChild(tr);
  }

  const sequentialBody = document.getElementById("wt-sequential-runs");
  const sequentialRuns = [
    ...(study.weightTyingSequentialRuns || []),
    ...(study.weightTyingEmbeddingDecaySequentialRuns || []),
    ...(study.drWtEmbedWdGridChains || []),
  ];
  for (const run of sequentialRuns) {
    const stageStates = (run.stages || []).map((stage) => {
      const metrics = [];
      if (stage.status === "running" && finite(stage.progress?.percent)) {
        metrics.push(`${formatMetric(stage.progress.percent, 1)}%`);
      }
      return `E${stage.epoch}: ${stage.status || "planned"}${metrics.length ? ` (${metrics.join(", ")})` : ""}`;
    }).join(" · ");
    const experiment = run.experiment || run.beaker;
    const coordinateExperiments = Object.values(run.experimentsByCoordinate || {});
    const tr = document.createElement("tr");
    if (["submitted", "scheduled", "running"].includes(run.status)) tr.className = "run-active";
    if (["failed", "canceled"].includes(run.status)) tr.className = "run-failed";
    const variant = run.variant || (run.decayEmbeddings ? "DR+WT+EmbedWD" : "DR+WT");
    const gridLrs = [...new Set((run.coordinates || []).map((coordinate) => coordinate.lr))].join(", ");
    const gridWds = [...new Set((run.coordinates || []).map((coordinate) => coordinate.wd))].join(", ");
    const lr = run.lr || gridLrs || "—";
    const wd = run.wd || gridWds || "—";
    const trigger = !coordinateExperiments.length && finite(run.triggerThreshold)
      ? `waiting for ${run.completedSmallChainsAtLastCheck || 0}/${run.triggerThreshold} small chains`
      : (run.trigger ? run.reason : "");
    const experimentLinks = coordinateExperiments.length
      ? coordinateExperiments.map((id) => `<a href="https://beaker.org/orgs/ai2/workspaces/flex2/work/${id}">${id}</a>`).join("<br>")
      : (experiment ? `<a href="https://beaker.org/orgs/ai2/workspaces/flex2/work/${experiment}">${experiment}</a>` : "—");
    tr.innerHTML = `<td>BS${run.batchSequences}</td><td>${variant}</td><td>${lr}</td><td>${wd}</td><td>${run.status}</td><td>${run.currentEpoch ? `E${run.currentEpoch}` : (["planned", "held", "conditional_held"].includes(run.status) ? "waiting" : "done")}</td><td>${stageStates || trigger || "—"}</td><td>${experimentLinks}</td>`;
    sequentialBody.appendChild(tr);
  }

  const newRunsBody = document.getElementById("new-runs");
  for (const run of study.runs || []) {
    if (String(run.method || "").includes("mlpupperwd1")) continue;
    const resultEpochs = new Set(Object.keys(run.results || {}).map(Number));
    for (const epoch of run.attemptedEpochs || []) resultEpochs.add(Number(epoch));
    if (run.activeEpoch !== undefined && run.activeEpoch !== null) resultEpochs.add(Number(run.activeEpoch));
    for (const epoch of Array.from(resultEpochs).sort((a, b) => a - b)) {
      const result = run.results && run.results[epochKey(epoch)];
      const status = result
        ? result.status
        : (Number(run.activeEpoch) === Number(epoch) ? run.status : "planned");
      const tr = document.createElement("tr");
      if (["running", "scheduled", "submitted"].includes(status)) tr.className = "run-active";
      if (["failed", "canceled"].includes(status)) tr.className = "run-failed";
      const wandb = (result && (result.wandb || result.postDecay?.wandb || result.preDecay?.wandb)) || (Number(run.activeEpoch) === epoch ? run.wandb : null);
      const beaker = (result && (result.beaker || result.experiment)) || run.beaker || run.experiment;
      const gap = result && (finite(result.gap)
        ? Number(result.gap)
        : (finite(result.train) && finite(result.validation)
          ? Number(result.validation) - Number(result.train)
          : null));
      const label = (methodByKey[run.method] && methodByKey[run.method].label) || run.label || run.method;
      tr.innerHTML = `<td>${label}</td><td>${run.batchSequences}</td><td>E${formatEpoch(epoch)}</td><td>${run.lr}</td><td>${run.wd}</td><td>${status || "planned"}</td><td>${formatMetric(result && result.train)}</td><td>${validationPair(result)}</td><td>${formatMetric(gap, 4)}</td><td>${wandb ? `<a href="https://wandb.ai/ai2-llm/sewonm-icsl/runs/${wandb}">${wandb}</a>` : "—"}</td><td>${beaker ? `<a href="https://beaker.org/orgs/ai2/workspaces/flex2/work/${beaker}">${beaker}</a>` : "—"}</td>`;
      newRunsBody.appendChild(tr);
    }
  }
})();
