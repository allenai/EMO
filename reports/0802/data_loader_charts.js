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
    columnByKey["baseline-256"],
    columnByKey.dr256,
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
    return lr < 2e-3 || (batch === 256 && lr === 2e-3 && wdNumber(sweep.wd) === 0.333);
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

  const coordinateColumns = [
    columnByKey["baseline-64"],
    columnByKey.dr64,
    columnByKey.drwt64,
    columnByKey.drwtembwd64,
    columnByKey["baseline-128"],
    columnByKey.dr128,
    columnByKey["baseline-256"],
    columnByKey.dr256,
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
        return `<span class="${classes.join(" ")}">(LR ${candidate.lr}, WD ${candidate.wd}) · ${candidate.status || "planned"}${value !== null ? ` · CE ${formatMetric(value)}` : ""}${attempts}</span>`;
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

  const sequentialBody = document.getElementById("wt-sequential-runs");
  const sequentialRuns = [
    ...(study.weightTyingSequentialRuns || []),
    ...(study.weightTyingEmbeddingDecaySequentialRuns || []),
  ];
  for (const run of sequentialRuns) {
    const stageStates = (run.stages || []).map((stage) => {
      const metrics = [];
      if (finite(stage.train)) metrics.push(`train ${formatMetric(stage.train)}`);
      if (finite(stage.validation)) metrics.push(`val ${formatMetric(stage.validation, 5)}`);
      if (finite(stage.gap)) metrics.push(`gap ${formatMetric(stage.gap, 5)}`);
      if (stage.status === "running" && finite(stage.progress?.percent)) {
        metrics.push(`${formatMetric(stage.progress.percent, 1)}%`);
      }
      return `E${stage.epoch}: ${stage.status || "planned"}${metrics.length ? ` (${metrics.join(", ")})` : ""}`;
    }).join(" · ");
    const experiment = run.experiment || run.beaker;
    const tr = document.createElement("tr");
    if (["submitted", "scheduled", "running"].includes(run.status)) tr.className = "run-active";
    if (["failed", "canceled"].includes(run.status)) tr.className = "run-failed";
    tr.innerHTML = `<td>BS${run.batchSequences}</td><td>${run.decayEmbeddings ? "DR+WT+EmbedWD" : "DR+WT"}</td><td>${run.lr}</td><td>${run.wd}</td><td>${run.status}</td><td>${run.currentEpoch ? `E${run.currentEpoch}` : "done"}</td><td>${stageStates}</td><td>${experiment ? `<a href="https://beaker.org/orgs/ai2/workspaces/flex2/work/${experiment}">${experiment}</a>` : "—"}</td>`;
    sequentialBody.appendChild(tr);
  }

  const gapBody = document.getElementById("wt-gap-summary");
  const weightTiedRuns = (study.runs || []).filter((run) => run.weightTying);
  for (const run of weightTiedRuns) {
    const baselineMethod = `dr${run.batchSequences}`;
    const baselineRun = (study.runs || []).find((candidate) =>
      candidate.method === baselineMethod &&
      lrNumber(candidate.lr) === lrNumber(run.lr) &&
      wdNumber(candidate.wd) === wdNumber(run.wd)
    );
    for (const epoch of Object.keys(run.results || {}).map(Number).sort((a, b) => a - b)) {
      const tied = run.results[epochKey(epoch)];
      if (!tied || tied.status !== "complete") continue;
      const baseline = baselineRun && baselineRun.results && baselineRun.results[epochKey(epoch)];
      const tiedGap = finite(tied.gap)
        ? Number(tied.gap)
        : (finite(tied.train) && finite(tied.validation) ? Number(tied.validation) - Number(tied.train) : null);
      const baselineGap = baseline && finite(baseline.gap)
        ? Number(baseline.gap)
        : (baseline && finite(baseline.train) && finite(baseline.validation)
          ? Number(baseline.validation) - Number(baseline.train)
          : null);
      const delta = tiedGap !== null && baselineGap !== null ? tiedGap - baselineGap : null;
      const verdict = delta === null ? "—" : delta < 0 ? "Yes" : delta > 0 ? "No" : "Tie";
      const tr = document.createElement("tr");
      if (delta !== null) tr.className = delta < 0 ? "gap-smaller" : delta > 0 ? "gap-larger" : "";
      tr.innerHTML = `<td>BS${run.batchSequences}</td><td>E${formatEpoch(epoch)}</td><td>${run.wd}</td><td>${formatMetric(baseline && baseline.train)}</td><td>${formatMetric(baseline && baseline.validation)}</td><td>${formatMetric(baselineGap, 4)}</td><td>${formatMetric(tied.train)}</td><td>${formatMetric(tied.validation)}</td><td>${formatMetric(tiedGap, 4)}</td><td>${delta === null ? "—" : `${delta >= 0 ? "+" : ""}${formatMetric(delta, 4)}`}</td><td>${verdict}</td>`;
      gapBody.appendChild(tr);
    }
  }

  const newRunsBody = document.getElementById("new-runs");
  for (const run of study.runs || []) {
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
      const wandb = (result && result.wandb) || (Number(run.activeEpoch) === epoch ? run.wandb : null);
      const beaker = (result && (result.beaker || result.experiment)) || run.beaker || run.experiment;
      const gap = result && (finite(result.gap)
        ? Number(result.gap)
        : (finite(result.train) && finite(result.validation)
          ? Number(result.validation) - Number(result.train)
          : null));
      tr.innerHTML = `<td>${methodByKey[run.method].label}</td><td>${run.batchSequences}</td><td>E${formatEpoch(epoch)}</td><td>${run.lr}</td><td>${run.wd}</td><td>${status || "planned"}</td><td>${formatMetric(result && result.train)}</td><td>${formatMetric(result && result.validation)}</td><td>${formatMetric(gap, 4)}</td><td>${wandb ? `<a href="https://wandb.ai/ai2-llm/sewonm-icsl/runs/${wandb}">${wandb}</a>` : "—"}</td><td>${beaker ? `<a href="https://beaker.org/orgs/ai2/workspaces/flex2/work/${beaker}">${beaker}</a>` : "—"}</td>`;
      newRunsBody.appendChild(tr);
    }
  }
})();
