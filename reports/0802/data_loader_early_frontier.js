(function () {
  "use strict";

  const study = window.ICSL_DATA_LOADER_DATA;
  const body = document.getElementById("early-frontier-evaluations");
  if (!study || !body) return;
  const finite = (value) => value !== null && value !== undefined && Number.isFinite(Number(value));
  const metric = (value) => finite(value) ? Number(value).toFixed(5) : "—";
  for (const item of study.earlyFrontierEvaluations || []) {
    const result = item.result || {};
    const experiment = item.experiment || item.beaker;
    const checkpoint = result.checkpoint || item.checkpoint || item.endpointCheckpoint;
    const row = document.createElement("tr");
    if (["submitted", "scheduled", "running"].includes(item.status)) row.className = "run-active";
    if (["failed", "canceled"].includes(item.status)) row.className = "run-failed";
    row.innerHTML = `<td>BS${item.batchSequences}</td><td>E${item.epoch}</td><td>${item.lr}</td><td>${item.wd}</td><td>${item.status}</td><td>${metric(result.validationExact ?? item.validationExact)}</td><td>${checkpoint || "—"}</td><td>${experiment ? `<a href="https://beaker.org/orgs/ai2/workspaces/flex2/work/${experiment}">${experiment}</a>` : "—"}</td>`;
    body.appendChild(row);
  }
})();
