(function () {
  "use strict";

  const study = window.ICSL_DATA_LOADER_DATA;
  const body = document.getElementById("redecay-retries");
  if (!study || !body) return;

  const finite = (value) => value !== null && value !== undefined && Number.isFinite(Number(value));
  const metric = (value, digits = 5) => finite(value) ? Number(value).toFixed(digits) : "—";
  for (const retry of study.redecayRetries || []) {
    const result = retry.result || {};
    const current = result.validationExact ?? retry.validationExact;
    const delta = finite(current) && finite(retry.originalValidationExact)
      ? Number(current) - Number(retry.originalValidationExact)
      : null;
    const experiment = retry.experiment || retry.beaker;
    const checkpoint = result.checkpoint || retry.checkpoint || retry.endpointCheckpoint;
    const row = document.createElement("tr");
    if (["submitted", "scheduled", "running"].includes(retry.status)) row.className = "run-active";
    if (["failed", "canceled"].includes(retry.status)) row.className = "run-failed";
    row.innerHTML = `<td>BS${retry.batchSequences}</td><td>E${retry.epoch}</td><td>${retry.lr}</td><td>${retry.wd}</td><td>${retry.status}</td><td>${metric(retry.originalValidationExact)}</td><td>${metric(current)}</td><td>${delta === null ? "—" : `${delta >= 0 ? "+" : ""}${metric(delta)}`}</td><td>${checkpoint || "—"}</td><td>${experiment ? `<a href="https://beaker.org/orgs/ai2/workspaces/flex2/work/${experiment}">${experiment}</a>` : "—"}</td>`;
    body.appendChild(row);
  }
})();
