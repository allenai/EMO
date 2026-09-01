(() => {
  "use strict";

  const historical = window.ICSL_HISTORICAL_1B_DATA;
  const historicalSmall = {
    "474m": window.ICSL_HISTORICAL_474M_DATA,
    "153m": window.ICSL_HISTORICAL_153M_DATA,
  };
  const current = window.ICSL_CHECKPOINT_PRODUCER_GRID;
  if (!historical || !historicalSmall["474m"] || !historicalSmall["153m"] || !current) {
    throw new Error("Checkpoint-grid report inputs are missing");
  }

  const finite = (value) =>
    value !== null && value !== undefined && value !== "" && Number.isFinite(Number(value));
  const numeric = (value) => finite(value) ? Number(value) : null;
  const formatMetric = (value) => finite(value) ? Number(value).toFixed(3) : "—";
  const formatEpoch = (value) => `E${Number(value).toLocaleString()}`;
  const formatPool = (pool) => ({
    dclm333m: "Pool-333M",
    dclm1b: "Pool-1B",
    dclm3b: "Pool-3B",
  }[pool] || pool);
  const formatModel = (model) => ({"1b": "1B", "474m": "474M", "153m": "153M"}[model] || model);
  const isActive = (status) => ["submitted", "scheduled", "running"].includes(status);
  const isFailed = (status) => ["failed", "canceled", "cancelled", "error"].includes(status);

  const integratedRuns = current.dclm333mIntegratedRuns || [];
  const producers = [...(current.producers || []), ...integratedRuns];
  const dense1bEvaluators = current.evaluators || [];
  const evaluators = [
    ...dense1bEvaluators,
    ...dense1bEvaluators.flatMap((evaluator) =>
      (evaluator.additionalExperiments || []).map((additional) => ({
        ...evaluator,
        ...additional,
        id: `${evaluator.id}-post-e${additional.epoch}`,
        epochs: [additional.epoch],
        resolvedPostEpochs: additional.postDecayResult ? [additional.epoch] : [],
        postDecayResults: additional.postDecayResult
          ? {[String(additional.epoch)]: additional.postDecayResult}
          : {},
        additionalExperiments: [],
      })),
    ),
    ...(current.smallEvaluators || []).map((evaluator) => ({
      ...evaluator,
      epochs: evaluator.epochs || [evaluator.epoch],
      resolvedPostEpochs: evaluator.resolvedPostEpochs ||
        (evaluator.resolvedPostEpoch ? [evaluator.resolvedPostEpoch] : []),
      postDecayResults: evaluator.postDecayResults ||
        (evaluator.postDecayResult ? {[String(evaluator.epoch)]: evaluator.postDecayResult} : {}),
    })),
    ...integratedRuns.map((run) => ({
      id: `${run.id}-integrated-post`,
      producerId: run.id,
      status: run.currentPhase === "post" ? run.status :
        run.status === "complete" ? "complete" : "idle",
      epochs: run.evaluationEpochs || [],
      resolvedPostEpochs: run.resolvedPostEpochs || [],
      postDecayResults: run.postDecayResults || {},
      currentEvaluationEpoch: run.currentPostEpoch ?? null,
    })),
  ];
  const evaluatorsByProducer = new Map();
  for (const evaluator of evaluators) {
    const list = evaluatorsByProducer.get(evaluator.producerId) || [];
    list.push(evaluator);
    evaluatorsByProducer.set(evaluator.producerId, list);
  }

  const columns = [
    {key: "1b-pool1b-bs64", model: "1b", pool: "dclm1b", batch: 64, historicalIds: ["drwtembwd64-lr1e-3-wd0.3"], modelStart: true},
    {key: "1b-pool1b-bs128", model: "1b", pool: "dclm1b", batch: 128, historicalIds: ["drwtembwd128-lr1e-3-wd0.3", "drwtembwd128-lr1e-3-wd1.0"]},
    {key: "1b-pool3b-bs64", model: "1b", pool: "dclm3b", batch: 64, poolStart: true},
    {key: "1b-pool3b-bs128", model: "1b", pool: "dclm3b", batch: 128},
    {key: "474m-pool333m-bs32", model: "474m", pool: "dclm333m", batch: 32, modelStart: true},
    {key: "474m-pool333m-bs64", model: "474m", pool: "dclm333m", batch: 64},
    {key: "474m-pool333m-bs128", model: "474m", pool: "dclm333m", batch: 128},
    {key: "474m-pool1b-bs128", model: "474m", pool: "dclm1b", batch: 128, historicalSmall: true, poolStart: true},
    {key: "474m-pool1b-bs256", model: "474m", pool: "dclm1b", batch: 256, historicalSmall: true},
    {key: "474m-pool3b-bs128", model: "474m", pool: "dclm3b", batch: 128, poolStart: true},
    {key: "474m-pool3b-bs256", model: "474m", pool: "dclm3b", batch: 256},
    {key: "474m-pool3b-bs512", model: "474m", pool: "dclm3b", batch: 512},
    {key: "153m-pool333m-bs32", model: "153m", pool: "dclm333m", batch: 32, modelStart: true},
    {key: "153m-pool333m-bs64", model: "153m", pool: "dclm333m", batch: 64},
    {key: "153m-pool333m-bs128", model: "153m", pool: "dclm333m", batch: 128},
    {key: "153m-pool1b-bs128", model: "153m", pool: "dclm1b", batch: 128, historicalSmall: true, poolStart: true},
    {key: "153m-pool1b-bs256", model: "153m", pool: "dclm1b", batch: 256, historicalSmall: true},
    {key: "153m-pool3b-bs128", model: "153m", pool: "dclm3b", batch: 128, poolStart: true},
    {key: "153m-pool3b-bs256", model: "153m", pool: "dclm3b", batch: 256},
    {key: "153m-pool3b-bs512", model: "153m", pool: "dclm3b", batch: 512},
  ];

  const runById = Object.fromEntries((historical.runs || []).map((run) => [run.id, run]));

  function historicalPoints(column) {
    const points = [];
    for (const id of column.historicalIds || []) {
      const run = runById[id];
      if (!run) continue;
      if (!String(run.method || "").startsWith("drwtembwd") ||
          run.weightTying !== true || run.decayEmbeddings !== true) {
        throw new Error(`Historical 1B run ${id} is not DR+WT+EmbedWD`);
      }
      for (const [epochText, result] of Object.entries(run.results || {})) {
        const value = numeric(result.validationExact ?? result.validation);
        if (value === null || result.status !== "complete") continue;
        const epoch = Number(epochText);
        points.push({
          epoch,
          value,
          lr: run.lr,
          wd: run.wd,
          producerId: id,
        });
      }
    }
    return points;
  }

  function historicalSmallPoints(column) {
    const data = historicalSmall[column.model];
    const chains = (data.adaptiveDrWtEmbedWdChains || []).filter((chain) =>
      Number(chain.batchSequences) === column.batch,
    );
    if (chains.length !== 1) {
      throw new Error(`Expected one ${column.model} BS${column.batch} DR+WT+EmbedWD chain`);
    }
    const chain = chains[0];
    if (chain.variant !== "DR+WT+EmbedWD" || chain.weightTying !== true ||
        chain.decayEmbeddings !== true || chain.embeddingWeightDecay !== "global" ||
        String(chain.lr) !== "2e-3") {
      throw new Error(`${column.model} BS${column.batch} source is not DR+WT+EmbedWD`);
    }
    const selected = new Map();
    for (const [epochText, frontier] of Object.entries(chain.frontiers || {})) {
      const wd = String(frontier.selectedWd);
      const result = chain.results?.[epochText]?.[wd];
      const value = numeric(result?.validationExact ?? result?.validation);
      if (result?.status !== "complete" || value === null) continue;
      selected.set(Number(epochText), {
        epoch: Number(epochText), value, lr: chain.lr, wd,
        producerId: chain.id,
      });
    }
    for (const [epochText, result] of Object.entries(chain.postDecayResults || {})) {
      const value = numeric(result?.validationExact ?? result?.validation);
      if (result?.status !== "complete" || result?.comparisonGroup !== "post_decay" ||
          value === null) continue;
      selected.set(Number(epochText), {
        epoch: Number(epochText), value, lr: chain.lr, wd: String(chain.lockedWd),
        producerId: chain.id,
      });
    }
    return [...selected.values()];
  }

  function currentPoints(column) {
    const points = [];
    const matching = producers.filter((producer) =>
      producer.model === column.model && producer.pool === column.pool &&
      Number(producer.batchSequences) === column.batch,
    );
    for (const producer of matching) {
      for (const evaluator of evaluatorsByProducer.get(producer.id) || []) {
        for (const [epochText, result] of Object.entries(evaluator.postDecayResults || {})) {
          const value = numeric(result.validationExact ?? result.validation);
          if (value === null || result.status !== "complete") continue;
          const epoch = Number(epochText);
          points.push({
            epoch,
            value,
            lr: producer.learningRate,
            wd: producer.weightDecay,
            producerId: producer.id,
          });
        }
      }
    }
    return points;
  }

  const pointsByColumn = new Map(columns.map((column) => [
    column.key,
    column.historicalIds ? historicalPoints(column) :
      column.historicalSmall ? historicalSmallPoints(column) : currentPoints(column),
  ]));

  function choose(points) {
    return points.slice().sort((left, right) =>
      left.value - right.value || Number(left.wd) - Number(right.wd) || Number(left.lr) - Number(right.lr),
    )[0] || null;
  }

  function selectedAtEpoch(column, epoch) {
    return choose((pointsByColumn.get(column.key) || []).filter((point) => point.epoch === Number(epoch)));
  }

  const validationEpochs = [...new Set(columns.flatMap((column) =>
    (pointsByColumn.get(column.key) || []).map((point) => point.epoch),
  ))].sort((left, right) => left - right);

  const columnBest = new Map(columns.map((column) => {
    const values = (pointsByColumn.get(column.key) || []).map((point) => point.value);
    return [column.key, values.length ? Math.min(...values) : null];
  }));

  const validationBody = document.getElementById("validation-summary");
  for (const epoch of validationEpochs) {
    const selected = columns.map((column) => selectedAtEpoch(column, epoch));
    const rowBestByModel = new Map(["1b", "474m", "153m"].map((model) => {
      const values = selected.filter((point, index) => columns[index].model === model)
        .map((point) => point?.value).filter(finite).map(Number);
      return [model, values.length ? Math.min(...values) : null];
    }));
    const cells = selected.map((point, index) => {
      const divider = columns[index].modelStart ? "model-start" :
        columns[index].poolStart ? "pool-start" : "";
      if (!point) return `<td class="${divider}">—</td>`;
      const classes = [];
      if (divider) classes.push(divider);
      if (point.value === columnBest.get(columns[index].key)) classes.push("summary-best");
      if (point.value === rowBestByModel.get(columns[index].model)) classes.push("summary-row-best");
      return `<td class="${classes.join(" ")}">${formatMetric(point.value)}</td>`;
    }).join("");
    validationBody.insertAdjacentHTML("beforeend", `<tr><td>${formatEpoch(epoch)}</td>${cells}</tr>`);
  }

  const groupOrder = [
    ["1b", "dclm3b", 64], ["1b", "dclm3b", 128],
    ["474m", "dclm333m", 32], ["474m", "dclm333m", 64], ["474m", "dclm333m", 128],
    ["474m", "dclm3b", 128], ["474m", "dclm3b", 256], ["474m", "dclm3b", 512],
    ["153m", "dclm333m", 32], ["153m", "dclm333m", 64], ["153m", "dclm333m", 128],
    ["153m", "dclm3b", 128], ["153m", "dclm3b", 256], ["153m", "dclm3b", 512],
  ];
  const coordinateBody = document.getElementById("coordinate-grid");

  function evaluatorActiveEpoch(evaluator) {
    if (!isActive(evaluator.status)) return null;
    if (evaluator.currentEvaluationEpoch !== null &&
        evaluator.currentEvaluationEpoch !== undefined) {
      return Number(evaluator.currentEvaluationEpoch);
    }
    const resolved = new Set((evaluator.resolvedPostEpochs || []).map(Number));
    return (evaluator.epochs || []).map(Number).find((epoch) => !resolved.has(epoch)) ?? null;
  }

  for (const [model, pool, batch] of groupOrder) {
    const groupProducers = producers.filter((producer) =>
      producer.model === model && producer.pool === pool && Number(producer.batchSequences) === batch,
    );
    const epochs = [...new Set(groupProducers.flatMap((producer) => {
      const evaluatorEpochs = (evaluatorsByProducer.get(producer.id) || []).flatMap((evaluator) => [
        ...(evaluator.resolvedPostEpochs || []),
        evaluatorActiveEpoch(evaluator),
      ]);
      return [...(producer.resolvedCheckpointEpochs || []), producer.currentEpoch, ...evaluatorEpochs]
        .filter((epoch) => epoch !== null && epoch !== undefined)
        .map(Number);
    }))]
      .filter((epoch) => epoch !== 1 || groupProducers.some((producer) =>
        Number(producer.currentEpoch) === 1 && isActive(producer.status),
      ))
      .sort((left, right) => left - right);

    for (const epoch of epochs) {
      const postCandidates = [];
      const chips = groupProducers.map((producer) => {
        const resolvedPd = (producer.resolvedCheckpointEpochs || []).map(Number).includes(epoch);
        const producerPhase = producer.role === "constant_lr_checkpoint_producer" ||
          !producer.currentPhase || producer.currentPhase === "producer";
        const producerActive = Number(producer.currentEpoch) === epoch &&
          producer.status === "running" && producerPhase;
        const producerQueued = Number(producer.currentEpoch) === epoch &&
          ["submitted", "scheduled"].includes(producer.status) && producerPhase;
        const producerFailed = Number(producer.currentEpoch) === epoch && isFailed(producer.status);
        const producerEvaluators = evaluatorsByProducer.get(producer.id) || [];
        let postResult = null;
        let postRunning = false;
        let postQueued = false;
        let postFailed = false;
        for (const evaluator of producerEvaluators) {
          const candidate = evaluator.postDecayResults?.[String(epoch)];
          const value = numeric(candidate?.validationExact ?? candidate?.validation);
          if (candidate?.status === "complete" && value !== null) {
            const point = {producerId: producer.id, value, lr: producer.learningRate, wd: producer.weightDecay};
            postCandidates.push(point);
            if (!postResult || value < postResult.value) postResult = point;
          }
          if (evaluatorActiveEpoch(evaluator) === epoch) {
            if (evaluator.status === "running") postRunning = true;
            else if (["submitted", "scheduled"].includes(evaluator.status)) postQueued = true;
          }
          if (isFailed(evaluator.status) && !(evaluator.resolvedPostEpochs || []).map(Number).includes(epoch)) postFailed = true;
        }
        const states = [];
        if (resolvedPd) states.push("PD retained");
        if (postResult) states.push(`POST ${formatMetric(postResult.value)}`);
        else if (postRunning) states.push("POST running");
        else if (postQueued) states.push("POST queued");
        if (producerActive) states.push("producer running");
        if (producerQueued) states.push("producer queued");
        if (producerFailed) states.push("producer failed");
        if (postFailed) states.push("POST failed");
        if (!states.length) return null;
        const classes = ["tuple"];
        if (producerActive || producerQueued || postRunning || postQueued) classes.push("active");
        if (producerFailed || postFailed) classes.push("failed");
        return {producer, postResult, classes, text: `(LR ${producer.learningRate}, WD ${producer.weightDecay}) · ${states.join(" · ")}`};
      });
      const winner = choose(postCandidates);
      const visibleChips = chips.filter(Boolean);
      if (!visibleChips.length) continue;
      const chipHtml = visibleChips.map((chip) => {
        if (winner && chip.producer.id === winner.producerId) chip.classes.push("selected");
        return `<span class="${chip.classes.join(" ")}">${chip.text}</span>`;
      }).join("");
      const selection = winner
        ? `LR ${winner.lr}, WD ${winner.wd} · CE ${formatMetric(winner.value)}`
        : "pending";
      const label = `${formatModel(model)} · ${formatPool(pool)} · BS${batch}`;
      coordinateBody.insertAdjacentHTML("beforeend", `<tr><td>${formatEpoch(epoch)}</td><td>${label}</td><td><div class="tuple-list">${chipHtml}</div></td><td>${selection}</td></tr>`);
    }
  }
})();
