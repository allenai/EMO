(function(){
  "use strict";
  const batchData=window.ICSL_BATCH_BASELINE_DATA;
  const wdData=window.ICSL_WD_BASELINE_DATA;
  const study=window.ICSL_DATA_LOADER_DATA;
  const warmdown=window.ICSL_BATCH_WARMDOWN_DATA;
  if(!batchData||!wdData||!study||!warmdown)throw new Error("Batch warmdown report inputs are missing");

  const batches=study.baselineBatches.map(Number);
  const epochs=study.targetEpochs.map(Number);
  const colors={64:"#f97316",128:"#0d9488",256:"#2563eb",512:"#ea580c",1024:"#e11d48"};
  const columns=batches.flatMap(batch=>[
    {key:`baseline-${batch}`,label:`BS${batch} · Original`,batchSequences:batch,color:colors[batch],baseline:true},
    {key:`dr${batch}`,label:`BS${batch} · DR`,batchSequences:batch,color:colors[batch],baseline:false},
  ]);
  const warmdownChains=[
    {key:"warmdown",label:"Warmdown · fixed coordinate",color:"#7c3aed",runs:warmdown.runs||[]},
    {key:"best-warmdown",label:"Warmdown · best coordinate",color:"#16a34a",runs:warmdown.bestCoordinateRuns||[]},
  ];
  const finite=value=>value!==null&&value!==undefined&&value!==""&&Number.isFinite(Number(value));
  const numeric=value=>finite(value)?Number(value):null;
  const epochKey=value=>String(Number(value));
  const wdNumber=value=>Number(String(value));
  const lrNumber=value=>Number(String(value));
  const formatEpoch=value=>Number(value).toLocaleString(undefined,{maximumFractionDigits:3});
  const formatMetric=value=>finite(value)?Number(value).toFixed(3):"—";
  const formatDuration=seconds=>{
    const total=Math.max(0,Math.round(Number(seconds)));
    const hours=Math.floor(total/3600),minutes=Math.floor((total%3600)/60),secs=total%60;
    return hours?`${hours}h ${minutes}m ${secs}s`:`${minutes}m ${secs}s`;
  };
  const healthMap=Object.assign({},batchData.healthAudit?.unhealthy,wdData.healthAudit?.unhealthy,study.healthAudit?.unhealthy);

  function admissibleBaseline(batch,sweep,result){
    if(!result||result.status!=="complete"||!finite(result.validation))return false;
    if(result.wandb&&healthMap[result.wandb])return false;
    const lr=lrNumber(sweep.lr);
    return lr<2e-3||(batch===256&&lr===2e-3&&wdNumber(sweep.wd)===0.333);
  }
  function choose(candidates){
    return candidates.slice().sort((a,b)=>Number(a.result.validation)-Number(b.result.validation)||wdNumber(a.wd)-wdNumber(b.wd)||lrNumber(a.lr)-lrNumber(b.lr))[0]||null;
  }
  function baselineCandidate(batch,epoch){
    const candidates=[];
    if(batch<1024){
      for(const sweep of batchData.batchSweeps||[]){
        if(Number(sweep.batchSequences)!==batch)continue;
        const result=sweep.results?.[epochKey(epoch)];
        if(!admissibleBaseline(batch,sweep,result))continue;
        candidates.push({method:`baseline-${batch}`,batchSequences:batch,lr:sweep.lr,wd:sweep.wd,result});
      }
    }else{
      for(const run of wdData.runs||[]){
        if(Number(run.epoch)!==Number(epoch)||run.status!=="complete"||!finite(run.validation))continue;
        if(run.wandb&&healthMap[run.wandb])continue;
        if(lrNumber(run.lr)>=2e-3)continue;
        candidates.push({method:"baseline-1024",batchSequences:1024,lr:run.lr,wd:run.wd,result:run});
      }
    }
    return choose(candidates);
  }

  const selected=new Map();
  for(const batch of batches){
    for(const epoch of epochs)selected.set(`baseline-${batch}:${epochKey(epoch)}`,baselineCandidate(batch,epoch));
  }
  for(const batch of batches){
    const key=`dr${batch}`;
    let wdFloor=-Infinity;
    for(const epoch of epochs){
      if(epoch===1){
        selected.set(`${key}:${epochKey(epoch)}`,baselineCandidate(batch,epoch));
        continue;
      }
      const candidates=[];
      for(const run of study.runs||[]){
        if(run.method!==key||wdNumber(run.wd)<wdFloor)continue;
        const result=run.results?.[epochKey(epoch)];
        if(!result||result.status!=="complete"||!finite(result.validation))continue;
        if(result.wandb&&healthMap[result.wandb])continue;
        candidates.push({method:key,batchSequences:batch,lr:run.lr,wd:run.wd,result});
      }
      const winner=choose(candidates);
      selected.set(`${key}:${epochKey(epoch)}`,winner);
      if(winner)wdFloor=wdNumber(winner.wd);
    }
  }
  const getSelected=(column,epoch)=>selected.get(`${column.key}:${epochKey(epoch)}`);
  const warmdownAtEpoch=(chain,epoch)=>chain.runs.find(run=>Number(run.accumulatedEpoch)===Number(epoch))||null;
  const optimizerStepsForEpoch=(epoch,batch)=>Math.ceil(Number(epoch)*1_000_000_000/(Number(batch)*4096));

  document.getElementById("title").textContent=warmdown.title;
  document.getElementById("setup").textContent=warmdown.setup;
  document.getElementById("updated").textContent=`Updated ${warmdown.updated}`;
  document.getElementById("selection").textContent=warmdown.selection;
  document.getElementById("recalibration").textContent=warmdown.recalibration.description;
  document.getElementById("timing-note").textContent=`Source-report curves preserve every selected Original and DR datapoint and use the report's idealized 0.6 seconds per optimizer step. The purple warmdown chain uses its cumulative one-node timing: ${warmdown.timing.description}`;

  const legend=document.getElementById("legend");
  for(const column of columns.concat(warmdownChains)){
    const label=document.createElement("label");
    label.innerHTML=`<span class="legend-line ${column.baseline?'legend-line-dashed':''}" style="border-color:${column.color};opacity:${column.baseline?.65:1}"></span>${column.label}`;
    legend.appendChild(label);
  }

  const graphSeries=[];
  for(const column of columns){
    const points=epochs.map(epoch=>{
      const winner=getSelected(column,epoch);
      if(!winner)return null;
      return {seconds:optimizerStepsForEpoch(epoch,column.batchSequences)*0.6,value:numeric(winner.result.validation),epoch,winner};
    }).filter(point=>point&&point.value!==null);
    if(points.length)graphSeries.push({column,points});
  }
  for(const chain of warmdownChains){
    const points=chain.runs.filter(run=>finite(run.validation)).map(run=>({seconds:Number(run.idealizedTrainingSeconds),value:Number(run.validation),epoch:Number(run.accumulatedEpoch),run}));
    if(points.length)graphSeries.push({column:chain,points});
  }

  const width=1180,height=430,margin={left:66,right:28,top:28,bottom:62};
  const allPoints=graphSeries.flatMap(series=>series.points);
  const positiveTimes=allPoints.map(point=>point.seconds).filter(value=>value>0);
  const xMin=Math.min(...positiveTimes)*.85,xMax=Math.max(...positiveTimes)*1.12;
  const values=allPoints.map(point=>point.value);
  const yPad=Math.max((Math.max(...values)-Math.min(...values))*.08,.03);
  const yMin=Math.min(...values)-yPad,yMax=Math.max(...values)+yPad;
  const x=value=>margin.left+(Math.log(value)-Math.log(xMin))/(Math.log(xMax)-Math.log(xMin))*(width-margin.left-margin.right);
  const y=value=>margin.top+(yMax-value)/(yMax-yMin)*(height-margin.top-margin.bottom);
  const timeTicks=[60,120,300,600,1200,1800,3600,7200,14400,28800,57600].filter(value=>value>=xMin&&value<=xMax);
  const xGrid=timeTicks.map(value=>`<line x1="${x(value)}" x2="${x(value)}" y1="${margin.top}" y2="${height-margin.bottom}" stroke="#eef1f6"/><text x="${x(value)}" y="${height-32}" text-anchor="middle">${formatDuration(value)}</text>`).join("");
  const yGrid=[0,.25,.5,.75,1].map(f=>yMin+f*(yMax-yMin)).map(value=>`<line x1="${margin.left}" x2="${width-margin.right}" y1="${y(value)}" y2="${y(value)}" stroke="#e2e8f0"/><text x="${margin.left-8}" y="${y(value)+4}" text-anchor="end">${value.toFixed(3)}</text>`).join("");
  const paths=graphSeries.map(series=>{
    const ordered=series.points.slice().sort((a,b)=>a.seconds-b.seconds);
    const isWarmdown=warmdownChains.some(chain=>chain.key===series.column.key);
    const path=ordered.length>1?`<path d="${ordered.map((point,index)=>`${index?'L':'M'} ${x(point.seconds)} ${y(point.value)}`).join(' ')}" fill="none" stroke="${series.column.color}" stroke-width="${isWarmdown?4:series.column.baseline?1.8:2.4}" stroke-opacity="${series.column.baseline?.65:1}"${series.column.baseline?' stroke-dasharray="7 5"':''}/>`:"";
    const dots=ordered.map(point=>`<circle cx="${x(point.seconds)}" cy="${y(point.value)}" r="${isWarmdown?6:series.column.baseline?2.5:4}" fill="${series.column.color}" fill-opacity="${series.column.baseline?.5:1}"><title>${series.column.label} · E${formatEpoch(point.epoch)} · ${formatDuration(point.seconds)} · validation CE ${point.value.toFixed(3)}</title></circle>`).join("");
    return path+dots;
  }).join("");
  document.getElementById("chart").innerHTML=`<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Validation loss versus idealized training time">${yGrid}${xGrid}<line x1="${margin.left}" x2="${width-margin.right}" y1="${height-margin.bottom}" y2="${height-margin.bottom}" stroke="#98a2b3"/>${paths}<text x="${width/2}" y="${height-8}" text-anchor="middle" class="axis-label">Idealized cumulative training time · logarithmic scale</text><text transform="translate(18 ${height/2}) rotate(-90)" text-anchor="middle" class="axis-label">DCLM validation CE · lower is better</text></svg>`;

  const allTableColumns=columns.concat(warmdownChains);
  function installHeader(id,firstColumns){
    document.getElementById(id).innerHTML=`<tr>${firstColumns.map(label=>`<th>${label}</th>`).join("")}${allTableColumns.map(column=>`<th>${column.label}</th>`).join("")}</tr>`;
  }
  installHeader("validation-head",["Epoch"]);
  installHeader("coordinate-head",["Epoch"]);
  installHeader("optimizer-head",["Optimizer steps","Training time"]);

  const validationBody=document.getElementById("validation-summary");
  const coordinateBody=document.getElementById("coordinate-summary");
  const columnBest=new Map();
  for(const column of columns){
    const values=epochs.map(epoch=>numeric(getSelected(column,epoch)?.result.validation)).filter(value=>value!==null);
    if(values.length)columnBest.set(column.key,Math.min(...values));
  }
  for(const chain of warmdownChains){
    const values=chain.runs.map(run=>numeric(run.validation)).filter(value=>value!==null);
    if(values.length)columnBest.set(chain.key,Math.min(...values));
  }
  for(const epoch of epochs){
    const entries=columns.map(column=>({column,winner:getSelected(column,epoch)}));
    for(const chain of warmdownChains){
      const wr=warmdownAtEpoch(chain,epoch);
      entries.push({column:chain,winner:wr?{lr:wr.lr||"1e-3",wd:wr.wd||"0.333",result:wr}:null});
    }
    const rowValues=entries.map(entry=>numeric(entry.winner?.result.validation)).filter(value=>value!==null);
    const rowBest=rowValues.length?Math.min(...rowValues):null;
    const validationCells=entries.map(entry=>{
      const value=numeric(entry.winner?.result.validation);
      if(value===null)return `<td>${entry.winner?.result.status||"—"}</td>`;
      const classes=[];
      if(value===columnBest.get(entry.column.key))classes.push("summary-best");
      if(value===rowBest)classes.push("summary-row-best");
      return `<td class="${classes.join(' ')}">${value.toFixed(3)}</td>`;
    }).join("");
    const coordinateCells=entries.map(entry=>entry.winner?`<td>(${entry.winner.lr}, ${entry.winner.wd})</td>`:"<td>—</td>").join("");
    validationBody.insertAdjacentHTML("beforeend",`<tr><td>E${formatEpoch(epoch)}</td>${validationCells}</tr>`);
    coordinateBody.insertAdjacentHTML("beforeend",`<tr><td>E${formatEpoch(epoch)}</td>${coordinateCells}</tr>`);
  }

  const comparisonByStep=new Map((batchData.optimizerStepComparisons||[]).map(comparison=>[Number(comparison.optimizerSteps),comparison]));
  for(const chain of warmdownChains)for(const run of chain.runs)if(!comparisonByStep.has(Number(run.optimizerStep)))comparisonByStep.set(Number(run.optimizerStep),{optimizerSteps:Number(run.optimizerStep),epochs:{}});
  const optimizerBody=document.getElementById("optimizer-step-summary");
  for(const comparison of [...comparisonByStep.values()].sort((a,b)=>Number(a.optimizerSteps)-Number(b.optimizerSteps))){
    const step=Number(comparison.optimizerSteps);
    const entries=columns.map(column=>{
      const epoch=comparison.epochs?.[String(column.batchSequences)];
      return {column,epoch,winner:epoch===undefined?null:getSelected(column,epoch)};
    });
    const stepRuns=[];
    for(const chain of warmdownChains){
      const wr=chain.runs.find(run=>Number(run.optimizerStep)===step)||null;
      stepRuns.push(wr);
      entries.push({column:chain,epoch:wr?.accumulatedEpoch,winner:wr?{result:wr}:null});
    }
    const values=entries.map(entry=>numeric(entry.winner?.result.validation)).filter(value=>value!==null);
    const rowBest=values.length?Math.min(...values):null;
    const timedRun=stepRuns.find(run=>run&&finite(run.idealizedTrainingSeconds));
    const time=timedRun?Number(timedRun.idealizedTrainingSeconds):step*.6;
    const cells=entries.map(entry=>{
      if(!entry.winner)return "<td>—</td>";
      const value=numeric(entry.winner.result.validation);
      const display=value===null?(entry.winner.result.status||"pending"):`E${formatEpoch(entry.epoch)} · ${value.toFixed(3)}`;
      const best=value!==null&&value===rowBest;
      return `<td class="${best?'summary-best summary-row-best':''}">${best?`<strong>${display}</strong>`:display}</td>`;
    }).join("");
    optimizerBody.insertAdjacentHTML("beforeend",`<tr><td>${step.toLocaleString()}</td><td>≈${formatDuration(time)}</td>${cells}</tr>`);
  }

  const stages=document.getElementById("stage-summary");
  for(const chain of warmdownChains)for(const run of chain.runs){
    const wandb=run.wandb?`<a href="https://wandb.ai/ai2-llm/sewonm-icsl/runs/${run.wandb}">${run.wandb}</a>`:"—";
    const beaker=run.beaker?`<a href="https://beaker.org/orgs/ai2/workspaces/flex2/work/${run.beaker}">${run.beaker}</a>`:"—";
    stages.insertAdjacentHTML("beforeend",`<tr><td>${chain.label}</td><td>${run.stage}</td><td>BS${run.batchSequences}</td><td>E${run.accumulatedEpoch}</td><td>${run.addedSteps.toLocaleString()}</td><td>${run.optimizerStep.toLocaleString()}</td><td>${finite(run.wallClockSeconds)?formatDuration(run.wallClockSeconds):"—"}</td><td>${run.status}</td><td>${formatMetric(run.validation)}</td><td>${wandb}</td><td>${beaker}</td></tr>`);
  }

  const newRuns=warmdownChains.flatMap(chain=>chain.runs.filter(run=>run.stage!=="source").map(run=>({chain,run})));
  const gridBody=document.getElementById("warmdown-coordinate-grid");
  const newRunsBody=document.getElementById("new-runs");
  for(const {chain,run} of newRuns){
    const history=run.stage==="bs256"?"BS1024 E2 → BS256 E4":"BS1024 E2 → BS256 E4 → BS64 E8";
    gridBody.insertAdjacentHTML("beforeend",`<tr><td>E${run.accumulatedEpoch}</td><td>BS${run.batchSequences}</td><td>${chain.label}: ${history}</td><td><span class="tuple">(LR ${run.lr||"1e-3"}, WD ${run.wd||"0.333"})</span></td><td>${run.status}</td></tr>`);
    const wandb=run.wandb?`<a href="https://wandb.ai/ai2-llm/sewonm-icsl/runs/${run.wandb}">${run.wandb}</a>`:"—";
    const beaker=run.beaker?`<a href="https://beaker.org/orgs/ai2/workspaces/flex2/work/${run.beaker}">${run.beaker}</a>`:"—";
    newRunsBody.insertAdjacentHTML("beforeend",`<tr><td>${chain.label}</td><td>BS${run.batchSequences}</td><td>E${run.accumulatedEpoch}</td><td>${run.lr||"1e-3"}</td><td>${run.wd||"0.333"}</td><td>${run.status}</td><td>${formatMetric(run.train)}</td><td>${formatMetric(run.validation)}</td><td>${wandb}</td><td>${beaker}</td></tr>`);
  }
})();
