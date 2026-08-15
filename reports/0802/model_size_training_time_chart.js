(() => {
  const models=[
    {name:'Dense 153M',short:'153M',data:window.ICSL_UNIFIED_BATCH_153M,secondsPerStep:0.267,microbatch:32,color:'#059669'},
    {name:'Dense 474M',short:'474M',data:window.ICSL_UNIFIED_BATCH_474M,secondsPerStep:0.4,microbatch:16,color:'#2563eb'},
    {name:'Dense 1B',short:'1B',data:window.ICSL_UNIFIED_BATCH_1B,baseline:window.ICSL_UNIFIED_WD_1B,secondsPerStep:0.6,microbatch:8,color:'#7c3aed'},
  ];
  const uniquePoolTokens=1_000_000_000;
  const sequenceLength=4096;
  const metric=run=>run.validation??run.c4;
  const wandbId=run=>run.wandb||run.activeWandb;
  const key=(batch,epoch)=>`${batch}|${epoch}`;
  const finite=value=>Number.isFinite(Number(value));
  const complete=run=>run.status==='complete'&&Number.isFinite(metric(run))&&run.kind!=='evaluation';
  const formatNumber=value=>Number(value).toLocaleString('en-US');
  const formatDuration=seconds=>{
    const total=Math.max(0,Math.round(seconds));
    const hours=Math.floor(total/3600);
    const minutes=Math.floor((total%3600)/60);
    const remainder=total%60;
    if(hours)return `${hours}h ${minutes}m ${remainder}s`;
    if(minutes)return `${minutes}m ${remainder}s`;
    return `${remainder}s`;
  };
  const formatLearningRate=value=>{
    const numeric=Number(value);
    if(!Number.isFinite(numeric))return String(value);
    if(numeric===0)return '0';
    const exponent=Math.floor(Math.log10(Math.abs(numeric)));
    const coefficient=numeric/(10**exponent);
    return `${Number(coefficient.toFixed(3))}e${exponent}`;
  };
  const allRunsFor=model=>{
    const runs=(model.data?.batchSweeps||[]).flatMap(sweep=>Object.entries(sweep.results||{}).map(([epoch,result])=>({
      ...sweep,
      ...result,
      batchSequences:Number(sweep.batchSequences),
      epoch:Number(epoch),
      status:result.status,
    })));
    if(model.baseline){
      runs.push(...(model.baseline.runs||[]).map(run=>({...run,batchSequences:1024,epoch:Number(run.epoch)})));
    }
    return runs;
  };
  const admissible=(model,run)=>{
    const lr=Number(run.lr);
    const wd=Number(run.wd);
    if(!Number.isFinite(lr))return false;
    const policy=model.data?.selectionPolicy||{};
    const maxLearningRate=Number(policy.maxLearningRate);
    if(Number.isFinite(maxLearningRate)&&lr>maxLearningRate)return false;
    if(policy.allowAllCompletedCoordinates)return true;
    if(lr<2e-3)return true;
    return run.batchSequences===256&&lr===2e-3&&wd===0.333;
  };
  const selectionFor=model=>{
    const unhealthy={...(model.baseline?.healthAudit?.unhealthy||{}),...(model.data?.healthAudit?.unhealthy||{})};
    const candidates=allRunsFor(model).filter(complete).filter(run=>admissible(model,run)).filter(run=>!unhealthy[wandbId(run)]);
    const selected=new Map();
    const batches=model.data?.summaryBatches||[16,32,64,128,256,512,1024];
    if(model.data?.selectionPolicy?.nondecreasingWd){
      batches.forEach(batch=>{
        let wdFloor=-Infinity;
        const epochs=[...new Set(candidates.filter(run=>run.batchSequences===batch).map(run=>Number(run.epoch)))].sort((a,b)=>a-b);
        epochs.forEach(epoch=>{
          const eligible=candidates.filter(run=>run.batchSequences===batch&&Number(run.epoch)===epoch&&Number(run.wd)>=wdFloor);
          if(!eligible.length)return;
          const best=eligible.reduce((current,run)=>metric(run)<metric(current)||(metric(run)===metric(current)&&Number(run.wd)<Number(current.wd))?run:current);
          selected.set(key(batch,epoch),best);
          wdFloor=Number(best.wd);
        });
      });
    }else{
      candidates.forEach(run=>{
        const runKey=key(run.batchSequences,run.epoch);
        const current=selected.get(runKey);
        if(!current||metric(run)<metric(current))selected.set(runKey,run);
      });
    }
    const stoppedImproving=new Map();
    batches.forEach(batch=>{
      const completed=[...selected.values()].filter(run=>run.batchSequences===batch).sort((a,b)=>Number(a.epoch)-Number(b.epoch));
      if(completed.length<2)return;
      const terminal=completed.at(-1);
      const priorBest=completed.slice(0,-1).reduce((best,run)=>metric(run)<metric(best)?run:best);
      const activeAtOrBeyond=(model.data?.batchSweeps||[]).some(sweep=>sweep.batchSequences===batch&&['active','running','scheduled','submitted','pending','queued','planned'].includes(sweep.status)&&Number(sweep.activeEpoch)>=Number(terminal.epoch));
      if(!activeAtOrBeyond&&metric(terminal)>=metric(priorBest))stoppedImproving.set(batch,{terminal,priorBest});
    });
    return {selected,stoppedImproving};
  };
  const stepsForComparison=(model,comparison)=>{
    const batches=model.data?.summaryBatches||[16,32,64,128,256,512,1024];
    const recalculated=batches.map(batch=>{
      const epoch=Number(comparison.epochs?.[String(batch)]);
      return Number.isFinite(epoch)?Math.round(epoch*uniquePoolTokens/(batch*sequenceLength)):null;
    }).filter(Number.isFinite);
    return recalculated.length?Math.round(recalculated.reduce((sum,value)=>sum+value,0)/recalculated.length):Number(comparison.optimizerSteps);
  };
  const pointsFor=model=>{
    const {selected,stoppedImproving}=selectionFor(model);
    const batches=model.data?.summaryBatches||[16,32,64,128,256,512,1024];
    return (model.data?.optimizerStepComparisons||[]).map(comparison=>{
      const steps=stepsForComparison(model,comparison);
      const rowEntries=batches.map(batch=>{
        const epoch=Number(comparison.epochs?.[String(batch)]);
        if(!Number.isFinite(epoch))return null;
        const run=selected.get(key(batch,epoch));
        const stopped=stoppedImproving.get(batch);
        if(stopped&&epoch>Number(stopped.priorBest.epoch))return {run:stopped.priorBest,replacedRun:run,value:metric(stopped.priorBest),epoch,carried:true,sourceEpoch:Number(stopped.priorBest.epoch),stopEpoch:Number(stopped.terminal.epoch)};
        return run?{run,value:metric(run),epoch,carried:false,sourceEpoch:Number(run.epoch)}:null;
      }).filter(Boolean);
      if(!rowEntries.length)return null;
      const winner=rowEntries.reduce((best,entry)=>entry.value<best.value?entry:best);
      return {
        model:model.name,
        short:model.short,
        color:model.color,
        secondsPerStep:model.secondsPerStep,
        microbatch:model.microbatch,
        steps,
        seconds:steps*model.secondsPerStep,
        validation:winner.value,
        batch:winner.run.batchSequences,
        epoch:winner.epoch,
        lr:winner.run.lr,
        wd:winner.run.wd,
        carried:winner.carried,
        sourceEpoch:winner.sourceEpoch,
        stopEpoch:winner.stopEpoch,
      };
    }).filter(Boolean);
  };

  models.forEach(model=>{model.points=pointsFor(model);});
  const allPoints=models.flatMap(model=>model.points);
  const assumptions=document.querySelector('#assumptions');
  assumptions.innerHTML=models.map(model=>`<div class="assumption"><strong style="color:${model.color}">${model.name}</strong><span>Microbatch ${model.microbatch} · ${model.secondsPerStep}s per optimizer step</span></div>`).join('');
  document.querySelector('#legend').innerHTML=models.map(model=>`<div class="legend-item"><span class="legend-line" style="background:${model.color}"></span><strong>${model.name}</strong></div>`).join('');
  document.querySelector('#updated').textContent=`Data loaded from the three live Step 1-1 reports · ${new Date().toLocaleString('en-US',{dateStyle:'medium',timeStyle:'short'})}`;

  const modelCards=document.querySelector('#model-cards');
  modelCards.innerHTML=models.map(model=>{
    const best=model.points.reduce((current,point)=>!current||point.validation<current.validation?point:current,null);
    const first=model.points[0],last=model.points.at(-1);
    return `<div class="model-card" style="--series-color:${model.color}"><strong>${model.name} · best ${best?best.validation.toFixed(3):'pending'}</strong><span>${model.points.length} matched-step points · ${first?formatDuration(first.seconds):'—'} to ${last?formatDuration(last.seconds):'—'}</span></div>`;
  }).join('');

  const results=document.querySelector('#results');
  results.innerHTML=models.flatMap(model=>model.points.map(point=>`<tr class="${point.carried?'stopped-row':''}"${point.carried?` title="E${point.sourceEpoch} is the real source epoch, carried into the missing E${point.epoch} matched-step slot after terminal E${point.stopEpoch} non-improvement."`:''}><td class="model-name" style="color:${point.color}">${point.model}</td><td>≈${formatDuration(point.seconds)}</td><td>${formatNumber(point.steps)}</td><td>BS${point.batch} / E${point.carried?point.sourceEpoch:point.epoch}</td><td>${formatLearningRate(point.lr)}</td><td>${point.wd}</td><td><strong>${point.validation.toFixed(3)}</strong></td></tr>`)).join('');

  const svg=document.querySelector('#plot');
  const width=1080,height=620;
  const margin={top:32,right:94,bottom:84,left:76};
  const innerWidth=width-margin.left-margin.right;
  const innerHeight=height-margin.top-margin.bottom;
  const losses=allPoints.map(point=>point.validation);
  const rawMinLoss=Math.min(...losses),rawMaxLoss=Math.max(...losses);
  const lossPadding=Math.max((rawMaxLoss-rawMinLoss)*0.08,0.02);
  const minLoss=Math.floor((rawMinLoss-lossPadding)*20)/20;
  const maxLoss=Math.ceil((rawMaxLoss+lossPadding)*20)/20;
  const y=value=>margin.top+(maxLoss-value)/(maxLoss-minLoss)*innerHeight;
  const durationTicks=[60,120,300,600,1800,3600,7200,14400,28800,72000,144000];
  const escapeXml=value=>String(value).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
  let scale='log';
  const draw=()=>{
    const seconds=allPoints.map(point=>point.seconds);
    const rawMinTime=Math.min(...seconds),rawMaxTime=Math.max(...seconds);
    const minTime=scale==='log'?rawMinTime:0;
    const maxTime=rawMaxTime;
    const transform=value=>scale==='log'?Math.log10(value):value;
    const transformedMin=transform(minTime),transformedMax=transform(maxTime);
    const x=value=>margin.left+(transform(value)-transformedMin)/(transformedMax-transformedMin)*innerWidth;
    const xTicks=scale==='log'
      ? durationTicks.filter(value=>value>=minTime*.95&&value<=maxTime*1.05)
      : Array.from({length:7},(_,index)=>maxTime*index/6);
    const yTickCount=6;
    const yTicks=Array.from({length:yTickCount},(_,index)=>minLoss+(maxLoss-minLoss)*index/(yTickCount-1));
    const xGrid=xTicks.map(value=>`<line x1="${x(value)}" y1="${margin.top}" x2="${x(value)}" y2="${margin.top+innerHeight}" stroke="var(--grid)"/><text class="tick" x="${x(value)}" y="${height-margin.bottom+24}" text-anchor="middle">${formatDuration(value)}</text>`).join('');
    const yGrid=yTicks.map(value=>`<line x1="${margin.left}" y1="${y(value)}" x2="${margin.left+innerWidth}" y2="${y(value)}" stroke="var(--grid)"/><text class="tick" x="${margin.left-12}" y="${y(value)+4}" text-anchor="end">${value.toFixed(2)}</text>`).join('');
    const series=models.map(model=>{
      const sorted=[...model.points].sort((a,b)=>a.seconds-b.seconds);
      const path=sorted.length>1?`<path d="${sorted.map((point,index)=>`${index?'L':'M'} ${x(point.seconds)} ${y(point.validation)}`).join(' ')}" fill="none" stroke="${model.color}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>`:'';
      const points=sorted.map(point=>{
        const carryNote=point.carried?`\nE${point.sourceEpoch} is the real source epoch, carried into the missing E${point.epoch} matched-step slot after terminal E${point.stopEpoch} non-improvement.`:'';
        const title=`${point.model}\nTraining time ≈${formatDuration(point.seconds)}\n${formatNumber(point.steps)} optimizer steps\nWinner: BS${point.batch} / E${point.carried?point.sourceEpoch:point.epoch}\nLR ${formatLearningRate(point.lr)} / WD ${point.wd}\nDCLM validation CE ${point.validation.toFixed(3)}${carryNote}`;
        return `<circle class="point" cx="${x(point.seconds)}" cy="${y(point.validation)}" r="5" fill="${model.color}" tabindex="0"><title>${escapeXml(title)}</title></circle>`;
      }).join('');
      const last=sorted.at(-1);
      const label=last?`<text class="series-label" x="${x(last.seconds)+9}" y="${y(last.validation)+4}" fill="${model.color}">${model.short}</text>`:'';
      return path+points+label;
    }).join('');
    svg.innerHTML=`<title id="chart-title">Best DCLM validation loss versus idealized training time</title><desc id="chart-description">Three lines compare Dense 153M, Dense 474M, and Dense 1B using the best available validation loss at each matched optimizer-step row.</desc>${xGrid}${yGrid}<line x1="${margin.left}" y1="${margin.top+innerHeight}" x2="${margin.left+innerWidth}" y2="${margin.top+innerHeight}" stroke="#98a2b3"/><line x1="${margin.left}" y1="${margin.top}" x2="${margin.left}" y2="${margin.top+innerHeight}" stroke="#98a2b3"/>${series}<text class="axis-label" x="${margin.left+innerWidth/2}" y="${height-22}" text-anchor="middle">Idealized training time (${scale} scale)</text><text class="axis-label" transform="translate(20 ${margin.top+innerHeight/2}) rotate(-90)" text-anchor="middle">DCLM validation CE · lower is better</text>`;
  };
  document.querySelectorAll('[data-scale]').forEach(button=>button.addEventListener('click',()=>{
    scale=button.dataset.scale;
    document.querySelectorAll('[data-scale]').forEach(candidate=>candidate.classList.toggle('active',candidate===button));
    draw();
  }));
  draw();
})();
