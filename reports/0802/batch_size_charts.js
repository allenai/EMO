(() => {
  const d=window.ICSL_REPORT_DATA||{batchSweeps:[],targetEpochs:[]};
  const simulation=window.ICSL_BATCH_SIMULATION_DATA||{columns:[],runs:[]};
  const dataLoader=window.ICSL_DATA_LOADER_DATA||{runs:[]};
  const showDataLoaderColumns=(dataLoader.runs||[]).length>0;
  const adaptiveChains=d.adaptiveDrWtEmbedWdChains||[];
  const lockedWdPhasePolicies=new Set([
    'locked_wd_predecay_saturation_v1',
    'locked_wd_requested_postdecay_finalizer_v1',
    'locked_wd_all_postdecay_saturation_v1',
  ]);
  const adaptiveBatches=new Set(adaptiveChains.map(chain=>Number(chain.batchSequences)));
  const showAdaptiveColumns=adaptiveChains.length>0;
  const healthStyles=document.createElement('style');
  healthStyles.textContent=`
    .health-key{display:inline-block;width:14px;height:14px;border:0;border-radius:4px;background:#fecaca;vertical-align:-2px}
    .health-key.suspicious{background:#fb923c}
    .tuple.unhealthy{background:#fff1f2!important;font-weight:400!important}
    .tuple.suspicious{background:#fed7aa!important;color:#9a3412!important;font-weight:400!important}
    .run-unhealthy td{background:#fff1f2;font-weight:400}
    .run-suspicious td{background:#ffedd5;font-weight:400}
    .matched-stopped{color:#7e22ce}
    .matched-stopped-key{color:#7e22ce;font-weight:700}
    .pd-value{color:#64748b;font-weight:400}
    .phase-separator{color:#94a3b8;font-weight:400}
  `;
  document.head.append(healthStyles);
  const baseline=window.ICSL_WD_BASELINE_DATA||{runs:[],fixedLrByEpoch:{}};
  const optimizerTiming=window.ICSL_OPTIMIZER_STEP_TIMING||{};
  document.querySelector('#title').textContent=d.title;
  document.querySelector('#setup').textContent=d.setup;
  document.querySelector('#updated').textContent=`Updated ${d.updated}`;
  document.querySelector('#selection').textContent=d.selection;

  const colors={'BS 16':'#059669','BS 32':'#0d9488','BS 64':'#2563eb','BS 128':'#0284c7','BS 256':'#7c3aed','BS 512':'#c026d3','BS 1024':'#dc5a39'};
  const summaryBatches=d.summaryBatches||[16,32,64,128,256,512,1024];
  const series=summaryBatches.map(batch=>`BS ${batch}`);
  const health={...(baseline.healthAudit?.unhealthy||{}),...(d.healthAudit?.unhealthy||{}),...(simulation.healthAudit?.unhealthy||{}),...(dataLoader.healthAudit?.unhealthy||{})};
  const suspiciousHealth={...(baseline.healthAudit?.suspicious||{}),...(d.healthAudit?.suspicious||{}),...(simulation.healthAudit?.suspicious||{}),...(dataLoader.healthAudit?.suspicious||{})};
  const wandbId=r=>r.wandb||r.activeWandb;
  const healthRecord=r=>health[wandbId(r)];
  const suspiciousRecord=r=>suspiciousHealth[wandbId(r)];
  const unhealthy=r=>Boolean(healthRecord(r));
  const suspicious=r=>Boolean(suspiciousRecord(r)&&!unhealthy(r));
  const statusRecord=r=>healthRecord(r)||suspiciousRecord(r);
  const escapeAttribute=value=>String(value||'').replaceAll('&','&amp;').replaceAll('"','&quot;');
  const visible=r=>(!['failed','canceled'].includes(r.status)||unhealthy(r)||suspicious(r))&&r.kind!=='evaluation';
  const active=r=>['active','running','scheduled','submitted','pending','queued','planned'].includes(r.status);
  const metric=r=>r.validation??r.c4;
  const complete=r=>r.status==='complete'&&Number.isFinite(metric(r));
  const admissibleCoordinate=r=>{
    const lr=Number(r.lr),wd=Number(r.wd);
    if(!Number.isFinite(lr))return false;
    const maxLearningRate=Number(d.selectionPolicy?.maxLearningRate);
    if(Number.isFinite(maxLearningRate)&&lr>maxLearningRate)return false;
    if(d.selectionPolicy?.allowAllCompletedCoordinates)return true;
    if(lr<2e-3)return true;
    return [256,512].includes(r.batchSequences)&&lr===2e-3&&wd===0.333;
  };
  const key=r=>`${r.batchSequences}|${r.epoch}`;

  const targetsForBatch=batch=>{
    const configured=d.batchTargetEpochs?.[String(batch)]||[];
    const represented=(d.batchSweeps||[]).filter(sweep=>sweep.batchSequences===batch).flatMap(sweep=>[
      ...Object.keys(sweep.results||{}).map(Number),
      Number(sweep.activeEpoch),
    ]);
    const baselineRepresented=batch===1024?(baseline.runs||[]).map(run=>Number(run.epoch)):[];
    return [...new Set([...configured,...represented,...baselineRepresented].filter(Number.isFinite))].sort((a,b)=>a-b);
  };
  const matchedEpochsForBatch=batch=>(d.optimizerStepComparisons||[]).map(comparison=>comparison.epochs?.[String(batch)]).filter(Number.isFinite);
  const optimizerStepsForEpochBatch=(epoch,batch)=>{
    const uniquePoolTokens=Number(optimizerTiming.uniquePoolTokens);
    const sequenceLength=Number(optimizerTiming.sequenceLength)||4096;
    if(!Number.isFinite(uniquePoolTokens)||uniquePoolTokens<=0||!Number.isFinite(epoch)||!Number.isFinite(batch)||batch<=0)return null;
    return Math.round(epoch*uniquePoolTokens/(batch*sequenceLength));
  };
  const optimizerStepsForComparison=comparison=>{
    const recalculated=summaryBatches.map(batch=>optimizerStepsForEpochBatch(comparison.epochs?.[String(batch)],batch)).filter(Number.isFinite);
    if(recalculated.length)return Math.round(recalculated.reduce((sum,steps)=>sum+steps,0)/recalculated.length);
    return Number(comparison.optimizerSteps);
  };
  const customSimulationOptimizerSteps=(simulation.columns||[]).flatMap(column=>
    Object.values(column.matchedOptimizerStepsByEpoch||{}).map(Number).filter(Number.isFinite)
  );
  const optimizerStepComparisons=[...(d.optimizerStepComparisons||[])];
  customSimulationOptimizerSteps.forEach(optimizerSteps=>{
    if(!optimizerStepComparisons.some(comparison=>optimizerStepsForComparison(comparison)===optimizerSteps)){
      optimizerStepComparisons.push({optimizerSteps,epochs:{},simulationOnly:true});
    }
  });
  optimizerStepComparisons.sort((a,b)=>optimizerStepsForComparison(a)-optimizerStepsForComparison(b));
  const optimizerStepRows=optimizerStepComparisons.map(optimizerStepsForComparison).filter(Number.isFinite);
  const endpointOptimizerStep=run=>{
    const explicit=Number(run.cumulativeOptimizerSteps??run.endpointOptimizerStep);
    if(Number.isFinite(explicit))return explicit;
    const match=String(run.endpointCheckpoint||'').match(/\/step(\d+)\/?$/);
    return match?Number(match[1]):null;
  };
  const closestOptimizerStepRow=steps=>optimizerStepRows.reduce((closest,row)=>
    closest===null||Math.abs(Math.log(steps/row))<Math.abs(Math.log(steps/closest))?row:closest
  ,null);
  const formatDuration=seconds=>{
    if(!Number.isFinite(seconds))return '—';
    const totalSeconds=Math.max(0,Math.round(seconds));
    const hours=Math.floor(totalSeconds/3600);
    const minutes=Math.floor((totalSeconds%3600)/60);
    const remainingSeconds=totalSeconds%60;
    if(hours)return `${hours}h ${minutes}m ${remainingSeconds}s`;
    if(minutes)return `${minutes}m ${remainingSeconds}s`;
    return `${remainingSeconds}s`;
  };
  const newRuns=(d.batchSweeps||[]).flatMap(sweep=>targetsForBatch(sweep.batchSequences).map(epoch=>{
    const result=sweep.results?.[epoch];
    if(result)return {...sweep,...result,epoch,series:`BS ${sweep.batchSequences}`,status:result.status};
    const candidate={...sweep,wandb:sweep.activeWandb,epoch,series:`BS ${sweep.batchSequences}`};
    if(epoch===sweep.activeEpoch&&(active(sweep)||unhealthy(candidate)))return candidate;
    return null;
  })).filter(Boolean).filter(visible);
  const attemptHistoryRuns=(d.batchSweeps||[]).flatMap(sweep=>(sweep.attemptHistory||[]).map(attempt=>({
    ...sweep,...attempt,epoch:sweep.activeEpoch,series:`BS ${sweep.batchSequences}`,historicalAttempt:true,
  })));
  const baselineRuns=summaryBatches.includes(1024)?(baseline.runs||[]).filter(r=>
    visible(r)&&r.status!=='queued'&&d.targetEpochs.includes(r.epoch)
  ).map(r=>({...r,batchSequences:1024,contextLength:4096,series:'BS 1024'})):[];
  const matchedBaselineRuns=summaryBatches.includes(1024)?(baseline.runs||[]).filter(r=>
    visible(r)&&r.status!=='queued'&&matchedEpochsForBatch(1024).includes(r.epoch)
  ).map(r=>({...r,batchSequences:1024,contextLength:4096,series:'BS 1024'})):[];
  const coordinateRuns=[...newRuns,...baselineRuns];
  const selectionRuns=[...coordinateRuns,...matchedBaselineRuns];
  const selected=new Map();
  const selectable=selectionRuns.filter(complete).filter(admissibleCoordinate).filter(r=>!unhealthy(r));
  if(d.selectionPolicy?.nondecreasingWd){
    summaryBatches.forEach(batch=>{
      let wdFloor=-Infinity;
      targetsForBatch(batch).forEach(epoch=>{
        const candidates=selectable.filter(r=>r.batchSequences===batch&&Number(r.epoch)===epoch&&Number(r.wd)>=wdFloor);
        if(!candidates.length)return;
        const override=d.selectionPolicy?.selectedCoordinateOverrides?.[String(batch)]?.[String(epoch)];
        const overrideCandidate=override?candidates.find(r=>Number(r.lr)===Number(override.lr)&&Number(r.wd)===Number(override.wd)):null;
        const best=overrideCandidate||candidates.reduce((current,r)=>metric(r)<metric(current)||(metric(r)===metric(current)&&Number(r.wd)<Number(current.wd))?r:current);
        selected.set(`${batch}|${epoch}`,best);
        wdFloor=Number(best.wd);
      });
    });
  }else{
    selectable.forEach(r=>{
      const current=selected.get(key(r));
      if(!current||metric(r)<metric(current))selected.set(key(r),r);
    });
  }
  const simulationRuns=(simulation.runs||[]).flatMap(sweep=>Object.entries(sweep.results||{}).map(([epoch,result])=>({
    ...sweep,...result,
    train:result.train??result.trainCe,
    validation:result.validation??result.validationCe,
    acc:result.acc??result.hellaswagAccuracy,
    bpb:result.bpb??result.hellaswagBpb,
    epoch:Number(epoch),status:result.status||sweep.status,series:simulation.columns?.find(column=>column.key===sweep.method)?.label||sweep.method,
  })));
  const simulationSelected=new Map();
  simulationRuns.filter(complete).filter(run=>!unhealthy(run)).forEach(run=>{
    const simulationKey=`${run.method}|${run.epoch}`;
    const current=simulationSelected.get(simulationKey);
    if(!current||metric(run)<metric(current))simulationSelected.set(simulationKey,run);
  });
  const drSelected=new Map();
  if(showDataLoaderColumns)summaryBatches.forEach(batch=>{
    let wdFloor=-Infinity;
    (dataLoader.targetEpochs||d.targetEpochs||[]).map(Number).sort((a,b)=>a-b).forEach(epoch=>{
      if(epoch===1){
        const baselineRun=selected.get(`${batch}|${epoch}`);
        if(baselineRun)drSelected.set(`${batch}|${epoch}`,baselineRun);
        return;
      }
      const candidates=(dataLoader.runs||[]).filter(sweep=>
        sweep.method===`dr${batch}`&&Number(sweep.wd)>=wdFloor
      ).map(sweep=>{
        const result=sweep.results?.[String(epoch)];
        return result?{...sweep,...result,epoch,status:result.status||sweep.status}:null;
      }).filter(run=>run&&complete(run)&&!unhealthy(run));
      if(!candidates.length)return;
      const winner=candidates.reduce((best,run)=>
        metric(run)<metric(best)||metric(run)===metric(best)&&Number(run.wd)<Number(best.wd)||
        metric(run)===metric(best)&&Number(run.wd)===Number(best.wd)&&Number(run.lr)<Number(best.lr)?run:best
      );
      drSelected.set(`${batch}|${epoch}`,winner);
      wdFloor=Number(winner.wd);
    });
  });
  const adaptiveSelected=new Map();
  const adaptivePreDecaySelected=new Map();
  adaptiveChains.forEach(chain=>{
    Object.entries(chain.frontiers||{}).forEach(([epoch,frontier])=>{
      const wd=String(frontier.selectedWd);
      const result=chain.results?.[String(epoch)]?.[wd];
      if(!result||!Number.isFinite(metric(result)))return;
      adaptiveSelected.set(`${chain.batchSequences}|${Number(epoch)}`,{
        ...chain,...result,
        epoch:Number(epoch),wd,lr:chain.lr,
        status:result.status||'complete',
        series:`BS ${chain.batchSequences} · DR+WT+EmbedWD`,
      });
    });
    if(lockedWdPhasePolicies.has(chain.policy)){
      Object.entries(chain.preDecayResults||{}).forEach(([epoch,result])=>{
        if(result.status!=='complete'||result.comparisonGroup!=='pre_decay'||!Number.isFinite(metric(result)))return;
        adaptivePreDecaySelected.set(`${chain.batchSequences}|${Number(epoch)}`,{
          ...chain,...result,
          epoch:Number(epoch),wd:String(chain.lockedWd),lr:chain.lr,
          status:'complete',
          series:`BS ${chain.batchSequences} · DR+WT+EmbedWD`,
        });
      });
      Object.entries(chain.postDecayResults||{}).forEach(([epoch,result])=>{
        if(result.status!=='complete'||result.comparisonGroup!=='post_decay'||!Number.isFinite(metric(result)))return;
        adaptiveSelected.set(`${chain.batchSequences}|${Number(epoch)}`,{
          ...chain,...result,
          epoch:Number(epoch),wd:String(chain.lockedWd),lr:chain.lr,
          status:'complete',
          series:`BS ${chain.batchSequences} · DR+WT+EmbedWD`,
        });
      });
    }
  });
  const methodColumnsForBatch=batch=>{
    const columns=[{batch,kind:'original',label:`BS ${batch} · Original`}];
    if(adaptiveBatches.has(batch))columns.push({batch,kind:'adaptive',label:`BS ${batch} · DR+WT+EmbedWD`});
    if(showDataLoaderColumns)columns.push({batch,kind:'dr',label:`BS ${batch} · DR`});
    return columns;
  };
  const summaryMethodColumns=summaryBatches.flatMap(methodColumnsForBatch);
  const runForSummaryColumn=(column,epoch)=>{
    if(column.kind==='adaptive')return adaptiveSelected.get(`${column.batch}|${Number(epoch)}`);
    if(column.kind==='dr')return drSelected.get(`${column.batch}|${Number(epoch)}`);
    return selected.get(`${column.batch}|${Number(epoch)}`);
  };
  const preDecayForSummaryColumn=(column,epoch)=>column.kind==='adaptive'
    ?adaptivePreDecaySelected.get(`${column.batch}|${Number(epoch)}`)
    :null;
  const validationOptimizerStepComparisons=(()=>{
    const comparisonsByStep=new Map();
    const addComparisonEpoch=(batch,epoch)=>{
      const optimizerSteps=optimizerStepsForEpochBatch(Number(epoch),Number(batch));
      if(!Number.isFinite(optimizerSteps))return;
      const comparison=comparisonsByStep.get(optimizerSteps)||{optimizerSteps,epochs:{}};
      comparison.epochs[String(batch)]=Number(epoch);
      comparisonsByStep.set(optimizerSteps,comparison);
    };
    optimizerStepComparisons.forEach(comparison=>{
      const optimizerSteps=optimizerStepsForComparison(comparison);
      if(!Number.isFinite(optimizerSteps))return;
      const merged=comparisonsByStep.get(optimizerSteps)||{...comparison,optimizerSteps,epochs:{}};
      merged.epochs={...merged.epochs,...comparison.epochs};
      comparisonsByStep.set(optimizerSteps,merged);
    });
    [selected,drSelected,adaptiveSelected,adaptivePreDecaySelected].forEach(results=>{
      results.forEach(run=>{
        if(!summaryBatches.includes(Number(run.batchSequences))||!Number.isFinite(metric(run)))return;
        addComparisonEpoch(run.batchSequences,run.epoch);
      });
    });
    return [...comparisonsByStep.values()].sort((a,b)=>optimizerStepsForComparison(a)-optimizerStepsForComparison(b));
  })();
  const setAdaptiveSummaryHeader=(bodyId,leadingLabel,{trainingTime=false}={})=>{
    if(!showAdaptiveColumns)return;
    const table=document.querySelector(`#${bodyId}`)?.closest('table');
    const row=table?.querySelector('thead tr');
    if(!row)return;
    table.classList.add('batch-grouped-table');
    if(trainingTime)table.classList.add('matched-step-table');
    const methodHeaders=summaryMethodColumns.map(column=>`<th>${column.label}</th>`).join('');
    const simulationHeaders=(simulation.columns||[]).map(column=>`<th>${column.label}</th>`).join('');
    row.innerHTML=`<th>${leadingLabel}</th>${trainingTime?'<th class="training-time-header">Training time</th>':''}${methodHeaders}${simulationHeaders}`;
  };
  setAdaptiveSummaryHeader('validation-summary','Epoch');
  setAdaptiveSummaryHeader('optimizer-step-summary','Optimizer steps',{trainingTime:true});
  setAdaptiveSummaryHeader('epoch-hs-accuracy-summary','Epoch');
  setAdaptiveSummaryHeader('epoch-hs-bpb-summary','Epoch');
  setAdaptiveSummaryHeader('epoch-avg8-accuracy-summary','Epoch');
  setAdaptiveSummaryHeader('epoch-avg8-bpb-summary','Epoch');
  setAdaptiveSummaryHeader('coordinate-summary','Epoch');
  setAdaptiveSummaryHeader('optimizer-step-hs-accuracy-summary','Optimizer steps',{trainingTime:true});
  setAdaptiveSummaryHeader('optimizer-step-hs-bpb-summary','Optimizer steps',{trainingTime:true});
  setAdaptiveSummaryHeader('optimizer-step-avg8-accuracy-summary','Optimizer steps',{trainingTime:true});
  setAdaptiveSummaryHeader('optimizer-step-avg8-bpb-summary','Optimizer steps',{trainingTime:true});
  const simulationRunAtMatchedSteps=(column,optimizerSteps,comparison)=>{
    const explicitMappings=Object.entries(column.matchedOptimizerStepsByEpoch||{});
    const explicitEpoch=explicitMappings.find(([,steps])=>Number(steps)===optimizerSteps)?.[0];
    if(explicitMappings.length)return explicitEpoch!==undefined?simulationSelected.get(`${column.key}|${Number(explicitEpoch)}`)||null:null;
    if(!Number.isFinite(Number(column.sourceBatchSequences))){
      const globalBatch=Number(column.globalBatchSequences)||512;
      const matchedEpoch=comparison.epochs?.[String(globalBatch)];
      return Number.isFinite(matchedEpoch)?simulationSelected.get(`${column.key}|${matchedEpoch}`)||null:null;
    }
    const candidates=[...simulationSelected.values()].filter(run=>{
      if(run.method!==column.key)return false;
      const endpointStep=endpointOptimizerStep(run);
      return Number.isFinite(endpointStep)&&closestOptimizerStepRow(endpointStep)===optimizerSteps;
    });
    return candidates.length?candidates.reduce((best,run)=>metric(run)<metric(best)?run:best):null;
  };
  const stoppedImprovingByBatch=new Map();
  summaryBatches.forEach(batch=>{
    const completed=[...selected.values()].filter(run=>run.batchSequences===batch).sort((a,b)=>Number(a.epoch)-Number(b.epoch));
    if(completed.length<2)return;
    const terminal=completed.at(-1);
    const priorBest=completed.slice(0,-1).reduce((best,run)=>metric(run)<metric(best)?run:best);
    const activeAtOrBeyond=(d.batchSweeps||[]).some(sweep=>sweep.batchSequences===batch&&active(sweep)&&Number(sweep.activeEpoch)>=Number(terminal.epoch));
    if(!activeAtOrBeyond&&metric(terminal)>=metric(priorBest))stoppedImprovingByBatch.set(batch,{terminal,priorBest});
  });
  const matchedEntryFor=(batch,epoch)=>{
    const run=selected.get(`${batch}|${epoch}`);
    const stopped=stoppedImprovingByBatch.get(batch);
    if(stopped&&Number(epoch)>Number(stopped.priorBest.epoch))return {run:stopped.priorBest,replacedRun:run,value:metric(stopped.priorBest),carried:true,sourceEpoch:Number(stopped.priorBest.epoch),stopEpoch:Number(stopped.terminal.epoch)};
    if(run)return {run,value:metric(run),carried:false,sourceEpoch:Number(run.epoch)};
    return null;
  };
  const matchedSummaryEntryFor=(column,epoch)=>{
    if(column.kind==='original')return matchedEntryFor(column.batch,epoch);
    const run=runForSummaryColumn(column,epoch);
    return run?{run,value:metric(run),carried:false,sourceEpoch:Number(run.epoch)}:null;
  };
  const coordinateKeys=new Set(coordinateRuns.map(key));
  const chartRuns=[...selected.values()].filter(run=>coordinateKeys.has(key(run)));
  const chartEpochs=chartRuns.map(run=>Number(run.epoch)).filter(Number.isFinite);
  const minChartEpoch=chartEpochs.length?Math.min(...chartEpochs):1;
  const maxChartEpoch=chartEpochs.length?Math.max(...chartEpochs):24;

  const legend=document.querySelector('#legend');
  series.forEach(s=>legend.insertAdjacentHTML('beforeend',`<label><span class="dot" style="background:${colors[s]}"></span>${s}</label>`));
  if(Object.keys(suspiciousHealth).length)legend.insertAdjacentHTML('beforeend','<label><span class="health-key suspicious"></span>Suspicious / monitor</label>');
  legend.insertAdjacentHTML('beforeend','<label><span class="health-key unhealthy"></span>Unhealthy / policy-inadmissible</label>');
  const hellaswagAcc=r=>r.acc??r.hellaswagAccuracy??r.downstream?.hellaswag??null;
  const hellaswagBpb=r=>r.bpb??r.hellaswagBpb??r.downstreamBpb?.hellaswag??null;
  const downstreamTasks=['arc_challenge','arc_easy','csqa','hellaswag','openbookqa','piqa','socialiqa','winogrande'];
  const averageCompleteMetrics=metrics=>{
    const completeMetrics=metrics.filter(Number.isFinite);
    return completeMetrics.length===8?completeMetrics.reduce((sum,metric)=>sum+metric,0)/8:null;
  };
  const avg8Acc=r=>averageCompleteMetrics(downstreamTasks.map(task=>task==='hellaswag'?hellaswagAcc(r):r.downstream?.[task]));
  const avg8Bpb=r=>r.avg8Bpb??averageCompleteMetrics(downstreamTasks.map(task=>task==='hellaswag'?hellaswagBpb(r):r.downstreamBpb?.[task]))??baseline.avg8BpbByWandb?.[r.wandb]??null;
  const value=(r,k)=>{
    if(k==='validation')return metric(r);
    if(k==='acc')return hellaswagAcc(r);
    if(k==='bpb')return hellaswagBpb(r);
    if(k==='avg8_acc')return avg8Acc(r);
    if(k==='avg8_bpb')return avg8Bpb(r);
    return r[k];
  };
  const metrics=[['validation','DCLM validation CE'],['bpb','HellaSwag non-v2 BPB'],['acc','HellaSwag length-normalized accuracy'],['avg8_bpb','8-task average BPB · no BoolQ'],['avg8_acc','8-task average accuracy · no BoolQ']];
  const higherIsBetter=name=>['acc','avg8_acc'].includes(name);
  const bestPoint=(points,name)=>points.reduce((best,point)=>(higherIsBetter(name)?point.v>best.v:point.v<best.v)?point:best);
  const layoutBestLabels=(svg,bounds)=>{
    const items=[...svg.querySelectorAll('text.series-best')].map((node,index)=>{const pointX=Number(node.dataset.pointX),pointY=Number(node.dataset.pointY);node.setAttribute('x',pointX);node.setAttribute('y',pointY);const box=node.getBBox();return {node,index,pointX,pointY,height:box.height,topOffset:box.y-pointY,leader:svg.querySelector(`line[data-best-label-id="${node.dataset.bestLabelId}"]`)};}).sort((a,b)=>a.pointY-b.pointY||a.index-b.index);
    if(!items.length)return;
    const textHeight=items.reduce((sum,item)=>sum+item.height,0),gap=items.length>1?Math.max(0,Math.min(3,(bounds.maxY-bounds.minY-textHeight)/(items.length-1))):0;
    items.forEach((item,index)=>{const desired=Math.max(bounds.minY,Math.min(bounds.maxY-item.height,item.pointY+item.topOffset));item.top=index?Math.max(desired,items[index-1].top+items[index-1].height+gap):desired;});
    for(let index=items.length-1;index>=0;index--){const ceiling=index===items.length-1?bounds.maxY-items[index].height:items[index+1].top-items[index].height-gap;items[index].top=Math.min(items[index].top,ceiling);}
    if(items[0].top<bounds.minY){items[0].top=bounds.minY;for(let index=1;index<items.length;index++)items[index].top=Math.max(items[index].top,items[index-1].top+items[index-1].height+gap);}
    items.forEach(item=>{const y=item.top-item.topOffset;let x=item.pointX;item.node.setAttribute('x',x);item.node.setAttribute('y',y);const box=item.node.getBBox();if(box.x<bounds.minX)x+=bounds.minX-box.x;if(box.x+box.width>bounds.maxX)x-=box.x+box.width-bounds.maxX;item.node.setAttribute('x',x);if(item.leader){item.leader.setAttribute('x2',x);item.leader.setAttribute('y2',y);}});
  };
  const charts=document.querySelector('#charts');
  metrics.forEach(([name,label])=>{
    const points=chartRuns.map(r=>({r,v:value(r,name)})).filter(x=>Number.isFinite(x.v));
    const card=document.createElement('div');card.className='card chart';card.innerHTML=`<h3>${label}</h3>`;
    const svg=document.createElementNS('http://www.w3.org/2000/svg','svg');svg.setAttribute('viewBox','0 0 310 240');card.append(svg);charts.append(card);
    if(!points.length){svg.innerHTML='<text x="18" y="36">Pending results</text>';return;}
    const lo=Math.min(...points.map(x=>x.v)),hi=Math.max(...points.map(x=>x.v)),pad=Math.max((hi-lo)*.18,.01);
    const epochSpan=Math.max(maxChartEpoch-minChartEpoch,1);
    const y=v=>200-(v-(lo-pad))/(hi-lo+2*pad)*160,x=e=>42+(e-minChartEpoch)/epochSpan*244;
    svg.innerHTML=`<line x1="36" y1="200" x2="286" y2="200" stroke="#aeb7c7"/><text x="2" y="203">${(lo-pad).toFixed(3)}</text><text x="2" y="42">${(hi+pad).toFixed(3)}</text>`;
    series.forEach((s,si)=>{
      const p=points.filter(z=>z.r.series===s).sort((a,b)=>a.r.epoch-b.r.epoch),color=colors[s];
      if(p.length>1)svg.insertAdjacentHTML('beforeend',`<polyline fill="none" stroke="${color}" stroke-width="2" points="${p.map(z=>`${x(z.r.epoch)},${y(z.v)}`).join(' ')}"/>`);
      p.forEach(z=>svg.insertAdjacentHTML('beforeend',`<circle cx="${x(z.r.epoch)}" cy="${y(z.v)}" r="4" fill="${color}"><title>${s} · ${label} ${z.v.toFixed(3)} · E${z.r.epoch} · LR ${z.r.lr} · WD ${z.r.wd}</title></circle><text x="${x(z.r.epoch)-5}" y="218">E${z.r.epoch}</text>`));
      if(p.length){const best=bestPoint(p,name),id=`${name}-best-${si}`,bx=x(best.r.epoch),by=y(best.v),direction=si%2?1:-1;svg.insertAdjacentHTML('beforeend',`<line data-best-label-id="${id}" x1="${bx}" y1="${by}" x2="${bx}" y2="${by+direction*18}" stroke="${color}" stroke-width="1"/><text class="series-best" data-best-label-id="${id}" data-point-x="${bx}" data-point-y="${by}" data-direction="${direction}" x="${bx}" y="${by+direction*18}" text-anchor="middle" font-size="8" font-weight="800" fill="${color}" stroke="#fff" stroke-width="3" paint-order="stroke">${s} · best ${best.v.toFixed(3)}</text>`);}
    });
    layoutBestLabels(svg,{minX:20,maxX:290,minY:12,maxY:190});
  });

  const epochSummaryBatches=summaryBatches;
  const optimizerSummaryBatches=summaryBatches;
  const bestByBatch=new Map();
  epochSummaryBatches.forEach(batch=>{
    const completed=targetsForBatch(batch).map(epoch=>selected.get(`${batch}|${epoch}`)).filter(Boolean);
    if(completed.length)bestByBatch.set(batch,completed.reduce((best,run)=>metric(run)<metric(best)?run:best));
  });
  const bestByDRBatch=new Map();
  epochSummaryBatches.forEach(batch=>{
    const completed=[...drSelected.entries()].filter(([key])=>key.startsWith(`${batch}|`)).map(([,run])=>run);
    if(completed.length)bestByDRBatch.set(batch,completed.reduce((best,run)=>metric(run)<metric(best)?run:best));
  });
  const bestByAdaptiveBatch=new Map();
  epochSummaryBatches.forEach(batch=>{
    const completed=[...adaptiveSelected.entries()].filter(([key])=>key.startsWith(`${batch}|`)).map(([,run])=>run);
    if(completed.length)bestByAdaptiveBatch.set(batch,completed.reduce((best,run)=>metric(run)<metric(best)?run:best));
  });
  const bestForSummaryColumn=column=>{
    if(column.kind==='adaptive')return bestByAdaptiveBatch.get(column.batch);
    if(column.kind==='dr')return bestByDRBatch.get(column.batch);
    return bestByBatch.get(column.batch);
  };
  const bestBySimulation=new Map();
  (simulation.columns||[]).forEach(column=>{
    const completed=[...simulationSelected.values()].filter(run=>run.method===column.key);
    if(completed.length)bestBySimulation.set(column.key,completed.reduce((best,run)=>metric(run)<metric(best)?run:best));
  });
  const validationSummary=document.querySelector('#validation-summary');
  const validationSummaryNote=validationSummary?.closest('.summary-panel')?.querySelector('.summary-note');
  if(validationSummaryNote)validationSummaryNote.textContent='Each DR+WT+EmbedWD column sits immediately beside its Original batch-size column. Locked-WD cells show [POST] | [PD]; when [POST] is unavailable, the left side reads Unknown. Historical [PD] begins at E8. Bold and underline apply only to [POST]: bold marks the best [POST] epoch within each column, and underline marks the best [POST] result within each epoch. [PD] values are displayed for monitoring and are never compared with [POST].';
  validationSummary.innerHTML=(d.targetEpochs||[]).map(epoch=>{
    const rowRuns=[
      ...summaryMethodColumns.map(column=>runForSummaryColumn(column,epoch)),
      ...(simulation.columns||[]).map(column=>simulationSelected.get(`${column.key}|${epoch}`)),
    ].filter(Boolean);
    const rowBestMetric=rowRuns.length?Math.min(...rowRuns.map(metric)):null;
    const cells=summaryMethodColumns.map(column=>{
      const candidate=runForSummaryColumn(column,epoch);
      const preDecay=preDecayForSummaryColumn(column,epoch);
      if(!candidate&&!preDecay)return '<td>—</td>';
      const lockedAdaptive=column.kind==='adaptive'&&(lockedWdPhasePolicies.has(candidate?.policy)||lockedWdPhasePolicies.has(preDecay?.policy));
      if(lockedAdaptive){
        const columnBest=Boolean(candidate)&&bestForSummaryColumn(column)===candidate;
        const rowBest=Boolean(candidate)&&metric(candidate)===rowBestMetric;
        const postFormatted=candidate?metric(candidate).toFixed(3):null;
        const postRowMarked=rowBest?`<span class="summary-row-best">${postFormatted}</span>`:postFormatted;
        const postDisplayed=candidate?(columnBest?`<strong>[POST] ${postRowMarked}</strong>`:`[POST] ${postRowMarked}`):'Unknown';
        const pdDisplayed=preDecay?metric(preDecay).toFixed(3):'—';
        const reference=candidate||preDecay;
        const title=`${column.label}; locked LR ${reference.lr}; WD ${reference.wd}; [POST] ${candidate?metric(candidate).toFixed(3):'not available'}; [PD] ${preDecay?metric(preDecay).toFixed(3):'not available'}; phases are not compared`;
        return `<td class="${columnBest?'summary-best':''}" title="${escapeAttribute(title)}"><span>${postDisplayed}</span> <span class="phase-separator">|</span> <span class="pd-value">[PD] ${pdDisplayed}</span></td>`;
      }
      if(!candidate)return '<td>—</td>';
      const columnBest=bestForSummaryColumn(column)===candidate;
      const rowBest=metric(candidate)===rowBestMetric;
      const formatted=metric(candidate).toFixed(3);
      const rowMarked=rowBest?`<span class="summary-row-best">${formatted}</span>`:formatted;
      return `<td class="${columnBest?'summary-best':''}" title="${column.label}; LR ${candidate.lr}; WD ${candidate.wd}">${columnBest?`<strong>${rowMarked}</strong>`:rowMarked}</td>`;
    }).join('');
    const simulationCells=(simulation.columns||[]).map(column=>{
      const run=simulationSelected.get(`${column.key}|${epoch}`);
      if(!run)return '<td>—</td>';
      const columnBest=bestBySimulation.get(column.key)===run;
      const rowBest=metric(run)===rowBestMetric;
      const formatted=metric(run).toFixed(3);
      const rowMarked=rowBest?`<span class="summary-row-best">${formatted}</span>`:formatted;
      return `<td class="${columnBest?'summary-best':''}" title="global BS ${run.batchSequences}; simulated BS ${run.simulatedBatchSequences}; LR ${run.lr}; WD ${run.wd}">${columnBest?`<strong>${rowMarked}</strong>`:rowMarked}</td>`;
    }).join('');
    return `<tr><td><strong>E${epoch}</strong></td>${cells}${simulationCells}</tr>`;
  }).join('');

  const epochDownstreamTables=[
    {bodyId:'epoch-hs-accuracy-summary',metricName:'acc',higherIsBetter:true,decimals:2,label:'HellaSwag accuracy'},
    {bodyId:'epoch-hs-bpb-summary',metricName:'bpb',higherIsBetter:false,decimals:3,label:'HellaSwag BPB'},
    {bodyId:'epoch-avg8-accuracy-summary',metricName:'avg8_acc',higherIsBetter:true,decimals:2,label:'8-task average accuracy'},
    {bodyId:'epoch-avg8-bpb-summary',metricName:'avg8_bpb',higherIsBetter:false,decimals:3,label:'8-task average BPB'},
  ];
  const renderEpochDownstreamTable=config=>{
    const body=document.querySelector(`#${config.bodyId}`);
    if(!body)return;
    const bestRunByColumn=new Map();
    summaryMethodColumns.forEach(column=>{
      const candidates=(d.targetEpochs||[]).map(epoch=>runForSummaryColumn(column,epoch)).filter(Boolean).filter(run=>Number.isFinite(value(run,config.metricName)));
      if(!candidates.length)return;
      bestRunByColumn.set(`${column.kind}|${column.batch}`,candidates.reduce((best,run)=>{
        const candidateValue=value(run,config.metricName),bestValue=value(best,config.metricName);
        return config.higherIsBetter?candidateValue>bestValue?run:best:candidateValue<bestValue?run:best;
      }));
    });
    body.innerHTML=(d.targetEpochs||[]).map(epoch=>{
      const rowRuns=summaryMethodColumns.map(column=>runForSummaryColumn(column,epoch)).filter(Boolean);
      const rowValues=rowRuns.map(run=>value(run,config.metricName)).filter(Number.isFinite);
      const rowBest=rowValues.length?(config.higherIsBetter?Math.max(...rowValues):Math.min(...rowValues)):null;
      const cells=summaryMethodColumns.map(column=>{
        const run=runForSummaryColumn(column,epoch);
        const metricValue=run?value(run,config.metricName):null;
        if(!run||!Number.isFinite(metricValue))return '<td>—</td>';
        const columnBest=bestRunByColumn.get(`${column.kind}|${column.batch}`)===run;
        const formatted=metricValue.toFixed(config.decimals);
        const rowMarked=metricValue===rowBest?`<span class="summary-row-best">${formatted}</span>`:formatted;
        const displayed=columnBest?`<strong>${rowMarked}</strong>`:rowMarked;
        return `<td class="${columnBest?'summary-best':''}" title="${column.label}; validation-selected LR ${run.lr}; WD ${run.wd}; ${config.label}">${displayed}</td>`;
      }).join('');
      return `<tr><td><strong>E${epoch}</strong></td>${cells}</tr>`;
    }).join('');
  };
  epochDownstreamTables.forEach(renderEpochDownstreamTable);

  const coordinateSummary=document.querySelector('#coordinate-summary');
  if(coordinateSummary)coordinateSummary.innerHTML=(d.targetEpochs||[]).map(epoch=>{
    const rowRuns=summaryMethodColumns.map(column=>runForSummaryColumn(column,epoch)).filter(Boolean);
    const rowBestMetric=rowRuns.length?Math.min(...rowRuns.map(metric)):null;
    const cells=summaryMethodColumns.map(column=>{
      const run=runForSummaryColumn(column,epoch);
      if(!run)return '<td>—</td>';
      const columnBest=bestForSummaryColumn(column)===run;
      const rowBest=metric(run)===rowBestMetric;
      const formatted=`(${run.lr}, ${run.wd})`;
      const rowMarked=rowBest?`<span class="summary-row-best">${formatted}</span>`:formatted;
      return `<td class="${columnBest?'summary-best':''}">${columnBest?`<strong>${rowMarked}</strong>`:rowMarked}</td>`;
    }).join('');
    return `<tr><td><strong>E${epoch}</strong></td>${cells}</tr>`;
  }).join('');

  const optimizerStepSummary=document.querySelector('#optimizer-step-summary');
  const optimizerStepTable=optimizerStepSummary?.closest('table');
  const optimizerStepHeaderRow=optimizerStepTable?.querySelector('thead tr');
  if(optimizerStepHeaderRow&&!optimizerStepHeaderRow.querySelector('.training-time-header')){
    optimizerStepHeaderRow.querySelector('th')?.insertAdjacentHTML('afterend','<th class="training-time-header">Training time</th>');
  }
  const optimizerStepNote=optimizerStepSummary?.closest('.summary-panel')?.querySelector('.summary-note');
  if(optimizerStepNote){
    const microbatch=Number(optimizerTiming.microbatchSequences);
    const secondsPerStep=Number(optimizerTiming.secondsPerStep);
    const hasSourceStepMappedSimulation=(simulation.columns||[]).some(column=>Number.isFinite(Number(column.sourceBatchSequences)));
    const simulationNote=(simulation.columns||[]).length?` Local-update columns use each method's declared global batch to select the matched epoch row; simulated batch size does not change that optimizer-step placement.${hasSourceStepMappedSimulation?' Methods initialized from a smaller-batch checkpoint are instead placed by their cumulative endpoint optimizer step, including the parent checkpoint history.':''}`:'';
    const endpointNote=(simulation.columns||[]).length?' Every DR column uses the data-loader report’s healthy nondecreasing-WD selection at the source epoch. E12 BS1024 values are post-decay endpoint evaluations at checkpoint step 2,862; step 2,575 is the pre-decay resume checkpoint and is not the displayed result. The matched row is labeled ≈2,861 because token-based optimizer-step arithmetic rounds to the nearest step.':'';
    optimizerStepNote.innerHTML=`Optimizer steps are recalculated per batch as epoch × 1B pool tokens ÷ (global batch × 4,096) and matched by row. Every completed validation endpoint contributes a row, including intermediate and latest [PD]/[POST] frontiers.${simulationNote}${endpointNote} For the BS64-E1-initialized DiLoCo arm, E2 is shown on its own ≈4,292-step row: ≈3,815 BS64-parent steps + ≈477 BS512 E1→E2 steps. Idealized training time assumes microbatch ${microbatch}, ${secondsPerStep}s/step, accumulation 1, unlimited GPUs, and GPU count = global batch ÷ ${microbatch}. Locked-WD cells show [POST] | [PD]; row minima use [POST] only, and [PD] is never compared with [POST]. Other result cells show epoch · validation CE. <span class="matched-stopped-key">Purple text</span> carries forward the best CE reached before a confirmed terminal non-improvement and shows the real source epoch; it is not a new higher-epoch measurement.`;
  }
  optimizerStepSummary.innerHTML=validationOptimizerStepComparisons.map(comparison=>{
    const optimizerSteps=optimizerStepsForComparison(comparison);
    const secondsPerStep=Number(optimizerTiming.secondsPerStep);
    const trainingSeconds=optimizerSteps*secondsPerStep;
    const timeCell=Number.isFinite(trainingSeconds)?`<td class="matched-value" title="${optimizerSteps.toLocaleString()} steps × ${secondsPerStep}s per step">≈${formatDuration(trainingSeconds)}</td>`:'<td>—</td>';
    const rowEntries=[...summaryMethodColumns.map(column=>{
      const epoch=comparison.epochs?.[String(column.batch)];
      return Number.isFinite(epoch)?matchedSummaryEntryFor(column,epoch):null;
    }),...(simulation.columns||[]).map(column=>{
      const run=simulationRunAtMatchedSteps(column,optimizerSteps,comparison);
      return run?{run,value:metric(run),carried:false,sourceEpoch:Number(run.epoch)}:null;
    })].filter(Boolean);
    const rowBestMetric=rowEntries.length?Math.min(...rowEntries.map(entry=>entry.value)):null;
    const cells=summaryMethodColumns.map(column=>{
      const epoch=comparison.epochs?.[String(column.batch)];
      if(!Number.isFinite(epoch))return '<td>—</td>';
      const entry=matchedSummaryEntryFor(column,epoch);
      const preDecay=preDecayForSummaryColumn(column,epoch);
      const recalculatedSteps=optimizerStepsForEpochBatch(epoch,column.batch);
      const arithmetic=Number.isFinite(recalculatedSteps)?`E${epoch} × ${Number(optimizerTiming.uniquePoolTokens).toLocaleString()} tokens ÷ (BS ${column.batch} × ${Number(optimizerTiming.sequenceLength)||4096}) = ${recalculatedSteps.toLocaleString()} optimizer steps`:'';
      const lockedAdaptive=column.kind==='adaptive'&&(lockedWdPhasePolicies.has(entry?.run?.policy)||lockedWdPhasePolicies.has(preDecay?.policy));
      if(lockedAdaptive){
        const postFormatted=entry?entry.value.toFixed(3):null;
        const postMarked=entry&&entry.value===rowBestMetric?`<strong><span class="summary-row-best">${postFormatted}</span></strong>`:postFormatted;
        const postDisplayed=entry?`[POST] ${postMarked}`:'Unknown [POST]';
        const pdDisplayed=preDecay?`[PD] ${metric(preDecay).toFixed(3)}`:'[PD] —';
        const reference=entry?.run||preDecay;
        const title=`${arithmetic}; ${column.label}; locked LR ${reference.lr}; WD ${reference.wd}; [POST] ${entry?entry.value.toFixed(3):'not available'}; [PD] ${preDecay?metric(preDecay).toFixed(3):'not available'}; phases are not compared`;
        return `<td class="matched-value" title="${escapeAttribute(title)}">E${epoch} · <span>${postDisplayed}</span> <span class="phase-separator">|</span> <span class="pd-value">${pdDisplayed}</span></td>`;
      }
      if(!entry)return `<td class="matched-value"${arithmetic?` title="${escapeAttribute(arithmetic)}"`:''}>E${epoch} · —</td>`;
      const formatted=entry.value.toFixed(3);
      const marked=entry.value===rowBestMetric?`<strong><span class="summary-row-best">${formatted}</span></strong>`:formatted;
      if(column.kind==='original'){
        const carryExplanation=entry.carried?(entry.replacedRun?` The E${epoch} endpoint measured CE ${metric(entry.replacedRun).toFixed(3)}, but the displayed E${entry.sourceEpoch} CE ${entry.value.toFixed(3)} remains the best result before the terminal E${entry.stopEpoch} non-improvement.`:` No E${epoch} endpoint was run: the displayed E${entry.sourceEpoch} CE ${entry.value.toFixed(3)} is carried into this matched-step slot as the best result before the terminal E${entry.stopEpoch} non-improvement.`):'';
        return `<td class="matched-value${entry.carried?' matched-stopped':''}" title="${escapeAttribute(arithmetic+carryExplanation)}">E${entry.carried?entry.sourceEpoch:epoch} · ${marked}</td>`;
      }
      const endpointStep=endpointOptimizerStep(entry.run);
      const endpointNote=column.kind==='dr'&&Number.isFinite(endpointStep)?`; post-decay endpoint step ${endpointStep.toLocaleString()}`:'';
      return `<td class="matched-value" title="${column.label}; LR ${entry.run.lr}; WD ${entry.run.wd}${endpointNote}">E${entry.run.epoch} · ${marked}</td>`;
    }).join('');
    const simulationCells=(simulation.columns||[]).map(column=>{
      const run=simulationRunAtMatchedSteps(column,optimizerSteps,comparison);
      if(!run)return '<td>—</td>';
      const formatted=metric(run).toFixed(3);
      const marked=metric(run)===rowBestMetric?`<strong><span class="summary-row-best">${formatted}</span></strong>`:formatted;
      const endpointStep=endpointOptimizerStep(run);
      const explicitMatchedSteps=Number(column.matchedOptimizerStepsByEpoch?.[String(run.epoch)]);
      const sourceMapping=Number.isFinite(explicitMatchedSteps)
        ?`; matched compute ${explicitMatchedSteps.toLocaleString()} steps, including the BS ${column.sourceBatchSequences} parent history`
        :Number.isFinite(Number(column.sourceBatchSequences))&&Number.isFinite(endpointStep)
        ?`; cumulative checkpoint step ${endpointStep.toLocaleString()} (initialized from BS ${column.sourceBatchSequences})`
        :'';
      return `<td class="matched-value" title="global BS ${column.globalBatchSequences}; simulated BS ${column.simulatedBatchSequences}; LR ${run.lr}; WD ${run.wd}${sourceMapping}">E${run.epoch} · ${marked}</td>`;
    }).join('');
    return `<tr><td>≈${optimizerSteps.toLocaleString()}</td>${timeCell}${cells}${simulationCells}</tr>`;
  }).join('');

  const matchedDownstreamTables=[
    {bodyId:'optimizer-step-hs-accuracy-summary',metricName:'acc',higherIsBetter:true,decimals:2,label:'HellaSwag accuracy'},
    {bodyId:'optimizer-step-hs-bpb-summary',metricName:'bpb',higherIsBetter:false,decimals:3,label:'HellaSwag BPB'},
    {bodyId:'optimizer-step-avg8-accuracy-summary',metricName:'avg8_acc',higherIsBetter:true,decimals:2,label:'8-task average accuracy'},
    {bodyId:'optimizer-step-avg8-bpb-summary',metricName:'avg8_bpb',higherIsBetter:false,decimals:3,label:'8-task average BPB'},
  ];
  const downstreamValue=(run,metricName)=>{
    const result=run?value(run,metricName):null;
    return Number.isFinite(result)?Number(result):null;
  };
  const markedDownstreamValue=(metricValue,rowBest,decimals)=>{
    if(!Number.isFinite(metricValue))return '—';
    const formatted=metricValue.toFixed(decimals);
    return metricValue===rowBest?`<strong><span class="summary-row-best">${formatted}</span></strong>`:formatted;
  };
  const renderMatchedDownstreamTable=config=>{
    const body=document.querySelector(`#${config.bodyId}`);
    if(!body)return;
    body.innerHTML=optimizerStepComparisons.map(comparison=>{
      const optimizerSteps=optimizerStepsForComparison(comparison);
      const secondsPerStep=Number(optimizerTiming.secondsPerStep);
      const trainingSeconds=optimizerSteps*secondsPerStep;
      const timeCell=Number.isFinite(trainingSeconds)?`<td class="matched-value" title="${optimizerSteps.toLocaleString()} steps × ${secondsPerStep}s per step">≈${formatDuration(trainingSeconds)}</td>`:'<td>—</td>';
      const rowRuns=[...summaryMethodColumns.map(column=>{
        const epoch=comparison.epochs?.[String(column.batch)];
        return Number.isFinite(epoch)?matchedSummaryEntryFor(column,epoch)?.run:null;
      }).filter(Boolean),...(simulation.columns||[]).map(column=>simulationRunAtMatchedSteps(column,optimizerSteps,comparison)).filter(Boolean)];
      const rowValues=rowRuns.map(run=>downstreamValue(run,config.metricName)).filter(Number.isFinite);
      const rowBest=rowValues.length?(config.higherIsBetter?Math.max(...rowValues):Math.min(...rowValues)):null;
      const cells=summaryMethodColumns.map(column=>{
        const epoch=comparison.epochs?.[String(column.batch)];
        if(!Number.isFinite(epoch))return '<td>—</td>';
        const entry=matchedSummaryEntryFor(column,epoch);
        const recalculatedSteps=optimizerStepsForEpochBatch(epoch,column.batch);
        const arithmetic=Number.isFinite(recalculatedSteps)?`E${epoch} × ${Number(optimizerTiming.uniquePoolTokens).toLocaleString()} tokens ÷ (BS ${column.batch} × ${Number(optimizerTiming.sequenceLength)||4096}) = ${recalculatedSteps.toLocaleString()} optimizer steps. `:'';
        if(!entry)return `<td class="matched-value"${arithmetic?` title="${escapeAttribute(arithmetic)}"`:''}>E${epoch} · —</td>`;
        const metricValue=downstreamValue(entry.run,config.metricName);
        const displayedEpoch=entry.carried?entry.sourceEpoch:entry.run.epoch;
        const carryExplanation=entry.carried?`The E${entry.sourceEpoch} validation-selected endpoint is carried into this matched-step slot after terminal E${entry.stopEpoch} non-improvement. `:'';
        const endpointStep=endpointOptimizerStep(entry.run);
        const endpointNote=column.kind==='dr'&&Number.isFinite(endpointStep)?` Post-decay endpoint step ${endpointStep.toLocaleString()}.`:'';
        const title=` title="${escapeAttribute(`${arithmetic}${carryExplanation}${column.label}; ${config.label}; LR ${entry.run.lr}; WD ${entry.run.wd}.${endpointNote}`)}"`;
        return `<td class="matched-value${entry.carried?' matched-stopped':''}"${title}>E${displayedEpoch} · ${markedDownstreamValue(metricValue,rowBest,config.decimals)}</td>`;
      }).join('');
      const simulationCells=(simulation.columns||[]).map(column=>{
        const run=simulationRunAtMatchedSteps(column,optimizerSteps,comparison);
        if(!run)return '<td>—</td>';
        const metricValue=downstreamValue(run,config.metricName);
        const endpointStep=endpointOptimizerStep(run);
        const explicitMatchedSteps=Number(column.matchedOptimizerStepsByEpoch?.[String(run.epoch)]);
        const sourceMapping=Number.isFinite(explicitMatchedSteps)
          ?`; matched compute ${explicitMatchedSteps.toLocaleString()} steps, including the BS ${column.sourceBatchSequences} parent history`
          :Number.isFinite(Number(column.sourceBatchSequences))&&Number.isFinite(endpointStep)
          ?`; cumulative checkpoint step ${endpointStep.toLocaleString()} (initialized from BS ${column.sourceBatchSequences})`
          :'';
        return `<td class="matched-value" title="global BS ${column.globalBatchSequences}; simulated BS ${column.simulatedBatchSequences}; validation-selected LR ${run.lr}; WD ${run.wd}${sourceMapping}">E${run.epoch} · ${markedDownstreamValue(metricValue,rowBest,config.decimals)}</td>`;
      }).join('');
      return `<tr><td>≈${optimizerSteps.toLocaleString()}</td>${timeCell}${cells}${simulationCells}</tr>`;
    }).join('');
  };
  matchedDownstreamTables.forEach(renderMatchedDownstreamTable);

  const grid=document.querySelector('#coordinate-grid');
  const groups=new Map();
  coordinateRuns.forEach(r=>{const g=key(r);if(!groups.has(g))groups.set(g,[]);groups.get(g).push(r);});
  if(grid)grid.innerHTML=[...groups.entries()].sort((a,b)=>{const [ab,ae]=a[0].split('|'),[bb,be]=b[0].split('|');return Number(ab)-Number(bb)||Number(ae)-Number(be)}).map(([g,runs])=>{
    const [batch,epoch]=g.split('|'),best=selected.get(g);
    const ordered=runs.sort((a,b)=>Number(a.wd)-Number(b.wd)||Number(a.lr)-Number(b.lr));
    const chips=ordered.map(r=>`<span class="tuple ${best===r?'selected':active(r)?'active':''} ${unhealthy(r)?'unhealthy':suspicious(r)?'suspicious':''}"${unhealthy(r)||suspicious(r)?` title="${escapeAttribute(statusRecord(r).reason)}"`:''}>(LR ${r.lr}, WD ${r.wd}) · ${r.status}${Number.isFinite(metric(r))?` · CE ${metric(r).toFixed(3)}`:''}</span>`).join('');
    const selection=best?`LR ${best.lr}, WD ${best.wd} · CE ${metric(best).toFixed(3)}`:'pending';
    return `<tr><td><strong>E${epoch}</strong></td><td>${batch}</td><td><div class="tuple-list">${chips}</div></td><td>${selection}</td></tr>`;
  }).join('');

  const tableMetric=v=>Number.isFinite(v)?v.toFixed(3):'—';
  const selectedBaseline=[...selected.values()].filter(r=>r.batchSequences===1024&&coordinateKeys.has(key(r)));
  const provenanceMap=new Map([...newRuns,...attemptHistoryRuns,...selectedBaseline,...baselineRuns.filter(r=>unhealthy(r)||suspicious(r))].map(r=>[`${r.batchSequences}|${r.epoch}|${r.lr}|${r.wd}|${wandbId(r)||r.beaker}`,r]));
  const provenance=[...provenanceMap.values()].sort((a,b)=>a.batchSequences-b.batchSequences||a.epoch-b.epoch||Number(a.lr)-Number(b.lr));
  const rows=document.querySelector('#rows');
  if(rows)provenance.forEach(r=>rows.insertAdjacentHTML('beforeend',`<tr class="${unhealthy(r)?'run-unhealthy':suspicious(r)?'run-suspicious':''}"${r.reason?` title="${escapeAttribute(r.reason)}"`:''}><td>${r.batchSequences}</td><td>${r.epoch}</td><td>${r.lr}</td><td>${r.wd}</td><td class="${active(r)?'run-active':''}">${r.status}${r.historicalAttempt?' · recovery provenance':''}</td><td>${tableMetric(r.train)}</td><td>${tableMetric(metric(r))}</td><td>${tableMetric(r.acc)}</td><td>${tableMetric(r.bpb)}</td><td>${wandbId(r)?`<a href="https://wandb.ai/ai2-llm/sewonm-icsl/runs/${wandbId(r)}">${wandbId(r)}</a>`:'—'}</td><td>${r.beaker?`<a href="https://beaker.org/ex/${r.beaker}">experiment</a>`:'—'}</td></tr>`));
})();
