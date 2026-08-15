(() => {
  const d=window.STEP2_1_REPORT_DATA||{uniqueRuns:[],chainExperiments:[],targetEpochs:[1,2,3,4,5]};
  const step2=window.STEP2_1_STEP2_DATA||{uniqueRuns:[]};
  const batch=window.STEP2_1_BATCH_DATA||{batchSweeps:[]};
  const baseline=window.STEP2_1_WD_DATA||{runs:[]};
  const targets=d.targetEpochs||[1,2,3,4,5];
  const fractionalTargets=d.matchedStepTargetEpochs||[0.125,0.25,0.5];
  const summaryBatches=[64,256,1024];
  document.querySelector('#title').textContent=d.title;
  document.querySelector('#setup').textContent=d.setup;
  document.querySelector('#updated').textContent=`Updated ${d.updated} · report 1-1 source: ${d.repeatedSourceUpdated||batch.updated||'pending'}`;
  document.querySelector('#selection').textContent=d.selection;

  const colors={Unique:'#15803d','Repeated BS64':'#2563eb','Repeated BS256':'#7c3aed','Repeated BS1024':'#dc5a39'};
  const series=['Unique','Repeated BS64','Repeated BS256','Repeated BS1024'];
  const uniqueHealth={...(step2.healthAudit?.unhealthy||{}),...(d.healthAudit?.unhealthy||{})};
  const repeatedHealth={...(baseline.healthAudit?.unhealthy||{}),...(batch.healthAudit?.unhealthy||{})};
  const uniqueEligible=r=>['5e-4','1e-3'].includes(r.lr);
  const active=r=>['active','running','submitted','pending','queued','planned'].includes(r.status);
  const metric=r=>r.validation??r.c4;
  const completeUnique=(r,health)=>r.status==='complete'&&uniqueEligible(r)&&Number.isFinite(metric(r))&&!health[r.wandb];
  const completeRepeated=(r,health)=>r.status==='complete'&&Number.isFinite(metric(r))&&!health[r.wandb];

  const repeatedCompletedTargets=[
    ...(batch.batchSweeps||[]).filter(sweep=>summaryBatches.includes(sweep.batchSequences)).flatMap(sweep=>Object.entries(sweep.results||{}).filter(([,result])=>completeRepeated({...sweep,...result},repeatedHealth)).map(([epoch])=>Number(epoch))),
    ...(baseline.runs||[]).filter(r=>completeRepeated(r,repeatedHealth)).map(r=>Number(r.epoch)),
  ].filter(Number.isFinite);
  const comparisonTargets=[...new Set([...targets,...repeatedCompletedTargets.filter(epoch=>epoch>12)])].sort((a,b)=>a-b);

  const unique1024=(step2.uniqueRuns||[]).filter(r=>targets.includes(r.epoch)&&uniqueEligible(r)&&r.status==='complete').map(r=>({...r,batchSequences:1024,warmupSteps:24,series:'Unique',condition:'Unique'}));
  const uniqueNewAll=(d.uniqueRuns||[]).filter(r=>uniqueEligible(r)).map(r=>({...r,series:'Unique',condition:'Unique'}));
  const uniqueNew=uniqueNewAll.filter(r=>targets.includes(r.epoch));
  const uniquePlaceholders=(d.chainExperiments||[]).filter(active).map(r=>({...r,epoch:r.activeEpoch||1,series:'Unique',condition:'Unique'}));
  const uniqueRuns=[...unique1024,...uniqueNew];
  const uniqueMatchedRuns=[...unique1024,...uniqueNewAll];
  const repeatedSmall=(batch.batchSweeps||[]).flatMap(sweep=>comparisonTargets.flatMap(epoch=>{const result=sweep.results?.[epoch];return result?[{...sweep,...result,epoch,status:result.status,series:`Repeated BS${sweep.batchSequences}`,condition:'Repeated'}]:[];}));
  const repeated1024=(baseline.runs||[]).filter(r=>comparisonTargets.includes(r.epoch)&&r.status==='complete').map(r=>({...r,batchSequences:1024,warmupSteps:24,series:'Repeated BS1024',condition:'Repeated'}));
  const repeatedRuns=[...repeatedSmall,...repeated1024];
  const bestByTarget=(runs,health,isComplete,targetList=targets)=>targetList.map(epoch=>runs.filter(r=>r.epoch===epoch&&isComplete(r,health)).sort((a,b)=>metric(a)-metric(b))[0]).filter(Boolean);
  const uniqueBest=bestByTarget(uniqueRuns,uniqueHealth,completeUnique);
  const repeatedBestByBatch=Object.fromEntries([64,256,1024].map(bs=>[bs,bestByTarget(repeatedRuns.filter(r=>r.batchSequences===bs),repeatedHealth,completeRepeated,comparisonTargets)]));
  const repeatedBest=Object.values(repeatedBestByBatch).flat().filter(r=>targets.includes(r.epoch));
  const chartRuns=[...uniqueBest,...repeatedBest];

  const legend=document.querySelector('#legend');
  series.forEach(s=>legend.insertAdjacentHTML('beforeend',`<label><span class="dot" style="background:${colors[s]}"></span><input type="checkbox" data-series-toggle="${s}" checked>${s==='Unique'?'Unique 5B pool · best BS/LR':`${s} · LR/WD tuned`}</label>`));
  legend.insertAdjacentHTML('beforeend','<label><span class="health-key unhealthy"></span>Unhealthy / excluded</label>');

  const tasks=['arc_challenge','arc_easy','csqa','hellaswag','openbookqa','piqa','socialiqa','winogrande'];
  const downstream=(r,task)=>task==='hellaswag'?r.acc:r.downstream?.[task];
  const avg8Acc=r=>{const values=tasks.map(task=>downstream(r,task));return values.every(Number.isFinite)?values.reduce((a,b)=>a+b,0)/values.length:null;};
  const avg8Bpb=r=>{if(Number.isFinite(r.avg8Bpb))return r.avg8Bpb;const mapped=step2.downstreamBpbByWandb?.[r.wandb]||baseline.avg8BpbByWandb?.[r.wandb];if(Number.isFinite(mapped))return mapped;const values=tasks.map(task=>task==='hellaswag'?r.bpb:(r.downstreamBpb?.[task]??mapped?.[task]));return values.every(Number.isFinite)?values.reduce((a,b)=>a+b,0)/values.length:null;};
  const value=(r,key)=>key==='validation'?metric(r):key==='avg8_acc'?avg8Acc(r):key==='avg8_bpb'?avg8Bpb(r):r[key];
  const metrics=[['validation','Held-out DCLM validation CE ↓'],['bpb','HellaSwag non-v2 BPB ↓'],['acc','HellaSwag length-normalized accuracy ↑'],['avg8_bpb','8-task average BPB · no BoolQ ↓'],['avg8_acc','8-task average accuracy · no BoolQ ↑']];
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
  const renderCharts=()=>{
    const enabled=new Set([...document.querySelectorAll('[data-series-toggle]:checked')].map(input=>input.dataset.seriesToggle));
    charts.innerHTML='';
    metrics.forEach(([name,label])=>{
      const points=chartRuns.filter(r=>enabled.has(r.series)).map(r=>({r,v:value(r,name)})).filter(x=>Number.isFinite(x.v));
      const card=document.createElement('div');card.className='card chart';card.innerHTML=`<h3>${label}</h3>`;
      const svg=document.createElementNS('http://www.w3.org/2000/svg','svg');svg.setAttribute('viewBox','0 0 310 240');card.append(svg);charts.append(card);
      if(!points.length){svg.innerHTML='<text x="18" y="36">Pending results</text>';return;}
      const lo=Math.min(...points.map(x=>x.v)),hi=Math.max(...points.map(x=>x.v)),pad=Math.max((hi-lo)*.18,.01);
      const minTarget=Math.min(...targets),maxTarget=Math.max(...targets);
      const y=v=>200-(v-(lo-pad))/(hi-lo+2*pad)*160,x=e=>42+(e-minTarget)/Math.max(1,maxTarget-minTarget)*244;
      svg.innerHTML=`<line x1="36" y1="200" x2="286" y2="200" stroke="#aeb7c7"/><text x="2" y="203">${(lo-pad).toFixed(3)}</text><text x="2" y="42">${(hi+pad).toFixed(3)}</text>`;
      series.filter(s=>enabled.has(s)).forEach((s,si)=>{
        const p=points.filter(z=>z.r.series===s).sort((a,b)=>a.r.epoch-b.r.epoch),color=colors[s];
        if(p.length>1)svg.insertAdjacentHTML('beforeend',`<polyline fill="none" stroke="${color}" stroke-width="2" points="${p.map(z=>`${x(z.r.epoch)},${y(z.v)}`).join(' ')}"/>`);
        p.forEach(z=>svg.insertAdjacentHTML('beforeend',`<circle cx="${x(z.r.epoch)}" cy="${y(z.v)}" r="4" fill="${color}"><title>${s} · ${label} ${z.v.toFixed(3)} · ${z.r.epoch}B · BS ${z.r.batchSequences} · LR ${z.r.lr} · WD ${z.r.wd}</title></circle><text x="${x(z.r.epoch)-5}" y="218">${z.r.epoch}B</text>`));
        if(p.length){const best=bestPoint(p,name),id=`${name}-best-${si}`,bx=x(best.r.epoch),by=y(best.v),direction=si%2?1:-1;svg.insertAdjacentHTML('beforeend',`<line data-best-label-id="${id}" x1="${bx}" y1="${by}" x2="${bx}" y2="${by+direction*18}" stroke="${color}" stroke-width="1"/><text class="series-best" data-best-label-id="${id}" data-point-x="${bx}" data-point-y="${by}" data-direction="${direction}" x="${bx}" y="${by+direction*18}" text-anchor="middle" font-size="8" font-weight="800" fill="${color}" stroke="#fff" stroke-width="3" paint-order="stroke">${s} · best ${best.v.toFixed(3)}</text>`);}
      });
      layoutBestLabels(svg,{minX:20,maxX:290,minY:12,maxY:190});
    });
  };
  document.querySelectorAll('[data-series-toggle]').forEach(input=>input.addEventListener('change',renderCharts));
  renderCharts();

  const fmt=v=>Number.isFinite(v)?v.toFixed(3):'—';
  const signed=v=>Number.isFinite(v)?`${v>0?'+':''}${v.toFixed(3)}`:'—';
  const link=(kind,id)=>id?`<a href="${kind==='wandb'?`https://wandb.ai/ai2-llm/sewonm-icsl/runs/${id}`:`https://beaker.org/ex/${id}`}">${kind==='wandb'?id:'experiment'}</a>`:'—';
  const uniqueBestByBatch=Object.fromEntries(summaryBatches.map(bs=>[bs,bestByTarget(uniqueRuns.filter(r=>r.batchSequences===bs),uniqueHealth,completeUnique)]));
  const uniqueColumnBest=Object.fromEntries(summaryBatches.map(bs=>{
    const runs=uniqueBestByBatch[bs];
    return [bs,runs.length?runs.reduce((best,run)=>metric(run)<metric(best)?run:best):null];
  }));
  const repeatedColumnBest=Object.fromEntries(summaryBatches.map(bs=>{
    const runs=repeatedBestByBatch[bs];
    return [bs,runs.length?runs.reduce((best,run)=>metric(run)<metric(best)?run:best):null];
  }));
  const uniqueClosestToRepeatedBest=Object.fromEntries(summaryBatches.map(bs=>{
    const repeated=repeatedColumnBest[bs],runs=uniqueBestByBatch[bs];
    if(!repeated||!runs.length)return [bs,null];
    return [bs,[...runs].sort((a,b)=>Math.abs(metric(a)-metric(repeated))-Math.abs(metric(b)-metric(repeated))||a.epoch-b.epoch)[0]];
  }));
  document.querySelector('#unique-5b-summary').innerHTML=targets.map(target=>`<tr><td><strong>${target}B</strong></td>${summaryBatches.map(bs=>{
    const run=uniqueBestByBatch[bs].find(r=>r.epoch===target);
    if(!run)return '<td>—</td>';
    const best=run===uniqueColumnBest[bs];
    return `<td class="${best?'summary-best':''}" title="LR ${run.lr} · WD ${run.wd}">${best?`<strong>${fmt(metric(run))}</strong>`:fmt(metric(run))}</td>`;
  }).join('')}</tr>`).join('');

  document.querySelector('#selected-endpoint-comparison').innerHTML=comparisonTargets.map(target=>`<tr><td>${target}B</td>${summaryBatches.map(bs=>{
    const unique=uniqueBestByBatch[bs].find(r=>r.epoch===target);
    const repeated=repeatedBestByBatch[bs].find(r=>r.epoch===target);
    if(!unique&&!repeated)return '<td class="endpoint-cell">—</td>';
    const delta=unique&&repeated?metric(unique)-metric(repeated):null;
    const deltaClass=Number.isFinite(delta)?(delta<0?'delta-better':delta>0?'delta-worse':''):'';
    const title=`Unique: ${unique?`LR ${unique.lr}, WD ${unique.wd}`:'pending'} · Repeated: ${repeated?`LR ${repeated.lr}, WD ${repeated.wd}`:'pending'}`;
    const uniqueValue=fmt(unique&&metric(unique)),repeatedValue=fmt(repeated&&metric(repeated));
    const uniqueDisplay=unique&&unique===uniqueClosestToRepeatedBest[bs]?`<strong>${uniqueValue}</strong>`:uniqueValue;
    const repeatedDisplay=repeated&&repeated===repeatedColumnBest[bs]?`<strong>${repeatedValue}</strong>`:repeatedValue;
    return `<td class="endpoint-cell" title="${title}">${uniqueDisplay} / ${repeatedDisplay} / <span class="${deltaClass}">${signed(delta)}</span></td>`;
  }).join('')}</tr>`).join('');

  const matchedSpecs=[
    {optimizerSteps:119,epochs:{'256':0.125},showPending:true},
    ...(batch.optimizerStepComparisons||[]).map(comparison=>({...comparison,showPending:[238,477,954,1907].includes(Number(comparison.optimizerSteps))})),
  ];
  const matchedComparisons=matchedSpecs.map(comparison=>{
    const runs=summaryBatches.map(bs=>{
      const epoch=comparison.epochs?.[String(bs)];
      if(!Number.isFinite(epoch)||(!targets.includes(epoch)&&!fractionalTargets.includes(epoch)))return null;
      return uniqueMatchedRuns.filter(r=>r.batchSequences===bs&&r.epoch===epoch&&completeUnique(r,uniqueHealth)).sort((a,b)=>metric(a)-metric(b))[0]||null;
    });
    return {...comparison,runs};
  }).filter(comparison=>comparison.showPending||comparison.runs.filter(Boolean).length>=2);
  document.querySelector('#unique-optimizer-step-summary').innerHTML=matchedComparisons.map(comparison=>{
    const available=comparison.runs.filter(Boolean);
    const best=available.length?available.reduce((winner,run)=>metric(run)<metric(winner)?run:winner):null;
    const cells=comparison.runs.map((run,index)=>{
      if(!run){const epoch=comparison.epochs?.[String(summaryBatches[index])];return Number.isFinite(epoch)&&fractionalTargets.includes(epoch)?`<td class="matched-pending">${epoch}B · pending</td>`:'<td>—</td>';}
      const formatted=fmt(metric(run));
      return `<td class="matched-value ${run===best?'summary-best':''}" title="BS ${run.batchSequences} · LR ${run.lr} · WD ${run.wd}">${run.epoch}B · ${run===best?`<strong>${formatted}</strong>`:formatted}</td>`;
    }).join('');
    return `<tr><td>≈${Number(comparison.optimizerSteps).toLocaleString()}</td>${cells}</tr>`;
  }).join('')||'<tr><td colspan="4">No matched checkpoints available.</td></tr>';

  const groups=new Map();[...uniqueRuns,...uniquePlaceholders].forEach(r=>{const key=`${r.batchSequences}|${r.epoch}`;if(!groups.has(key))groups.set(key,[]);groups.get(key).push(r);});
  document.querySelector('#coordinate-grid').innerHTML=targets.flatMap(epoch=>[64,256,1024].map(bs=>{const runs=(groups.get(`${bs}|${epoch}`)||[]).sort((a,b)=>Number(a.lr)-Number(b.lr)),best=runs.filter(r=>completeUnique(r,uniqueHealth)).sort((a,b)=>metric(a)-metric(b))[0],chips=runs.map(r=>`<span class="tuple ${best===r?'selected':active(r)?'active':''}">(LR ${r.lr}, WD 0.033) · ${r.status}${Number.isFinite(metric(r))?` · CE ${fmt(metric(r))}`:''}</span>`).join('');return `<tr><td><strong>${epoch}B</strong></td><td>${bs}</td><td><div class="tuple-list">${chips||'—'}</div></td><td>${best?`LR ${best.lr} · CE ${fmt(metric(best))}`:'pending'}</td></tr>`;})).join('');
  const provenance=[...unique1024,...uniqueNewAll,...uniquePlaceholders].sort((a,b)=>a.batchSequences-b.batchSequences||a.epoch-b.epoch||Number(a.lr)-Number(b.lr));
  document.querySelector('#rows').innerHTML=provenance.map(r=>`<tr class="${(r.condition==='Unique'?uniqueHealth:repeatedHealth)[r.wandb]?'run-unhealthy':''}"><td>${r.condition}</td><td>${r.batchSequences}</td><td>${r.epoch}</td><td>${r.lr}</td><td>${r.wd||'0.033'}</td><td class="${active(r)?'run-active':''}">${r.status}</td><td>${fmt(r.train)}</td><td>${fmt(metric(r))}</td><td>${fmt(r.acc)}</td><td>${fmt(r.bpb)}</td><td>${link('wandb',r.wandb||r.activeWandb)}</td><td>${link('beaker',r.beaker)}</td></tr>`).join('');
})();
