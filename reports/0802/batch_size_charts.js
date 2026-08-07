(() => {
  const d=window.ICSL_REPORT_DATA||{batchSweeps:[],targetEpochs:[]};
  const baseline=window.ICSL_WD_BASELINE_DATA||{runs:[],fixedLrByEpoch:{}};
  document.querySelector('#title').textContent=d.title;
  document.querySelector('#setup').textContent=d.setup;
  document.querySelector('#updated').textContent=`Updated ${d.updated}`;
  document.querySelector('#selection').textContent=d.selection;

  const colors={'BS 64':'#2563eb','BS 256':'#7c3aed','BS 1024':'#dc5a39'};
  const series=['BS 64','BS 256','BS 1024'];
  const visible=r=>!['failed','canceled'].includes(r.status)&&r.kind!=='evaluation';
  const active=r=>['active','running','submitted','pending','queued','planned'].includes(r.status);
  const metric=r=>r.validation??r.c4;
  const complete=r=>r.status==='complete'&&Number.isFinite(metric(r));
  const key=r=>`${r.batchSequences}|${r.epoch}`;

  const newRuns=(d.batchSweeps||[]).flatMap(sweep=>(d.targetEpochs||[]).map((epoch,index)=>{
    const result=sweep.results?.[epoch]||{};
    return {...sweep,...result,epoch,series:`BS ${sweep.batchSequences}`,status:result.status||(epoch===sweep.activeEpoch?sweep.status:'queued')};
  })).filter(visible);
  const baselineRuns=(baseline.runs||[]).filter(r=>
    visible(r)&&r.status!=='queued'&&d.targetEpochs.includes(r.epoch)
  ).map(r=>({...r,batchSequences:1024,contextLength:4096,series:'BS 1024'}));
  const coordinateRuns=[...newRuns,...baselineRuns];
  const selected=new Map();
  coordinateRuns.filter(complete).forEach(r=>{
    const current=selected.get(key(r));
    if(!current||metric(r)<metric(current))selected.set(key(r),r);
  });
  const chartRuns=[...selected.values()];

  const legend=document.querySelector('#legend');
  series.forEach(s=>legend.insertAdjacentHTML('beforeend',`<label><span class="dot" style="background:${colors[s]}"></span>${s}</label>`));
  const avg8Acc=r=>{const q=['arc_challenge','arc_easy','csqa','acc','openbookqa','piqa','socialiqa','winogrande'].map(x=>x==='acc'?r.acc:r.downstream?.[x]).filter(Number.isFinite);return q.length===8?q.reduce((a,b)=>a+b,0)/8:null;};
  const value=(r,k)=>{
    if(k==='validation')return metric(r);
    if(k==='avg8_acc')return avg8Acc(r);
    if(k==='avg8_bpb')return r.avg8Bpb??baseline.avg8BpbByWandb?.[r.wandb]??null;
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
    const y=v=>200-(v-(lo-pad))/(hi-lo+2*pad)*160,x=e=>42+(e-1)/23*244;
    svg.innerHTML=`<line x1="36" y1="200" x2="286" y2="200" stroke="#aeb7c7"/><text x="2" y="203">${(lo-pad).toFixed(3)}</text><text x="2" y="42">${(hi+pad).toFixed(3)}</text>`;
    series.forEach((s,si)=>{
      const p=points.filter(z=>z.r.series===s).sort((a,b)=>a.r.epoch-b.r.epoch),color=colors[s];
      if(p.length>1)svg.insertAdjacentHTML('beforeend',`<polyline fill="none" stroke="${color}" stroke-width="2" points="${p.map(z=>`${x(z.r.epoch)},${y(z.v)}`).join(' ')}"/>`);
      p.forEach(z=>svg.insertAdjacentHTML('beforeend',`<circle cx="${x(z.r.epoch)}" cy="${y(z.v)}" r="4" fill="${color}"><title>${s} · ${label} ${z.v.toFixed(3)} · E${z.r.epoch} · LR ${z.r.lr} · WD ${z.r.wd}</title></circle><text x="${x(z.r.epoch)-5}" y="218">E${z.r.epoch}</text>`));
      if(p.length){const best=bestPoint(p,name),id=`${name}-best-${si}`,bx=x(best.r.epoch),by=y(best.v),direction=si%2?1:-1;svg.insertAdjacentHTML('beforeend',`<line data-best-label-id="${id}" x1="${bx}" y1="${by}" x2="${bx}" y2="${by+direction*18}" stroke="${color}" stroke-width="1"/><text class="series-best" data-best-label-id="${id}" data-point-x="${bx}" data-point-y="${by}" data-direction="${direction}" x="${bx}" y="${by+direction*18}" text-anchor="middle" font-size="8" font-weight="800" fill="${color}" stroke="#fff" stroke-width="3" paint-order="stroke">${s} · best ${best.v.toFixed(3)}</text>`);}
    });
    layoutBestLabels(svg,{minX:20,maxX:290,minY:12,maxY:190});
  });

  const grid=document.querySelector('#coordinate-grid');
  const groups=new Map();
  coordinateRuns.forEach(r=>{const g=key(r);if(!groups.has(g))groups.set(g,[]);groups.get(g).push(r);});
  grid.innerHTML=[...groups.entries()].sort((a,b)=>{const [ab,ae]=a[0].split('|'),[bb,be]=b[0].split('|');return Number(ae)-Number(be)||Number(ab)-Number(bb)}).map(([g,runs])=>{
    const [batch,epoch]=g.split('|'),best=selected.get(g);
    const ordered=runs.sort((a,b)=>Number(a.wd)-Number(b.wd)||Number(a.lr)-Number(b.lr));
    const chips=ordered.map(r=>`<span class="tuple ${best===r?'selected':active(r)?'active':''}">(LR ${r.lr}, WD ${r.wd}) · ${r.status}${Number.isFinite(metric(r))?` · CE ${metric(r).toFixed(3)}`:''}</span>`).join('');
    const selection=best?`LR ${best.lr}, WD ${best.wd} · CE ${metric(best).toFixed(3)}`:'pending';
    return `<tr><td><strong>E${epoch}</strong></td><td>${batch}</td><td><div class="tuple-list">${chips}</div></td><td>${selection}</td></tr>`;
  }).join('');

  const tableMetric=v=>Number.isFinite(v)?v.toFixed(3):'—';
  const selectedBaseline=[...selected.values()].filter(r=>r.batchSequences===1024);
  const provenance=[...newRuns,...selectedBaseline].sort((a,b)=>a.epoch-b.epoch||a.batchSequences-b.batchSequences||Number(a.lr)-Number(b.lr));
  const rows=document.querySelector('#rows');
  provenance.forEach(r=>rows.insertAdjacentHTML('beforeend',`<tr><td>${r.batchSequences}</td><td>${r.epoch}</td><td>${r.lr}</td><td>${r.wd}</td><td class="${active(r)?'run-active':''}">${r.status}</td><td>${tableMetric(r.train)}</td><td>${tableMetric(metric(r))}</td><td>${tableMetric(r.acc)}</td><td>${tableMetric(r.bpb)}</td><td>${r.wandb?`<a href="https://wandb.ai/ai2-llm/sewonm-icsl/runs/${r.wandb}">${r.wandb}</a>`:'—'}</td><td>${r.beaker?`<a href="https://beaker.org/ex/${r.beaker}">experiment</a>`:'—'}</td></tr>`));
})();
