(() => {
  const d=window.ICSL_REPORT_DATA||{runs:[]};
  const healthStyles=document.createElement('style');
  healthStyles.textContent=`
    .health-key{display:inline-block;width:14px;height:14px;border:0;border-radius:4px;background:#fecaca;vertical-align:-2px}
    .health-key.suspicious{background:#fb923c}
    .tuple.unhealthy{background:#fff1f2!important;font-weight:400!important}
    .tuple.suspicious{background:#fed7aa!important;color:#9a3412!important;font-weight:400!important}
    .run-unhealthy td{background:#fff1f2;font-weight:400}
    .run-suspicious td{background:#ffedd5;font-weight:400}
  `;
  document.head.append(healthStyles);
  title.textContent=d.title; setup.textContent=d.setup; updated.textContent=`Updated ${d.updated}`; selection.textContent=d.selection;
  const colors=['#2563eb','#7c3aed','#dc5a39','#0f9d76','#e09200','#ca3f64','#1874a8','#7f6752'];
  const healthMap=d.healthAudit?.unhealthy||{};
  const suspiciousMap=d.healthAudit?.suspicious||{};
  const isUnhealthy=r=>Boolean(r.wandb&&healthMap[r.wandb]);
  const isSuspicious=r=>Boolean(r.wandb&&suspiciousMap[r.wandb]&&!isUnhealthy(r));
  const healthReason=r=>healthMap[r.wandb]?.reason||suspiciousMap[r.wandb]?.reason||'Health status unavailable.';
  const healthClass=r=>isUnhealthy(r)?'run-unhealthy':isSuspicious(r)?'run-suspicious':'';
  const tupleHealthClass=r=>isUnhealthy(r)?'unhealthy':isSuspicious(r)?'suspicious':'';
  const visible=r=>(!['failed','canceled'].includes(r.status)||isUnhealthy(r)||isSuspicious(r))&&r.kind!=='evaluation';
  const isActiveStatus=r=>['active','running','queued','planned'].includes(r.status);
  const visibleRuns=d.runs.filter(visible);
  const key=r=>r.series||r.scheduler||`WD ${r.wd}`;
  // Charts show exactly one point per series and epoch: the completed LR with
  // the lowest DCLM validation CE. The coordinate grid and provenance table
  // continue to expose every evaluated or active LR.
  const chartGroups=new Map();
  visibleRuns.filter(r=>r.status==='complete'&&!isUnhealthy(r)&&Number.isFinite(r.validation??r.c4)&&(d.coordinateMode!=='fixed-step2-lr'||r.lr===d.fixedLrByEpoch?.[r.epoch])).forEach(r=>{
    const group=`${key(r)}|${r.epoch}`, current=chartGroups.get(group);
    if(!current||(r.validation??r.c4)<(current.validation??current.c4))chartGroups.set(group,r);
  });
  let chartRuns=[...chartGroups.values()];
  if(d.chartMode==='untuned-vs-tuned'){
    chartRuns=[];
    const epochs=[...new Set(visibleRuns.map(r=>r.epoch))].sort((a,b)=>a-b);
    epochs.forEach(epoch=>{
      const fixed=d.fixedLrByEpoch?.[epoch];
      const eligible=visibleRuns.filter(r=>r.epoch===epoch&&r.status==='complete'&&!isUnhealthy(r)&&r.lr===fixed&&Number.isFinite(r.validation??r.c4));
      const baseline=eligible.filter(r=>String(r.wd)==='0.033').sort((a,b)=>(a.validation??a.c4)-(b.validation??b.c4))[0];
      if(baseline)chartRuns.push({...baseline,chartSeries:'Untuned WD (0.033)'});
      const tunedCandidates=eligible.filter(r=>String(r.wd)!=='0.033');
      if(tunedCandidates.length){
        const tuned=[...eligible].sort((a,b)=>(a.validation??a.c4)-(b.validation??b.c4))[0];
        chartRuns.push({...tuned,chartSeries:'Tuned WD'});
      }
    });
  }
  const chartKey=r=>r.chartSeries||key(r);
  const series=d.chartMode==='untuned-vs-tuned'?['Untuned WD (0.033)','Tuned WD']:[...new Set(visibleRuns.map(key))];
  series.forEach((s,i)=>legend.insertAdjacentHTML('beforeend',`<label><span class="dot" style="background:${colors[i%colors.length]}"></span>${s}</label>`));
  if(Object.keys(suspiciousMap).length)legend.insertAdjacentHTML('beforeend','<label title="Advisory only; remains eligible for selection"><span class="health-key suspicious"></span>Suspicious CE/gradient trajectory</label>');
  if(Object.keys(healthMap).length)legend.insertAdjacentHTML('beforeend','<label title="Excluded from endpoint selection"><span class="health-key unhealthy"></span>Unhealthy CE/gradient trajectory</label>');
  const metrics=d.chartMode==='untuned-vs-tuned'
    ? [['validation','DCLM validation CE'],['bpb','HellaSwag non-v2 BPB'],['acc','HellaSwag length-normalized accuracy'],['avg8_bpb','8-task average BPB · no BoolQ'],['avg8_acc','8-task average accuracy · no BoolQ']]
    : [['train','Train CE'],['validation','DCLM validation CE'],['acc','HellaSwag accuracy'],['bpb','HellaSwag BPB'],['avg8','8-task average (no BoolQ)'],['stable3','Stable 3-task average'],['arc_easy','ARC-Easy'],['arc_challenge','ARC-Challenge'],['csqa','CSQA'],['openbookqa','OpenBookQA'],['piqa','PIQA'],['socialiqa','SocialIQA'],['winogrande','Winogrande']];
  const bestOnly=String(d.title||'').startsWith('Step 1');
  const value=(r,k)=>{if(k==='validation')return r.validation??r.c4;if(k==='avg8'||k==='avg8_acc'){const q=['arc_challenge','arc_easy','csqa','acc','openbookqa','piqa','socialiqa','winogrande'].map(x=>x==='acc'?r.acc:r.downstream?.[x]).filter(Number.isFinite);return q.length===8?q.reduce((a,b)=>a+b,0)/8:null}if(k==='avg8_bpb')return r.avg8Bpb??d.avg8BpbByWandb?.[r.wandb]??null;if(k==='stable3'){const q=[r.downstream?.arc_easy,r.acc,r.downstream?.piqa].filter(Number.isFinite);return q.length===3?q.reduce((a,b)=>a+b,0)/3:null}return r[k]??r.downstream?.[k]};
  const higherIsBetter=k=>['acc','avg8','avg8_acc','stable3','arc_easy','arc_challenge','csqa','openbookqa','piqa','socialiqa','winogrande'].includes(k);
  const bestPoint=(points,k)=>points.reduce((best,point)=>(higherIsBetter(k)?point.v>best.v:point.v<best.v)?point:best);
  const layoutBestLabels=(svg,bounds)=>{
    const items=[...svg.querySelectorAll('text.series-best')].map((node,index)=>{
      const pointX=Number(node.dataset.pointX),pointY=Number(node.dataset.pointY);
      node.setAttribute('x',pointX);node.setAttribute('y',pointY);
      const box=node.getBBox();
      return {node,index,pointX,pointY,height:box.height,topOffset:box.y-pointY,leader:svg.querySelector(`line[data-best-label-id="${node.dataset.bestLabelId}"]`)};
    }).sort((a,b)=>a.pointY-b.pointY||a.index-b.index);
    if(!items.length)return;
    const textHeight=items.reduce((sum,item)=>sum+item.height,0),gap=items.length>1?Math.max(0,Math.min(3,(bounds.maxY-bounds.minY-textHeight)/(items.length-1))):0;
    items.forEach((item,index)=>{
      const desired=Math.max(bounds.minY,Math.min(bounds.maxY-item.height,item.pointY+item.topOffset));
      item.top=index?Math.max(desired,items[index-1].top+items[index-1].height+gap):desired;
    });
    for(let index=items.length-1;index>=0;index--){
      const ceiling=index===items.length-1?bounds.maxY-items[index].height:items[index+1].top-items[index].height-gap;
      items[index].top=Math.min(items[index].top,ceiling);
    }
    if(items[0].top<bounds.minY){items[0].top=bounds.minY;for(let index=1;index<items.length;index++)items[index].top=Math.max(items[index].top,items[index-1].top+items[index-1].height+gap);}
    items.forEach(item=>{
      const y=item.top-item.topOffset;let x=item.pointX;
      item.node.setAttribute('x',x);item.node.setAttribute('y',y);
      const box=item.node.getBBox();
      if(box.x<bounds.minX)x+=bounds.minX-box.x;if(box.x+box.width>bounds.maxX)x-=box.x+box.width-bounds.maxX;
      item.node.setAttribute('x',x);if(item.leader){item.leader.setAttribute('x2',x);item.leader.setAttribute('y2',y);}
    });
  };
  const layoutPointLabels=(svg,bounds)=>{
    const occupied=[];
    [...svg.querySelectorAll('text.value')].forEach(node=>{
      const leader=svg.querySelector(`line[data-label-id="${node.dataset.labelId}"]`),preferredX=Number(node.getAttribute('x')),preferredY=Number(node.getAttribute('y'))-3,direction=Number(node.dataset.direction)||1,candidates=[];
      [0,12,24,36,48,60,72,84].forEach(distance=>[0,-10,10,-20,20,-30,30].forEach(horizontal=>candidates.push({x:preferredX+horizontal,y:preferredY+direction*distance})));[12,24,36,48,60,72,84].forEach(distance=>[0,-10,10,-20,20,-30,30].forEach(horizontal=>candidates.push({x:preferredX+horizontal,y:preferredY-direction*distance})));
      let chosen=null;for(const candidate of candidates){const x=Math.max(bounds.minX,Math.min(bounds.maxX,candidate.x)),y=Math.max(bounds.minY,Math.min(bounds.maxY,candidate.y));node.setAttribute('x',x);node.setAttribute('y',y+3);const box=node.getBBox(),rect={left:box.x-2,right:box.x+box.width+2,top:box.y-1,bottom:box.y+box.height+1};if(!occupied.some(other=>rect.left<other.right&&rect.right>other.left&&rect.top<other.bottom&&rect.bottom>other.top)){chosen={x,y,rect};break;}}
      if(!chosen){const x=Math.max(bounds.minX,Math.min(bounds.maxX,preferredX)),y=Math.max(bounds.minY,Math.min(bounds.maxY,preferredY));node.setAttribute('x',x);node.setAttribute('y',y+3);const box=node.getBBox();chosen={x,y,rect:{left:box.x-2,right:box.x+box.width+2,top:box.y-1,bottom:box.y+box.height+1}};}occupied.push(chosen.rect);if(leader){leader.setAttribute('x2',chosen.x);leader.setAttribute('y2',chosen.y);}
    });
  };
  metrics.forEach(([m,label])=>{
    const pts=chartRuns.map((r,i)=>({r,i,v:value(r,m)})).filter(x=>Number.isFinite(x.v));
    const card=document.createElement('div');card.className='card chart';card.innerHTML=`<h3>${label}</h3>`;
    const svg=document.createElementNS('http://www.w3.org/2000/svg','svg');svg.setAttribute('viewBox','0 0 310 240');card.append(svg);charts.append(card);
    if(!pts.length){svg.innerHTML='<text x="18" y="36">Pending results</text>';return;}
    const lo=Math.min(...pts.map(x=>x.v)),hi=Math.max(...pts.map(x=>x.v)),pad=Math.max((hi-lo)*.18,.01),y=v=>200-(v-(lo-pad))/(hi-lo+2*pad)*160,minEpoch=Math.min(...pts.map(x=>x.r.epoch)),maxEpoch=Math.max(...pts.map(x=>x.r.epoch)),x=e=>maxEpoch===minEpoch?164:42+(e-minEpoch)/(maxEpoch-minEpoch)*244,clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
    svg.innerHTML=`<line x1="36" y1="200" x2="286" y2="200" stroke="#aeb7c7"/><text x="2" y="203">${(lo-pad).toFixed(3)}</text><text x="2" y="42">${(hi+pad).toFixed(3)}</text>`;
    series.forEach((s,si)=>{
      const p=pts.filter(z=>chartKey(z.r)===s).sort((a,b)=>a.r.epoch-b.r.epoch),color=colors[si%colors.length],paired=d.chartMode==='untuned-vs-tuned',dx=paired?(si===0?-24:24):((si%3)-1)*15,dy=paired?(si===0?-16:18):-12-(Math.floor(si/3)%2)*13;
      if(p.length>1)svg.insertAdjacentHTML('beforeend',`<polyline fill="none" stroke="${color}" stroke-width="2" points="${p.map(z=>`${x(z.r.epoch)},${y(z.v)}`).join(' ')}"/>`);
      if(bestOnly){
        p.forEach(z=>svg.insertAdjacentHTML('beforeend',`<circle cx="${x(z.r.epoch)}" cy="${y(z.v)}" r="4" fill="${color}"><title>${label} ${z.v.toFixed(3)} · epoch ${z.r.epoch} · LR ${z.r.lr||'—'} · WD ${z.r.wd||'—'}</title></circle><text x="${x(z.r.epoch)-5}" y="218">E${z.r.epoch}</text>`));
        if(p.length){const best=bestPoint(p,m),id=`${m}-best-${si}`,bx=x(best.r.epoch),by=y(best.v),direction=si%2?1:-1;svg.insertAdjacentHTML('beforeend',`<line data-best-label-id="${id}" x1="${bx}" y1="${by}" x2="${bx}" y2="${by+direction*18}" stroke="${color}" stroke-width="1"/><text class="series-best" data-best-label-id="${id}" data-point-x="${bx}" data-point-y="${by}" data-direction="${direction}" x="${bx}" y="${by+direction*18}" text-anchor="middle" font-size="8" font-weight="800" fill="${color}" stroke="#fff" stroke-width="3" paint-order="stroke">${s} · best ${best.v.toFixed(3)}</text>`);}
      }else{
        p.forEach((z,pi)=>{const ax=clamp(x(z.r.epoch)+dx,20,290),ay=clamp(y(z.v)+dy,12,190),id=`${m}-${si}-${pi}`;svg.insertAdjacentHTML('beforeend',`<line data-label-id="${id}" x1="${x(z.r.epoch)}" y1="${y(z.v)}" x2="${ax}" y2="${ay}" stroke="${color}" stroke-width="0.8" opacity="0.7"/><circle cx="${x(z.r.epoch)}" cy="${y(z.v)}" r="4" fill="${color}"><title>${label} ${z.v.toFixed(3)} · epoch ${z.r.epoch} · LR ${z.r.lr||'—'} · WD ${z.r.wd||'—'}</title></circle><text class="value" data-label-id="${id}" data-direction="${dy>=0?1:-1}" text-anchor="middle" x="${ax}" y="${ay+3}" fill="${color}">${z.v.toFixed(3)}</text><text x="${x(z.r.epoch)-5}" y="218">E${z.r.epoch}</text>`);});
      }
    });
    (bestOnly?layoutBestLabels:layoutPointLabels)(svg,{minX:20,maxX:290,minY:12,maxY:190});
  });
  const tableMetric=v=>Number.isFinite(v)?v.toFixed(3):'—';
  visibleRuns.forEach(r=>rows.insertAdjacentHTML('beforeend',`<tr class="${healthClass(r)}" title="${isUnhealthy(r)||isSuspicious(r)?healthReason(r):(r.reason||'')}"><td>${key(r)}</td><td>${r.epoch}</td><td>${r.lr||'—'}</td><td>${r.wd||'—'}</td><td class="${isActiveStatus(r)?'run-active':''}">${r.status}</td><td>${tableMetric(r.train)}</td><td>${tableMetric(r.validation??r.c4)}</td><td>${tableMetric(r.acc)}</td><td>${tableMetric(r.bpb)}</td><td>${r.wandb?`<a href="https://wandb.ai/ai2-llm/sewonm-icsl/runs/${r.wandb}">${r.wandb}</a>`:'—'}</td><td>${r.beaker?`<a href="https://beaker.org/ex/${r.beaker}">experiment</a>`:'—'}</td></tr>`));
  const gridMount=document.querySelector('#coordinate-grid');
  if(gridMount){
    if(d.coordinateMode==='fixed-step2-lr'){
      const section=gridMount.closest('section'),table=gridMount.closest('table');
      section.querySelector('h2').textContent='Step 1 coordinate grid — (LR, WD), one flat row per epoch';
      table.querySelector('thead tr').innerHTML='<th>Epoch</th><th>Completed (LR, WD) coordinates</th>';
      const epochs=[...new Set(visibleRuns.map(r=>r.epoch))].sort((a,b)=>a-b);
      gridMount.innerHTML=epochs.map(epoch=>{
        const ordered=visibleRuns.filter(r=>r.epoch===epoch).sort((a,b)=>Number(a.wd)-Number(b.wd)||Number(a.lr)-Number(b.lr));
        const complete=ordered.filter(r=>r.status==='complete'&&!isUnhealthy(r)&&Number.isFinite(r.validation??r.c4));
        const fixed=d.fixedLrByEpoch?.[epoch];
        const eligible=complete.filter(r=>r.lr===fixed);
        const selected=eligible.length?eligible.reduce((winner,r)=>(r.validation??r.c4)<(winner.validation??winner.c4)?r:winner):null;
        const chips=ordered.map(r=>`<span class="tuple ${tupleHealthClass(r)|| (selected===r?'selected':isActiveStatus(r)?'active':'')}" title="${isUnhealthy(r)||isSuspicious(r)?healthReason(r):(r.reason||'')}">(LR ${r.lr||'—'}, WD ${r.wd||'—'}) · ${r.status}${Number.isFinite(r.validation??r.c4)?` · CE ${(r.validation??r.c4).toFixed(3)}`:''}</span>`).join('');
        return `<tr><td><strong>E${epoch}</strong></td><td><div class="tuple-list">${chips||'<span class="pending">—</span>'}</div></td></tr>`;
      }).join('');
    }else{
      gridMount.closest('table').querySelector('thead tr').innerHTML='<th>Epoch</th><th>Series</th><th>Evaluated and active LRs</th>';
      const groups=new Map();
      visibleRuns.forEach(r=>{const g=`${r.epoch}|${key(r)}`;if(!groups.has(g))groups.set(g,[]);groups.get(g).push(r);});
      gridMount.innerHTML=[...groups.entries()].sort((a,b)=>{const [ae,ak]=a[0].split('|'),[be,bk]=b[0].split('|');return Number(ae)-Number(be)||ak.localeCompare(bk)}).map(([g,runs])=>{const [epoch,label]=g.split('|');const ordered=runs.sort((a,b)=>Number(a.lr)-Number(b.lr));const complete=ordered.filter(r=>r.status==='complete'&&!isUnhealthy(r)&&Number.isFinite(r.validation??r.c4));const best=complete.length?complete.reduce((winner,r)=>(r.validation??r.c4)<(winner.validation??winner.c4)?r:winner):null;const chips=ordered.map(r=>`<span class="tuple ${tupleHealthClass(r)||(best===r?'selected':isActiveStatus(r)?'active':'')}" title="${isUnhealthy(r)||isSuspicious(r)?healthReason(r):(r.reason||'')}">${r.lr||'—'} · ${r.status}${Number.isFinite(r.validation??r.c4)?` · ${(r.validation??r.c4).toFixed(3)}`:''}</span>`).join('');return `<tr><td><strong>E${epoch}</strong></td><td>${label}</td><td><div class="tuple-list">${chips}</div></td></tr>`;}).join('');
    }
  }
})();
