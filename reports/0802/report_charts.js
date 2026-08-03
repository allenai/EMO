(() => {
  const d=window.ICSL_REPORT_DATA||{runs:[]};
  title.textContent=d.title; setup.textContent=d.setup; updated.textContent=`Updated ${d.updated}`; selection.textContent=d.selection;
  const colors=['#2563eb','#7c3aed','#dc5a39','#0f9d76','#e09200','#ca3f64','#1874a8','#7f6752'];
  const visible=r=>!['failed','canceled'].includes(r.status)&&r.kind!=='evaluation';
  const visibleRuns=d.runs.filter(visible);
  const key=r=>r.series||r.scheduler||`WD ${r.wd}`;
  // Charts show exactly one point per series and epoch: the completed LR with
  // the lowest DCLM validation CE. The coordinate grid and provenance table
  // continue to expose every evaluated or active LR.
  const chartGroups=new Map();
  visibleRuns.filter(r=>r.status==='complete'&&Number.isFinite(r.validation??r.c4)&&(d.coordinateMode!=='fixed-step2-lr'||r.lr===d.fixedLrByEpoch?.[r.epoch])).forEach(r=>{
    const group=`${key(r)}|${r.epoch}`, current=chartGroups.get(group);
    if(!current||(r.validation??r.c4)<(current.validation??current.c4))chartGroups.set(group,r);
  });
  let chartRuns=[...chartGroups.values()];
  if(d.chartMode==='untuned-vs-tuned'){
    chartRuns=[];
    const epochs=[...new Set(visibleRuns.map(r=>r.epoch))].sort((a,b)=>a-b);
    epochs.forEach(epoch=>{
      const fixed=d.fixedLrByEpoch?.[epoch];
      const eligible=visibleRuns.filter(r=>r.epoch===epoch&&r.status==='complete'&&r.lr===fixed&&Number.isFinite(r.validation??r.c4));
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
  const metrics=[['train','Train CE'],['validation','DCLM validation CE'],['acc','HellaSwag accuracy'],['bpb','HellaSwag BPB'],['avg8','8-task average (no BoolQ)'],['stable3','Stable 3-task average'],['arc_easy','ARC-Easy'],['arc_challenge','ARC-Challenge'],['csqa','CSQA'],['openbookqa','OpenBookQA'],['piqa','PIQA'],['socialiqa','SocialIQA'],['winogrande','Winogrande']];
  const value=(r,k)=>{if(k==='validation')return r.validation??r.c4;if(k==='avg8'){const q=['arc_challenge','arc_easy','csqa','acc','openbookqa','piqa','socialiqa','winogrande'].map(x=>x==='acc'?r.acc:r.downstream?.[x]).filter(Number.isFinite);return q.length===8?q.reduce((a,b)=>a+b,0)/8:null}if(k==='stable3'){const q=[r.downstream?.arc_easy,r.acc,r.downstream?.piqa].filter(Number.isFinite);return q.length===3?q.reduce((a,b)=>a+b,0)/3:null}return r[k]??r.downstream?.[k]};
  metrics.forEach(([m,label])=>{const pts=chartRuns.map((r,i)=>({r,i,v:value(r,m)})).filter(x=>Number.isFinite(x.v));const card=document.createElement('div');card.className='card chart';card.innerHTML=`<h3>${label}</h3>`;const svg=document.createElementNS('http://www.w3.org/2000/svg','svg');svg.setAttribute('viewBox','0 0 310 240');card.append(svg);charts.append(card);if(!pts.length){svg.innerHTML='<text x="18" y="36">Pending results</text>';return}const lo=Math.min(...pts.map(x=>x.v)),hi=Math.max(...pts.map(x=>x.v)),pad=Math.max((hi-lo)*.18,.01),y=v=>200-(v-(lo-pad))/(hi-lo+2*pad)*160,x=e=>42+(e-1)*55,clamp=(v,a,b)=>Math.max(a,Math.min(b,v));svg.innerHTML=`<line x1="36" y1="200" x2="286" y2="200" stroke="#aeb7c7"/><text x="2" y="203">${(lo-pad).toFixed(3)}</text><text x="2" y="42">${(hi+pad).toFixed(3)}</text>`;series.forEach((s,si)=>{const p=pts.filter(z=>chartKey(z.r)===s).sort((a,b)=>a.r.epoch-b.r.epoch);const color=colors[si%colors.length],dx=((si%3)-1)*15,dy=-12-(Math.floor(si/3)%2)*13;if(p.length>1)svg.insertAdjacentHTML('beforeend',`<polyline fill="none" stroke="${color}" stroke-width="2" points="${p.map(z=>`${x(z.r.epoch)},${y(z.v)}`).join(' ')}"/>`);p.forEach(z=>{const ax=clamp(x(z.r.epoch)+dx,20,290),ay=clamp(y(z.v)+dy,12,190);svg.insertAdjacentHTML('beforeend',`<line x1="${x(z.r.epoch)}" y1="${y(z.v)}" x2="${ax}" y2="${ay}" stroke="${color}" stroke-width="0.8" opacity="0.7"/><circle cx="${x(z.r.epoch)}" cy="${y(z.v)}" r="4" fill="${color}"><title>${label} ${z.v.toFixed(3)} · epoch ${z.r.epoch} · LR ${z.r.lr||'—'} · WD ${z.r.wd||'—'}</title></circle><text class="value" text-anchor="middle" x="${ax}" y="${ay+3}" fill="${color}">${z.v.toFixed(3)}</text><text x="${x(z.r.epoch)-5}" y="218">E${z.r.epoch}</text>`);})})});
  const tableMetric=v=>Number.isFinite(v)?v.toFixed(3):'—';
  visibleRuns.forEach(r=>rows.insertAdjacentHTML('beforeend',`<tr><td>${key(r)}</td><td>${r.epoch}</td><td>${r.lr||'—'}</td><td>${r.wd||'—'}</td><td>${r.status}</td><td>${tableMetric(r.train)}</td><td>${tableMetric(r.validation??r.c4)}</td><td>${tableMetric(r.acc)}</td><td>${tableMetric(r.bpb)}</td><td>${r.wandb?`<a href="https://wandb.ai/ai2-llm/sewonm-icsl/runs/${r.wandb}">${r.wandb}</a>`:'—'}</td><td>${r.beaker?`<a href="https://beaker.org/ex/${r.beaker}">experiment</a>`:'—'}</td></tr>`));
  const gridMount=document.querySelector('#coordinate-grid');
  if(gridMount){
    if(d.coordinateMode==='fixed-step2-lr'){
      const section=gridMount.closest('section'),table=gridMount.closest('table');
      section.querySelector('h2').textContent='Step 1 coordinate grid — (LR, WD), one flat row per epoch';
      table.querySelector('thead tr').innerHTML='<th>Epoch</th><th>Completed (LR, WD) coordinates</th>';
      const epochs=[...new Set(visibleRuns.map(r=>r.epoch))].sort((a,b)=>a-b);
      gridMount.innerHTML=epochs.map(epoch=>{
        const ordered=visibleRuns.filter(r=>r.epoch===epoch).sort((a,b)=>Number(a.wd)-Number(b.wd)||Number(a.lr)-Number(b.lr));
        const complete=ordered.filter(r=>r.status==='complete'&&Number.isFinite(r.validation??r.c4));
        const fixed=d.fixedLrByEpoch?.[epoch];
        const eligible=complete.filter(r=>r.lr===fixed);
        const selected=eligible.length?eligible.reduce((winner,r)=>(r.validation??r.c4)<(winner.validation??winner.c4)?r:winner):null;
        const chips=ordered.map(r=>`<span class="tuple ${selected===r?'selected':['active','queued','planned'].includes(r.status)?'active':''}">(LR ${r.lr||'—'}, WD ${r.wd||'—'}) · ${r.status}${Number.isFinite(r.validation??r.c4)?` · CE ${(r.validation??r.c4).toFixed(3)}`:''}</span>`).join('');
        return `<tr><td><strong>E${epoch}</strong></td><td><div class="tuple-list">${chips||'<span class="pending">—</span>'}</div></td></tr>`;
      }).join('');
    }else{
      gridMount.closest('table').querySelector('thead tr').innerHTML='<th>Epoch</th><th>Series</th><th>Evaluated and active LRs</th>';
      const groups=new Map();
      visibleRuns.forEach(r=>{const g=`${r.epoch}|${key(r)}`;if(!groups.has(g))groups.set(g,[]);groups.get(g).push(r);});
      gridMount.innerHTML=[...groups.entries()].sort((a,b)=>{const [ae,ak]=a[0].split('|'),[be,bk]=b[0].split('|');return Number(ae)-Number(be)||ak.localeCompare(bk)}).map(([g,runs])=>{const [epoch,label]=g.split('|');const ordered=runs.sort((a,b)=>Number(a.lr)-Number(b.lr));const complete=ordered.filter(r=>r.status==='complete'&&Number.isFinite(r.validation??r.c4));const best=complete.length?complete.reduce((winner,r)=>(r.validation??r.c4)<(winner.validation??winner.c4)?r:winner):null;const chips=ordered.map(r=>`<span class="tuple ${best===r?'selected':['active','queued','planned'].includes(r.status)?'active':''}">${r.lr||'—'} · ${r.status}${Number.isFinite(r.validation??r.c4)?` · ${(r.validation??r.c4).toFixed(3)}`:''}</span>`).join('');return `<tr><td><strong>E${epoch}</strong></td><td>${label}</td><td><div class="tuple-list">${chips}</div></td></tr>`;}).join('');
    }
  }
})();
