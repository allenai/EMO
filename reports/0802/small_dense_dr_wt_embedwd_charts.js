(() => {
  const report=window.ICSL_REPORT_DATA||{};
  const chains=report.adaptiveDrWtEmbedWdChains||[];
  const escape=value=>String(value??'').replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
  const active=status=>['submitted','scheduled','running','pending','queued'].includes(status);
  const finite=value=>Number.isFinite(Number(value));
  const metric=value=>finite(value)?Number(value).toFixed(3):null;
  const statusFor=(chain,epoch,wd,result)=>{
    if(result)return result.status||'complete';
    if(Number(epoch)!==Number(chain.activeEpoch))return 'planned';
    if(String(wd)===String(chain.activeWd))return chain.status||'active';
    return chain.status==='failed'?'blocked':'queued';
  };
  const renderCoordinateGrid=()=>{
    const grid=document.querySelector('#coordinate-grid');
    if(!grid)return;
    const originalRows=Array.from(grid.querySelectorAll('tr'));
    originalRows.forEach(row=>{
      const batch=Number(row.cells[1]?.textContent.trim());
      const epoch=Number(row.cells[0]?.textContent.trim().replace(/^E/i,''));
      row.dataset.coordinateBatch=String(batch);
      row.dataset.coordinateMethod='0';
      row.dataset.coordinateEpoch=String(epoch);
      row.cells[1].textContent=`BS ${batch} · Original`;
    });
    const adaptiveRows=[];
    chains.slice().sort((a,b)=>Number(a.batchSequences)-Number(b.batchSequences)).forEach(chain=>{
      const epochs=new Set([
        ...Object.keys(chain.results||{}),
        ...Object.keys(chain.frontiers||{}),
        ...(chain.activeEpoch?[String(chain.activeEpoch)]:[]),
      ]);
      [...epochs].sort((a,b)=>Number(a)-Number(b)).forEach((epoch,index)=>{
        const results=chain.results?.[epoch]||{};
        const frontier=chain.frontiers?.[epoch]||{};
        const wds=new Set([
          ...Object.keys(results),
          ...(frontier.candidates||[]).map(String),
          ...(Number(epoch)===Number(chain.activeEpoch)?(chain.activeWds||[]).map(String):[]),
        ]);
        const chips=[...wds].sort((a,b)=>Number(a)-Number(b)).map(wd=>{
          const result=results[wd];
          const status=statusFor(chain,epoch,wd,result);
          const selected=String(frontier.selectedWd)===String(wd)&&frontier.status==='complete';
          const isCurrent=Number(epoch)===Number(chain.activeEpoch)&&String(chain.activeWd)===String(wd)&&!result;
          const unhealthy=['failed','blocked','canceled','cancelled'].includes(status);
          const progress=isCurrent&&chain.progress?.percent!=null?` · ${escape(chain.progress.percent)}%`:'';
          const validation=metric(result?.validationExact??result?.validation??frontier.candidateValidationExact?.[wd]);
          const classes=['tuple',selected?'selected':'',!selected&&active(status)?'active':'',unhealthy?'unhealthy':''].filter(Boolean).join(' ');
          const reason=unhealthy&&chain.reason?` title="${escape(chain.reason)}"`:'';
          return `<span class="${classes}"${reason}>(LR ${escape(result?.lr||chain.lr)}, WD ${escape(wd)}) · ${escape(status)}${validation?` · CE ${validation}`:''}${progress}</span>`;
        }).join('');
        const selection=frontier.status==='complete'&&frontier.selectedWd!=null
          ?`LR ${escape(chain.lr)}, WD ${escape(frontier.selectedWd)} · CE ${metric(frontier.selectedValidationExact)}`
          :Number(epoch)===Number(chain.activeEpoch)&&chain.status==='failed'
            ?'unresolved · failed'
            :'pending';
        const row=document.createElement('tr');
        if(index===0)row.classList.add('coordinate-method-start');
        row.dataset.coordinateBatch=String(chain.batchSequences);
        row.dataset.coordinateMethod='1';
        row.dataset.coordinateEpoch=String(epoch);
        row.innerHTML=`<td><strong>E${escape(epoch)}</strong></td><td>BS ${escape(chain.batchSequences)} · DR+WT+EmbedWD</td><td><div class="tuple-list">${chips||'—'}</div></td><td>${selection}</td>`;
        adaptiveRows.push(row);
      });
    });
    [...originalRows,...adaptiveRows]
      .sort((a,b)=>Number(a.dataset.coordinateBatch)-Number(b.dataset.coordinateBatch)||Number(a.dataset.coordinateMethod)-Number(b.dataset.coordinateMethod)||Number(a.dataset.coordinateEpoch)-Number(b.dataset.coordinateEpoch))
      .forEach(row=>grid.appendChild(row));
  };
  const phaseText=(chain,key,label)=>Object.entries(chain[key]||{})
    .sort((a,b)=>Number(a[0])-Number(b[0]))
    .map(([epoch,result])=>`${label} E${epoch}: CE ${metric(result.validationExact??result.validation)??'—'}`)
    .join(' → ');
  const frontierText=chain=>{
    if(['locked_wd_predecay_saturation_v1','locked_wd_requested_postdecay_finalizer_v1'].includes(chain.policy)){
      const pre=phaseText(chain,'preDecayResults','[PD]');
      const post=phaseText(chain,'postDecayResults','[POST]');
      return `${pre||'[PD] awaiting evaluation'}${post?` | ${post}`:''}`;
    }
    return Object.entries(chain.frontiers||{})
      .sort((a,b)=>Number(a[0])-Number(b[0]))
      .map(([epoch,frontier])=>`E${epoch}: WD${frontier.selectedWd} · ${Number(frontier.selectedValidationExact).toFixed(4)}`)
      .join(' → ')||'—';
  };
  const currentText=chain=>{
    if(['locked_wd_predecay_saturation_v1','locked_wd_requested_postdecay_finalizer_v1'].includes(chain.policy)){
      if(chain.postDecaySelection){
        if(chain.policy==='locked_wd_requested_postdecay_finalizer_v1'){
          const outcome=chain.postDecaySelection.postDecaySaturated?'saturated':'user-stopped';
          return `[POST] ${outcome} at E${chain.postDecaySelection.evaluatedThroughEpoch||chain.saturatedEpoch||'—'}; selected E${chain.selectedPostDecayEpoch} · CE ${metric(chain.selectedPostDecayValidationExact)}`;
        }
        return `[PD] saturated E${chain.saturatedEpoch}; [POST] selected E${chain.selectedPostDecayEpoch} · CE ${metric(chain.selectedPostDecayValidationExact)}`;
      }
      const label=String(chain.activePhase||'').startsWith('post')?'[POST]':'[PD]';
      const progress=chain.progress?` · ${chain.progress.percent}%`:'';
      return `${label} ${chain.activePhase||'pending'}${chain.activeEpoch?` · E${chain.activeEpoch}`:''} · locked WD${chain.lockedWd}${progress}`;
    }
    if(chain.saturatedEpoch)return `saturated at E${chain.saturatedEpoch}`;
    if(!chain.activeEpoch)return '—';
    const wds=(chain.activeWds||[]).map(wd=>`WD${wd}`).join(', ');
    const progress=chain.progress?` · ${chain.progress.percent}%` : '';
    return `E${chain.activeEpoch}${wds?` · ${wds}`:''}${progress}`;
  };
  renderCoordinateGrid();
  const body=document.querySelector('#dr-wt-embwd-chains');
  if(body)body.innerHTML=chains
      .slice()
      .sort((a,b)=>Number(a.batchSequences)-Number(b.batchSequences))
      .map(chain=>{
        const topology=`${chain.gpuCount} GPU · microbatch ${chain.rankMicrobatchSequences} · accum ${chain.gradientAccumulation}`;
        const initial=(chain.initialWds||[]).map(wd=>`WD${wd}`).join(', ');
        const status=escape(chain.status||'planned');
        return `<tr class="${active(chain.status)?'run-active':''}">
          <td>BS ${escape(chain.batchSequences)}</td>
          <td>${escape(chain.variant)}</td>
          <td>${escape(chain.lr)}</td>
          <td>${escape(initial)}</td>
          <td>${escape(topology)}</td>
          <td>${status}</td>
          <td>${escape(currentText(chain))}</td>
          <td>${escape(frontierText(chain))}</td>
          <td><a href="https://beaker.org/ex/${escape(chain.experiment)}">${escape(chain.experiment)}</a></td>
        </tr>`;
      }).join('');
})();
