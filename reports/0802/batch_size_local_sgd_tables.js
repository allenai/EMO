(()=>{
  const data=window.ICSL_BATCH_SIMULATION_DATA||{columns:[],runs:[]};
  const columns=data.columns||[];
  const columnByKey=new Map(columns.map(column=>[column.key,column]));
  const escapeHtml=value=>String(value??'').replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;');
  const headerCell=label=>`<th>${escapeHtml(label).replaceAll('\n','<br>')}</th>`;
  const methodLabel=column=>`<span class="method-label">${escapeHtml(column?.tableLabel||column?.label||'—').replaceAll('\n','<br>')}</span>`;
  const labelLines=column=>String(column?.tableLabel||column?.label||'—').split('\n').map(line=>line.trim()).filter(Boolean);
  const gridMethodLabel=column=>{
    const lines=labelLines(column).filter(line=>
      !/^H=\d+$/i.test(line)&&
      !/^simBS\d+$/i.test(line)&&
      !/\binit$/i.test(line)&&
      !/^scratch$/i.test(line)
    );
    return `<span class="method-label">${lines.map(escapeHtml).join('<br>')}</span>`;
  };
  const gridH=column=>finite(column?.syncInterval)?`H=${Number(column.syncInterval)}`:'—';
  const gridSimBS=column=>finite(column?.simulatedBatchSequences)?Number(column.simulatedBatchSequences):'—';
  const gridInit=column=>{
    const line=labelLines(column).find(value=>/\binit$/i.test(value)||/^scratch$/i.test(value));
    if(line)return line.replace(/\s+init$/i,'');
    const initialization=String(column?.initialization||'');
    if(/^scratch$/i.test(initialization))return 'scratch';
    const match=initialization.match(/(BS\d+).*?\b(E\d+)\b/i);
    return match?`${match[1]} ${match[2]}`:'—';
  };
  const simulationHeaders=columns.map(column=>headerCell(column.tableLabel||column.label)).join('');
  const validationHeader=document.querySelector('#validation-summary')?.closest('table')?.querySelector('thead tr');
  const baselineHeaders=[64,128,256,512,1024].flatMap(batch=>[`BS ${batch}\nOriginal`,`BS ${batch}\nDR`]);
  if(validationHeader)validationHeader.innerHTML=['Epoch',...baselineHeaders].map(headerCell).join('')+simulationHeaders;
  const optimizerHeader=document.querySelector('#optimizer-step-summary')?.closest('table')?.querySelector('thead tr');
  if(optimizerHeader)optimizerHeader.innerHTML=['Optimizer steps','Training time',...baselineHeaders].map(headerCell).join('')+simulationHeaders;
  const coordinateHeader=document.querySelector('#ls-coordinate-summary')?.closest('table')?.querySelector('thead tr');
  if(coordinateHeader)coordinateHeader.innerHTML=headerCell('Epoch')+simulationHeaders;
  const escapeAttribute=value=>String(value??'').replaceAll('&','&amp;').replaceAll('"','&quot;').replaceAll('<','&lt;').replaceAll('>','&gt;');
  const finite=value=>Number.isFinite(Number(value));
  const metric=value=>finite(value)?Number(value).toFixed(3):'—';
  const epochForAttempt=run=>Number(run.targetEpoch??run.chainThrough??run.startEpoch);
  const labelFor=run=>methodLabel(columnByKey.get(run.method)||{label:run.method});
  const resultRows=[];
  const attemptRows=[];
  const displayedMethods=new Set(columns.map(column=>column.key));

  (data.runs||[]).filter(run=>displayedMethods.has(run.method)).forEach(run=>{
    const results=Object.entries(run.results||{});
    results.forEach(([epoch,result])=>resultRows.push({
      ...run,...result,
      train:result.train??result.trainCe,
      validation:result.validation??result.validationCe,
      acc:result.acc??result.hellaswagAccuracy,
      bpb:result.bpb??result.hellaswagBpb,
      epoch:Number(epoch),status:result.status||run.status,sourceRun:run
    }));
    const represented=new Set(results.map(([epoch])=>Number(epoch)));
    const targets=Array.isArray(run.targetLadder)&&run.targetLadder.length
      ?run.targetLadder.map(Number)
      :[epochForAttempt(run)];
    targets.filter(epoch=>Number.isFinite(epoch)&&!represented.has(epoch)&&(
      !['failed','canceled','cancelled'].includes(String(run.status).toLowerCase())||
      String(run.status).toLowerCase()==='failed'&&run.failureClass
    )).forEach(epoch=>
      attemptRows.push({...run,epoch,sourceRun:run})
    );
  });

  const selected=new Map();
  resultRows.filter(run=>run.status==='complete'&&finite(run.validation)).forEach(run=>{
    const key=`${run.method}|${run.epoch}`;
    const current=selected.get(key);
    if(!current||Number(run.validation)<Number(current.validation))selected.set(key,run);
  });
  const epochs=[...new Set([...resultRows,...attemptRows].map(run=>run.epoch))].sort((a,b)=>a-b);

  const summary=document.querySelector('#ls-coordinate-summary');
  if(summary)summary.innerHTML=epochs.map(epoch=>{
    const cells=columns.map(column=>{
      const run=selected.get(`${column.key}|${epoch}`);
      return run?`<td title="DCLM validation CE ${metric(run.validation)}">(${run.lr}, ${run.wd})</td>`:'<td>—</td>';
    }).join('');
    return `<tr><td><strong>E${epoch}</strong></td>${cells}</tr>`;
  }).join('');

  const coordinateRows=[...resultRows,...attemptRows];
  const grid=document.querySelector('#ls-coordinate-grid');
  if(grid)grid.innerHTML=columns.flatMap(column=>{
    const methodRows=epochs.flatMap(epoch=>{
      const rawCandidates=coordinateRows.filter(run=>run.method===column.key&&run.epoch===epoch);
      const selectionRelevant=rawCandidates.some(run=>
        run.status==='complete'&&finite(run.validation)||
        !['failed','canceled','cancelled'].includes(String(run.status).toLowerCase())||
        String(run.status).toLowerCase()==='failed'&&run.failureClass
      );
      if(!rawCandidates.length||!selectionRelevant)return [];
      const grouped=new Map();
      rawCandidates.forEach(run=>{
        const key=`${run.lr}|${run.wd}`;
        const current=grouped.get(key);
        const runComplete=run.status==='complete'&&finite(run.validation);
        const currentComplete=current?.status==='complete'&&finite(current.validation);
        if(!current||runComplete&&!currentComplete||runComplete&&currentComplete&&Number(run.validation)<Number(current.validation))grouped.set(key,{...run,attempts:(current?.attempts||0)+1});
        else current.attempts=(current.attempts||1)+1;
      });
      const candidates=[...grouped.values()].sort((a,b)=>Number(a.lr)-Number(b.lr)||Number(a.wd)-Number(b.wd));
      const winner=selected.get(`${column.key}|${epoch}`);
      const chips=candidates.map(run=>{
        const chosen=winner&&Number(winner.lr)===Number(run.lr)&&Number(winner.wd)===Number(run.wd);
        const healthStatus=String(run.healthStatus||run.sourceRun?.healthStatus||'').toLowerCase();
        const unhealthy=run.unhealthy===true||run.sourceRun?.unhealthy===true||['red','unhealthy'].includes(healthStatus);
        const stopped=['failed','canceled','cancelled'].includes(run.status);
        const active=['active','running','submitted','queued','scheduled','pending','startup','healthy_startup'].includes(String(run.status).toLowerCase())||
          ['active','running','submitted','queued','scheduled','pending','startup','healthy_startup'].includes(healthStatus);
        const attempts=run.attempts>1?` · ${run.attempts} attempts`:'';
        const reason=run.reason||run.sourceRun?.reason||'';
        return `<span class="tuple ${chosen?'selected':unhealthy?'unhealthy':stopped?'stopped':active?'active':''}"${reason?` title="${escapeAttribute(reason)}"`:''}>(LR ${run.lr}, WD ${run.wd}) · ${run.status}${finite(run.validation)?` · CE ${metric(run.validation)}`:''}${attempts}</span>`;
      }).join('');
      const selection=winner?`LR ${winner.lr}, WD ${winner.wd} · CE ${metric(winner.validation)}`:'pending';
      return [{epoch,chips,selection}];
    });
    return methodRows.map((row,index)=>`<tr class="coordinate-row${index===0?' method-start':''}">${index===0?`<td class="coordinate-method" rowspan="${methodRows.length}">${gridMethodLabel(column)}</td><td class="coordinate-dimension" rowspan="${methodRows.length}">${escapeHtml(gridH(column))}</td><td class="coordinate-dimension" rowspan="${methodRows.length}">${escapeHtml(gridSimBS(column))}</td><td class="coordinate-dimension coordinate-init" rowspan="${methodRows.length}">${escapeHtml(gridInit(column))}</td>`:''}<td class="coordinate-epoch">E${row.epoch}</td><td><div class="tuple-list">${row.chips||'—'}</div></td><td>${row.selection}</td></tr>`);
  }).join('');

  const columnOrder=new Map(columns.map((column,index)=>[column.key,index]));
  const provenance=[...resultRows,...attemptRows].sort((a,b)=>
    (columnOrder.get(a.method)??999)-(columnOrder.get(b.method)??999)||Number(a.epoch)-Number(b.epoch)||Number(a.wd)-Number(b.wd)||String(a.beaker).localeCompare(String(b.beaker))
  );
  const rows=document.querySelector('#ls-rows');
  if(rows)rows.innerHTML=provenance.map(run=>{
    const reason=run.reason||run.sourceRun?.reason||'';
    const wandb=run.wandb||run.activeWandb||run.sourceRun?.wandb||run.sourceRun?.activeWandb;
    const beaker=run.beaker||run.sourceRun?.beaker;
    return `<tr${reason?` title="${escapeAttribute(reason)}"`:''}><td>${labelFor(run)}</td><td>E${run.epoch}</td><td>${run.batchSequences??'—'}</td><td>${run.simulatedBatchSequences??'—'}</td><td>${run.lr}</td><td>${run.wd}</td><td>${run.status}</td><td>${metric(run.train)}</td><td>${metric(run.validation)}</td><td>${metric(run.acc)}</td><td>${metric(run.bpb)}</td><td>${wandb?`<a href="https://wandb.ai/ai2-llm/sewonm-icsl/runs/${wandb}">${wandb}</a>`:'—'}</td><td>${beaker?`<a href="https://beaker.org/ex/${beaker}">experiment</a>`:'—'}</td></tr>`;
  }).join('');
})();
