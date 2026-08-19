(() => {
  const body=document.querySelector('#dr-wt-embwd-chains');
  if(!body)return;
  const report=window.ICSL_REPORT_DATA||{};
  const chains=report.adaptiveDrWtEmbedWdChains||[];
  const escape=value=>String(value??'').replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
  const active=status=>['submitted','scheduled','running','pending','queued'].includes(status);
  const frontierText=chain=>Object.entries(chain.frontiers||{})
    .sort((a,b)=>Number(a[0])-Number(b[0]))
    .map(([epoch,frontier])=>`E${epoch}: WD${frontier.selectedWd} · ${Number(frontier.selectedValidationExact).toFixed(4)}`)
    .join(' → ')||'—';
  const currentText=chain=>{
    if(chain.saturatedEpoch)return `saturated at E${chain.saturatedEpoch}`;
    if(!chain.activeEpoch)return '—';
    const wds=(chain.activeWds||[]).map(wd=>`WD${wd}`).join(', ');
    const progress=chain.progress?` · ${chain.progress.percent}%` : '';
    return `E${chain.activeEpoch}${wds?` · ${wds}`:''}${progress}`;
  };
  body.innerHTML=chains
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
