#!/usr/bin/env bash
# Standing monitor for the random-control experiments (user request 2026-09-22): every 30 min
#   1. keep the drivers and the twin scheduler alive (restart any that died; they are idempotent)
#   2. launch every missing evaluation (ensure_passes.py) and every missing/dead square training (ensure_squares.py)
#   3. collect finished held-out passes, rebuild + publish the report when new results landed
#   4. record every real failure (exit code not 0/143) of the last 3 h with its last error line in failures.log
#   bash scripts/sparse_experts/olmoe3_squares/monitor.sh   (detach it; log: sparse_experts/olmoe3_routing/monitor.log)
set -u; cd "$(git rev-parse --show-toplevel)"; export PATH=/root/.conda/envs/emo/bin:$PATH; export PYTHONPATH=external/OLMo-core/src
S=sparse_experts; R=$S/olmoe3_routing; D=scripts/sparse_experts/olmoe3_squares; FL=$R/failures.log; touch $FL
say() { echo "$(date -u +%m-%d\ %H:%M) $*"; }
alive() { pgrep -f "$1" > /dev/null; }
while true; do
  # 1. processes
  grep -q 'window 3 done' $S/olmoe3_squares_emorand/logs_driver_w3.log 2>/dev/null || alive '^bash scripts/sparse_experts/olmoe3_squares/squares_control_w3\.sh emo$' || { setsid nohup bash $D/squares_control_w3.sh emo >> $S/olmoe3_squares_emorand/logs_driver_w3.log 2>&1 < /dev/null & say "restarted driver emo w3"; }
  for v in std_k8 s128_k4 s128_k8 emo_k8; do sq=$([ $v = std_k8 ] && echo olmoe3_squares_stdrand8 || ([ $v = s128_k4 ] && echo olmoe3_squares_s128rand4 || ([ $v = s128_k8 ] && echo olmoe3_squares_s128rand8 || echo olmoe3_squares_emorand8)))
    grep -q "done" $S/$sq/logs_driver.log 2>/dev/null && grep -q "^.*\] done" $S/$sq/logs_driver.log 2>/dev/null && continue
    alive "^bash scripts/sparse_experts/olmoe3_squares/squares_randk\.sh $v$" || { setsid nohup bash $D/squares_randk.sh $v >> $S/$sq/logs_driver.log 2>&1 < /dev/null & say "restarted driver $v"; }
  done
  alive '^python scripts/sparse_experts/olmoe3_squares/twin_scheduler\.py' || { setsid nohup python $D/twin_scheduler.py --interval 120 >> $R/twin_scheduler/loop.log 2>&1 < /dev/null & say "restarted twin scheduler"; }
  alive '^bash scripts/sparse_experts/olmoe3_squares/init_point_evals\.sh' || grep -q "init-point evals done" $R/init_point_evals.log 2>/dev/null || { setsid nohup bash $D/init_point_evals.sh >> $R/init_point_evals.log 2>&1 < /dev/null & say "restarted init-point evals"; }
  # 2. missing evaluations
  python $D/ensure_passes.py 2>&1 | tail -n 1
  python $D/ensure_squares.py 2>&1 | grep -v '^  would' | tail -n 3
  # 3. collect + publish
  new=0; for d in $R/runs_heldout*/*/none; do [ -f $d/rank0/DONE ] && [ ! -f $d/meta.json ] && { python scripts/sparse_experts/olmoe3_routing/merge_routing.py $d > /dev/null 2>&1; new=$((new+1)); }; done
  if [ $new -gt 0 ] || [ -n "$(find $S/olmoe3_squares*/ppl_validation -name '*.json' -newer $R/monitor.stamp 2>/dev/null | head -1)" ]; then
    python scripts/olmoe3_routing/build_report.py > /dev/null 2>&1 && bash scripts/publish_reports.sh --no-build > /dev/null 2>&1 && say "collected $new passes, report republished"
  fi
  touch $R/monitor.stamp
  # 4. failures
  beaker workspace experiments ai2/flex2 --format=json 2>/dev/null | python -c "
import json,sys,datetime,subprocess,re
now=datetime.datetime.utcnow(); seen=set(open('$FL').read().split())
for e in json.load(sys.stdin):
    c=e.get('created','')[:19]
    try: age=(now-datetime.datetime.strptime(c,'%Y-%m-%dT%H:%M:%S')).total_seconds()
    except Exception: continue
    if age>3*3600: continue
    js=e.get('jobs') or []; st=(js[-1] if js else {}).get('status',{})
    if st.get('exitCode') in (None,0,143) or e['id'] in seen: continue
    log=subprocess.run(['beaker','experiment','logs',e['id']],capture_output=True,text=True).stdout
    err=[l for l in re.sub(r'\x1b\[[0-9;]*m','',log).split('\n') if re.search(r'Error|No space|Killed|Traceback',l) and 'INFO' not in l]
    line=f\"{e['id']} {e.get('name','')[:70]} exit={st.get('exitCode')} :: {(err[-1] if err else 'no error line')[:160]}\"
    open('$FL','a').write(line+'\n'); print('  FAILURE', line)
"
  say "loop done: free $(df -h /root/EMO | awk 'NR==2{print $4}'), $(pgrep -fc 'squares_randk|squares_control_w3|twin_scheduler|init_point_evals') local processes"
  sleep 1800
done
