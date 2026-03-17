"""Generate a self-contained HTML dashboard for exploring router activations.

Usage:
    python generate_router_dashboard.py [--input router_evals/] [--output router_dashboard.html]
"""

import argparse
import json
import os


def load_data(input_dir):
    data = {}
    for model_dir in sorted(os.listdir(input_dir)):
        model_path = os.path.join(input_dir, model_dir)
        if not os.path.isdir(model_path):
            continue
        data[model_dir] = {}
        for f in sorted(os.listdir(model_path)):
            if not f.endswith("-router.jsonl"):
                continue
            task = f.replace("task-", "").replace("-router.jsonl", "")
            with open(os.path.join(model_path, f)) as fh:
                line = json.loads(fh.readline())
                rounded = [
                    [round(v, 6) for v in layer]
                    for layer in line["avg_router_probabilities"]
                ]
                data[model_dir][task] = rounded
    return data


HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Router Activations Dashboard</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #f8f9fa;
    color: #333;
    padding: 20px;
  }
  h1 { font-size: 1.4rem; margin-bottom: 16px; color: #222; }
  .controls {
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
    margin-bottom: 20px;
    align-items: flex-start;
  }
  .control-group {
    background: #fff;
    border: 1px solid #ddd;
    border-radius: 8px;
    padding: 12px;
    min-width: 200px;
  }
  .control-group label {
    display: block;
    font-weight: 600;
    font-size: 0.8rem;
    text-transform: uppercase;
    color: #666;
    margin-bottom: 6px;
  }
  .control-group select, .control-group input {
    width: 100%;
    padding: 6px 8px;
    border: 1px solid #ccc;
    border-radius: 4px;
    font-size: 0.85rem;
    background: #fff;
  }
  select[multiple] { height: 140px; }
  .top-k-group { max-width: 120px; }
  .top-k-group input { width: 80px; }
  #charts {
    display: flex;
    flex-wrap: wrap;
    gap: 20px;
  }
  .chart-container {
    background: #fff;
    border: 1px solid #ddd;
    border-radius: 8px;
    padding: 16px;
    overflow-x: auto;
    max-width: fit-content;
  }
  .chart-title {
    font-size: 0.95rem;
    font-weight: 600;
    margin-bottom: 4px;
    color: #222;
  }
  .chart-subtitle {
    font-size: 0.78rem;
    color: #888;
    margin-bottom: 10px;
    word-wrap: break-word;
    overflow-wrap: break-word;
  }
  canvas { display: block; }
  .legend {
    display: flex;
    align-items: center;
    gap: 6px;
    margin-top: 8px;
    font-size: 0.75rem;
    color: #666;
  }
  .no-data {
    text-align: center;
    padding: 60px;
    color: #999;
    font-size: 0.95rem;
  }
  .hint {
    font-size: 0.75rem;
    color: #999;
    margin-top: 4px;
  }
  .tooltip {
    position: fixed;
    background: #fff;
    border: 1px solid #ccc;
    border-radius: 6px;
    padding: 8px 10px;
    font-size: 0.78rem;
    pointer-events: none;
    display: none;
    box-shadow: 0 2px 8px rgba(0,0,0,0.12);
    z-index: 100;
    line-height: 1.4;
  }
</style>
</head>
<body>

<h1>Router Activations Dashboard</h1>

<div class="controls">
  <div class="control-group">
    <label>Models</label>
    <select id="modelSelect" multiple></select>
    <div class="hint">Ctrl/Cmd+click to multi-select</div>
  </div>
  <div class="control-group">
    <label>Tasks</label>
    <select id="taskSelect" multiple></select>
    <div class="hint">Ctrl/Cmd+click to multi-select</div>
  </div>
  <div class="control-group top-k-group">
    <label>Top-K Experts</label>
    <input type="number" id="topK" value="8" min="1" max="200">
  </div>
  <div class="control-group">
    <label>Task mode</label>
    <select id="taskMode">
      <option value="separate">Separate (one chart per task)</option>
      <option value="combined">Combined (average across tasks)</option>
    </select>
  </div>
  <div class="control-group">
    <label>Top-K scope</label>
    <select id="topKScope">
      <option value="per_layer">Per layer</option>
      <option value="global">Global (across all layers)</option>
    </select>
  </div>
</div>

<div id="charts"><div class="no-data">Select a model and task to view router activations.</div></div>
<div class="tooltip" id="tooltip"></div>

<script>
const DATA = __DATA_PLACEHOLDER__;
const NICKNAMES = __NICKNAMES_PLACEHOLDER__;
function displayName(model) { return NICKNAMES[model] || model.replace(/-hf$/, ''); }

function populateSelects() {
  const modelSel = document.getElementById('modelSelect');
  const taskSel = document.getElementById('taskSelect');
  const models = Object.keys(DATA).sort();
  const taskSet = new Set();
  for (const m of models) {
    for (const t of Object.keys(DATA[m])) taskSet.add(t);
  }
  const tasks = [...taskSet].sort();
  for (const m of models) {
    const opt = document.createElement('option');
    opt.value = m;
    opt.textContent = displayName(m);
    modelSel.appendChild(opt);
  }
  for (const t of tasks) {
    const opt = document.createElement('option');
    opt.value = t;
    opt.textContent = t;
    taskSel.appendChild(opt);
  }
}

function getSelected(sel) {
  return [...sel.selectedOptions].map(o => o.value);
}

function interpolateColor(t) {
  const r = Math.round(245 - t * 200);
  const g = Math.round(248 - t * 160);
  const b = Math.round(255 - t * 40);
  return `rgb(${r},${g},${b})`;
}

function averageProbs(probsList) {
  const n = probsList.length;
  const numLayers = probsList[0].length;
  const numExperts = probsList[0][0].length;
  const avg = [];
  for (let l = 0; l < numLayers; l++) {
    const row = new Array(numExperts).fill(0);
    for (const p of probsList) {
      for (let e = 0; e < numExperts; e++) row[e] += p[l][e];
    }
    avg.push(row.map(v => v / n));
  }
  return avg;
}

function buildChart(container, probs, numExperts, topK, topKScope, titleText, subtitleText) {
  const numLayers = probs.length;

  let topKPerLayer;
  let globalTopK = null;

  if (topKScope === 'global') {
    // Compute average probability per expert across all layers
    const avgPerExpert = new Array(numExperts).fill(0);
    for (let l = 0; l < numLayers; l++) {
      for (let e = 0; e < numExperts; e++) avgPerExpert[e] += probs[l][e];
    }
    for (let e = 0; e < numExperts; e++) avgPerExpert[e] /= numLayers;
    const sorted = avgPerExpert.map((v, i) => ({v, i})).sort((a, b) => b.v - a.v);
    globalTopK = new Set(sorted.slice(0, topK).map(x => x.i));
    // Still compute per-layer for display, but expert columns are determined globally
    topKPerLayer = probs.map(layer => {
      const indexed = layer.map((v, i) => ({v, i}));
      indexed.sort((a, b) => b.v - a.v);
      return indexed.slice(0, topK);
    });
  } else {
    topKPerLayer = probs.map(layer => {
      const indexed = layer.map((v, i) => ({v, i}));
      indexed.sort((a, b) => b.v - a.v);
      return indexed.slice(0, topK);
    });
  }

  const expertSet = new Set();
  if (globalTopK) {
    for (const eid of globalTopK) expertSet.add(eid);
  } else {
    for (const layerTop of topKPerLayer) {
      for (const e of layerTop) expertSet.add(e.i);
    }
  }
  const expertIds = [...expertSet].sort((a, b) => a - b);
  const numCols = expertIds.length;

  const cellW = Math.max(18, Math.min(32, 900 / numCols));
  const cellH = 24;
  const labelW = 60;
  const headerH = 50;
  const canvasW = labelW + numCols * cellW + 10;
  const canvasH = headerH + numLayers * cellH + 10;

  let gMin = Infinity, gMax = -Infinity;
  for (let l = 0; l < numLayers; l++) {
    for (const eid of expertIds) {
      const v = probs[l][eid];
      if (v < gMin) gMin = v;
      if (v > gMax) gMax = v;
    }
  }

  const div = document.createElement('div');
  div.className = 'chart-container';

  const title = document.createElement('div');
  title.className = 'chart-title';
  title.textContent = titleText;
  div.appendChild(title);

  const subtitle = document.createElement('div');
  subtitle.className = 'chart-subtitle';
  const scopeLabel = topKScope === 'global' ? 'global top-' + topK : 'top-' + topK + ' per layer';
  subtitle.textContent = subtitleText + ' | Showing ' + numCols + ' experts (' + scopeLabel + ') out of ' + numExperts + ' total';
  div.appendChild(subtitle);

  const canvas = document.createElement('canvas');
  canvas.width = canvasW;
  canvas.height = canvasH;
  canvas.style.cursor = 'crosshair';

  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#fff';
  ctx.fillRect(0, 0, canvasW, canvasH);

  ctx.save();
  ctx.font = '10px monospace';
  ctx.fillStyle = '#666';
  ctx.textAlign = 'center';
  for (let c = 0; c < numCols; c++) {
    const x = labelW + c * cellW + cellW / 2;
    ctx.save();
    ctx.translate(x, headerH - 4);
    ctx.rotate(-Math.PI / 4);
    ctx.fillText(expertIds[c], 0, 0);
    ctx.restore();
  }
  ctx.restore();

  for (let l = 0; l < numLayers; l++) {
    const y = headerH + l * cellH;
    ctx.font = '11px monospace';
    ctx.fillStyle = '#555';
    ctx.textAlign = 'right';
    ctx.fillText('L' + l, labelW - 6, y + cellH / 2 + 4);

    const topKIds = globalTopK ? globalTopK : new Set(topKPerLayer[l].map(e => e.i));

    for (let c = 0; c < numCols; c++) {
      const eid = expertIds[c];
      const v = probs[l][eid];
      const t = gMax > gMin ? (v - gMin) / (gMax - gMin) : 0;
      const x = labelW + c * cellW;

      ctx.fillStyle = interpolateColor(t);
      ctx.fillRect(x, y, cellW - 1, cellH - 1);

      if (topKIds.has(eid)) {
        ctx.strokeStyle = 'rgba(40, 80, 180, 0.6)';
        ctx.lineWidth = 1.5;
        ctx.strokeRect(x + 0.5, y + 0.5, cellW - 2, cellH - 2);
      }
    }
  }

  canvas.addEventListener('mousemove', function(e) {
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const col = Math.floor((mx - labelW) / cellW);
    const row = Math.floor((my - headerH) / cellH);
    const tip = document.getElementById('tooltip');

    if (col >= 0 && col < numCols && row >= 0 && row < numLayers) {
      const eid = expertIds[col];
      const v = probs[row][eid];
      const layerSorted = probs[row].map((val, i) => ({val, i})).sort((a, b) => b.val - a.val);
      const rank = layerSorted.findIndex(x => x.i === eid) + 1;
      tip.innerHTML = '<b>Layer ' + row + ', Expert ' + eid + '</b><br>Prob: ' + v.toFixed(6) + '<br>Rank: ' + rank + '/' + numExperts;
      tip.style.display = 'block';
      tip.style.left = (e.clientX + 14) + 'px';
      tip.style.top = (e.clientY + 14) + 'px';
    } else {
      tip.style.display = 'none';
    }
  });
  canvas.addEventListener('mouseleave', function() {
    document.getElementById('tooltip').style.display = 'none';
  });

  div.appendChild(canvas);

  const legend = document.createElement('div');
  legend.className = 'legend';
  const lowSpan = document.createElement('span');
  lowSpan.textContent = 'Low';
  legend.appendChild(lowSpan);
  const legCanvas = document.createElement('canvas');
  legCanvas.width = 120;
  legCanvas.height = 12;
  const lctx = legCanvas.getContext('2d');
  for (let i = 0; i < 120; i++) {
    lctx.fillStyle = interpolateColor(i / 119);
    lctx.fillRect(i, 0, 1, 12);
  }
  legend.appendChild(legCanvas);
  const highSpan = document.createElement('span');
  highSpan.textContent = 'High';
  legend.appendChild(highSpan);
  const spacer = document.createElement('span');
  spacer.innerHTML = '&nbsp;&nbsp;';
  legend.appendChild(spacer);
  const box = document.createElement('span');
  box.style.cssText = 'border:1.5px solid rgba(40,80,180,0.6);width:12px;height:12px;display:inline-block';
  legend.appendChild(box);
  const topkLabel = document.createElement('span');
  topkLabel.textContent = 'Top-K in layer';
  legend.appendChild(topkLabel);
  div.appendChild(legend);

  container.appendChild(div);
}

function renderCharts() {
  const models = getSelected(document.getElementById('modelSelect'));
  const tasks = getSelected(document.getElementById('taskSelect'));
  const topK = parseInt(document.getElementById('topK').value) || 8;
  const mode = document.getElementById('taskMode').value;
  const topKScope = document.getElementById('topKScope').value;
  const container = document.getElementById('charts');

  if (!models.length || !tasks.length) {
    container.innerHTML = '<div class="no-data">Select a model and task to view router activations.</div>';
    return;
  }

  container.innerHTML = '';

  for (const model of models) {
    if (!DATA[model]) continue;

    if (mode === 'combined') {
      // Average probabilities across selected tasks for this model
      const available = tasks.filter(t => DATA[model][t]);
      if (!available.length) continue;
      const probsList = available.map(t => DATA[model][t]);
      // All tasks must have same num experts to combine; group by expert count
      const byExperts = {};
      for (let i = 0; i < available.length; i++) {
        const ne = probsList[i][0].length;
        if (!byExperts[ne]) byExperts[ne] = [];
        byExperts[ne].push({task: available[i], probs: probsList[i]});
      }
      for (const [ne, group] of Object.entries(byExperts)) {
        const avg = averageProbs(group.map(g => g.probs));
        const taskNames = group.map(g => g.task).join(', ');
        buildChart(
          container, avg, parseInt(ne), topK, topKScope,
          displayName(model),
          'Tasks (averaged): ' + taskNames
        );
      }
    } else {
      // Separate chart per task
      for (const task of tasks) {
        if (!DATA[model][task]) continue;
        const probs = DATA[model][task];
        buildChart(
          container, probs, probs[0].length, topK, topKScope,
          displayName(model),
          'Task: ' + task
        );
      }
    }
  }

  if (!container.children.length) {
    container.innerHTML = '<div class="no-data">No data for the selected model/task combination.</div>';
  }
}

document.getElementById('modelSelect').addEventListener('change', renderCharts);
document.getElementById('taskSelect').addEventListener('change', renderCharts);
document.getElementById('topK').addEventListener('input', renderCharts);
document.getElementById('taskMode').addEventListener('change', renderCharts);
document.getElementById('topKScope').addEventListener('change', renderCharts);

populateSelects();
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description="Generate router activations dashboard")
    parser.add_argument("--input", default="router_evals", help="Input directory with router eval JSONL files")
    parser.add_argument("--output", default="router_dashboard.html", help="Output HTML file")
    parser.add_argument("--nicknames", default=None, help="JSON file mapping model directory names to display nicknames")
    args = parser.parse_args()

    data = load_data(args.input)
    nicknames = {}
    if args.nicknames:
        with open(args.nicknames) as f:
            nicknames = json.load(f)
    data_json = json.dumps(data, separators=(",", ":"))
    nicknames_json = json.dumps(nicknames, separators=(",", ":"))
    html = HTML_TEMPLATE.replace("__DATA_PLACEHOLDER__", data_json)
    html = html.replace("__NICKNAMES_PLACEHOLDER__", nicknames_json)

    with open(args.output, "w") as f:
        f.write(html)

    n_models = len(data)
    n_tasks = len({t for m in data.values() for t in m})
    print(f"Generated {args.output} ({n_models} models, {n_tasks} tasks, {len(html) // 1024}KB)")


if __name__ == "__main__":
    main()
