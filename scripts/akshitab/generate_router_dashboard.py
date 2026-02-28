#!/usr/bin/env python3
"""Generate an interactive HTML dashboard for router probability analysis.

Reads all router JSONL files from router_evals/ and produces a single
self-contained HTML file (router_dashboard.html) with embedded data and
Plotly.js visualizations.

Usage:
    python scripts/akshitab/generate_router_dashboard.py
"""

import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR / ".." / ".."
ROUTER_EVALS_DIR = PROJECT_ROOT / "router_evals"
OUTPUT_FILE = PROJECT_ROOT / "router_dashboard.html"

# ── Model definitions ────────────────────────────────────────────────────────
MODELS = {
    "Base (128 experts)": {
        "dir": "moe_1b14b_128experts_olmoe-mix_130B_prenorm_noqknorm_1123_step30995-hf",
        "num_experts": 128,
        "stage": "base",
        "order": 0,
    },
    "Math Ext (init)": {
        "dir": "extensions_moe_1b14b_132experts_olmoe-mix_130B_1103_step30995_init_top2_average_noise_10perc-hf",
        "num_experts": 132,
        "stage": "math_init",
        "order": 1,
    },
    "Math Ext (trained)": {
        "dir": "freeze-fix-moe1b14b_132experts_4trained_math_init_top2_average_noise_10B_lr_4e-4_step2385-hf",
        "num_experts": 132,
        "stage": "math_trained",
        "order": 2,
    },
    "Code Ext (init)": {
        "dir": "extensions_moe_1b14b_132experts_olmoe-mix_130B_1103_step30995_init_top2_code_average_noise-hf",
        "num_experts": 132,
        "stage": "code_init",
        "order": 3,
    },
    "Code Ext (trained)": {
        "dir": "ff-moe1b14b_132experts_4trained_code_mix_init_top2_average_noise_10B_lr_4e-4_step2385-hf",
        "num_experts": 132,
        "stage": "code_trained",
        "order": 4,
    },
    "Merged": {
        "dir": None,
        "num_experts": 136,
        "stage": "merged",
        "order": 5,
    },
}

TASK_GROUPS = {
    "General": ["hellaswag_rc_test"],
    "Math": ["gsm8k", "minerva_math_500"],
    "Code": ["mbpp", "codex_humaneval"],
}

TASK_FILE_MAP = {
    "hellaswag_rc_test": "task-hellaswag_rc_test-router.jsonl",
    "gsm8k": "task-gsm8k-router.jsonl",
    "minerva_math_500": "task-minerva_math_500-router.jsonl",
    "mbpp": "task-mbpp-router.jsonl",
    "codex_humaneval": "task-codex_humaneval-router.jsonl",
}

TASK_TO_GROUP = {}
for group, tasks in TASK_GROUPS.items():
    for t in tasks:
        TASK_TO_GROUP[t] = group


def load_all_data():
    """Load all available router probability data."""
    data = {}
    available = {}
    for model_name, model_info in MODELS.items():
        model_dir = model_info["dir"]
        if model_dir is None:
            continue
        data[model_name] = {}
        available[model_name] = []
        for task_name, task_file in TASK_FILE_MAP.items():
            fpath = ROUTER_EVALS_DIR / model_dir / task_file
            if fpath.exists():
                with open(fpath) as f:
                    raw = json.loads(f.readline())
                data[model_name][task_name] = raw["avg_router_probabilities"]
                available[model_name].append(task_name)
                num_layers = len(raw["avg_router_probabilities"])
                num_experts = len(raw["avg_router_probabilities"][0])
                print(f"  Loaded {model_name} / {task_name}: {num_layers} layers x {num_experts} experts")
    return data, available


def build_embedded_json(data, available):
    """Build the JSON structure to embed in the HTML."""
    models_meta = {}
    for model_name, model_info in MODELS.items():
        models_meta[model_name] = {
            "num_experts": model_info["num_experts"],
            "stage": model_info["stage"],
            "order": model_info["order"],
        }

    task_group_map = TASK_TO_GROUP.copy()

    return json.dumps({
        "models": models_meta,
        "task_groups": TASK_GROUPS,
        "task_to_group": task_group_map,
        "available": available,
        "data": data,
    })


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Router Analysis Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
:root {
    --bg: #1a1a2e;
    --surface: #16213e;
    --surface2: #0f3460;
    --accent: #e94560;
    --accent2: #533483;
    --text: #eee;
    --text-dim: #aaa;
    --border: #333;
    --new-expert: #e94560;
    --original-expert: #4ea8de;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.5;
}
.header {
    background: var(--surface);
    padding: 16px 24px;
    border-bottom: 2px solid var(--accent);
}
.header h1 { font-size: 1.4rem; font-weight: 600; }
.header p { font-size: 0.85rem; color: var(--text-dim); margin-top: 4px; }
.tabs {
    display: flex;
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    overflow-x: auto;
    padding: 0 12px;
}
.tab {
    padding: 10px 18px;
    cursor: pointer;
    border-bottom: 3px solid transparent;
    font-size: 0.85rem;
    white-space: nowrap;
    color: var(--text-dim);
    transition: all 0.2s;
}
.tab:hover { color: var(--text); background: rgba(255,255,255,0.05); }
.tab.active { color: var(--accent); border-bottom-color: var(--accent); font-weight: 600; }
.tab-content { display: none; padding: 20px 24px; }
.tab-content.active { display: block; }
.controls {
    display: flex;
    gap: 16px;
    align-items: center;
    flex-wrap: wrap;
    margin-bottom: 16px;
    padding: 12px 16px;
    background: var(--surface);
    border-radius: 8px;
}
.controls label { font-size: 0.85rem; color: var(--text-dim); }
.controls select, .controls input {
    background: var(--surface2);
    color: var(--text);
    border: 1px solid var(--border);
    padding: 6px 12px;
    border-radius: 4px;
    font-size: 0.85rem;
}
.controls select:focus, .controls input:focus { outline: none; border-color: var(--accent); }
.chart-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
    gap: 16px;
}
.chart-box {
    background: var(--surface);
    border-radius: 8px;
    padding: 12px;
    border: 1px solid var(--border);
}
.chart-box h3 {
    font-size: 0.9rem;
    margin-bottom: 8px;
    color: var(--text-dim);
}
.chart-box.pending {
    opacity: 0.4;
    position: relative;
}
.chart-box.pending::after {
    content: 'Data pending';
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    font-size: 1.2rem;
    color: var(--text-dim);
}
table.data-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.82rem;
    background: var(--surface);
    border-radius: 8px;
    overflow: hidden;
}
table.data-table th, table.data-table td {
    padding: 8px 12px;
    text-align: left;
    border-bottom: 1px solid var(--border);
}
table.data-table th { background: var(--surface2); color: var(--text-dim); font-weight: 600; }
table.data-table tr:hover { background: rgba(255,255,255,0.03); }
.badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 10px;
    font-size: 0.75rem;
    font-weight: 600;
}
.badge-new { background: var(--new-expert); color: #fff; }
.badge-orig { background: var(--original-expert); color: #fff; }
.stat-cards {
    display: flex;
    gap: 12px;
    margin-bottom: 16px;
    flex-wrap: wrap;
}
.stat-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px 20px;
    min-width: 160px;
}
.stat-card .label { font-size: 0.75rem; color: var(--text-dim); }
.stat-card .value { font-size: 1.3rem; font-weight: 700; margin-top: 2px; }
.note {
    font-size: 0.8rem;
    color: var(--text-dim);
    padding: 8px 12px;
    background: rgba(233, 69, 96, 0.1);
    border-left: 3px solid var(--accent);
    border-radius: 0 4px 4px 0;
    margin-bottom: 12px;
}
.plotly-chart { width: 100%; }
</style>
</head>
<body>

<div class="header">
    <h1>Router Analysis Dashboard</h1>
    <p>Expert activation probabilities across models and evaluation tasks &mdash; Base (128 experts) vs Extended (132 experts, new: 128-131)</p>
</div>

<div class="tabs" id="tabs">
    <div class="tab active" data-tab="heatmaps">Heatmaps</div>
    <div class="tab" data-tab="top-experts">Top Experts</div>
    <div class="tab" data-tab="expert-tracker">Expert Tracker</div>
    <div class="tab" data-tab="new-expert">New Expert Analysis</div>
    <div class="tab" data-tab="cross-activation">Cross-Activation</div>
    <div class="tab" data-tab="redistribution">Redistribution</div>
    <div class="tab" data-tab="specialization">Specialization</div>
    <div class="tab" data-tab="rank-migration">Rank Migration</div>
    <div class="tab" data-tab="merge-readiness">Merge Readiness</div>
</div>

<!-- Tab 1: Heatmaps -->
<div class="tab-content active" id="tab-heatmaps">
    <div class="controls">
        <label>Task:</label>
        <select id="heatmap-task"></select>
    </div>
    <div class="chart-grid" id="heatmap-grid"></div>
</div>

<!-- Tab 2: Top Experts -->
<div class="tab-content" id="tab-top-experts">
    <div class="controls">
        <label>Task:</label>
        <select id="topexp-task"></select>
        <label>Top K:</label>
        <select id="topexp-k">
            <option value="8">8</option>
            <option value="15" selected>15</option>
            <option value="25">25</option>
        </select>
    </div>
    <div id="topexp-charts"></div>
</div>

<!-- Tab 3: Expert Tracker -->
<div class="tab-content" id="tab-expert-tracker">
    <div class="controls">
        <label>Task:</label>
        <select id="tracker-task"></select>
        <label>Expert IDs (comma-separated):</label>
        <input id="tracker-experts" type="text" value="128,129,130,131" style="width:200px">
        <label>Pipeline:</label>
        <select id="tracker-pipeline">
            <option value="math">Math (base → math init → math trained)</option>
            <option value="code">Code (base → code init → code trained)</option>
        </select>
        <button onclick="renderExpertTracker()" style="background:var(--accent);color:#fff;border:none;padding:6px 16px;border-radius:4px;cursor:pointer;">Update</button>
    </div>
    <div id="tracker-chart"></div>
</div>

<!-- Tab 4: New Expert Analysis -->
<div class="tab-content" id="tab-new-expert">
    <div class="controls">
        <label>Task:</label>
        <select id="newexp-task"></select>
    </div>
    <div class="stat-cards" id="newexp-stats"></div>
    <div id="newexp-chart"></div>
</div>

<!-- Tab 5: Cross-Activation -->
<div class="tab-content" id="tab-cross-activation">
    <div class="note">Shows total probability mass going to new experts (128-131) for each model/task-group combination. Answers: do math-extension experts activate on code tasks?</div>
    <div id="cross-activation-chart"></div>
    <div id="cross-activation-table"></div>
</div>

<!-- Tab 6: Redistribution -->
<div class="tab-content" id="tab-redistribution">
    <div class="controls">
        <label>Task:</label>
        <select id="redist-task"></select>
        <label>Compare:</label>
        <select id="redist-compare">
            <option value="math_init">Base → Math Ext (init)</option>
            <option value="math_trained">Base → Math Ext (trained)</option>
            <option value="code_init">Base → Code Ext (init)</option>
            <option value="code_trained">Base → Code Ext (trained)</option>
        </select>
    </div>
    <div id="redist-chart"></div>
    <div id="redist-table"></div>
</div>

<!-- Tab 7: Specialization -->
<div class="tab-content" id="tab-specialization">
    <div class="controls">
        <label>Model:</label>
        <select id="spec-model"></select>
    </div>
    <div class="note">Specialization index = variance of expert's avg probability across task groups. High = domain-specialist, low = generalist.</div>
    <div id="spec-chart"></div>
    <div id="spec-overlap"></div>
</div>

<!-- Tab 8: Rank Migration -->
<div class="tab-content" id="tab-rank-migration">
    <div class="controls">
        <label>Task:</label>
        <select id="rank-task"></select>
        <label>Pipeline:</label>
        <select id="rank-pipeline">
            <option value="math">Math (base → math init → math trained)</option>
            <option value="code">Code (base → code init → code trained)</option>
        </select>
        <label>Top N experts:</label>
        <select id="rank-topn">
            <option value="10">10</option>
            <option value="15" selected>15</option>
            <option value="25">25</option>
        </select>
    </div>
    <div id="rank-chart"></div>
</div>

<!-- Tab 9: Merge Readiness -->
<div class="tab-content" id="tab-merge-readiness">
    <div class="note">Compares expert usage between Math Ext (trained) and Code Ext (trained). Experts highly active in both = potential merge conflicts.</div>
    <div class="controls">
        <label>Math task:</label>
        <select id="merge-math-task"></select>
        <label>Code task:</label>
        <select id="merge-code-task"></select>
    </div>
    <div id="merge-chart"></div>
    <div id="merge-new-expert-chart"></div>
</div>

<script>
// ── Embedded data ────────────────────────────────────────────────────────
const DATA = __DATA_PLACEHOLDER__;

// ── Helpers ──────────────────────────────────────────────────────────────
const PLOTLY_LAYOUT_DEFAULTS = {
    paper_bgcolor: '#1a1a2e',
    plot_bgcolor: '#16213e',
    font: { color: '#eee', size: 11 },
    margin: { t: 40, b: 50, l: 60, r: 20 },
};

const PLOTLY_CONFIG = { responsive: true, displayModeBar: true, modeBarButtonsToRemove: ['lasso2d', 'select2d'] };

function pct(v) { return (v * 100).toFixed(3) + '%'; }

function getAllTasks() {
    const tasks = [];
    for (const [group, ts] of Object.entries(DATA.task_groups)) {
        for (const t of ts) tasks.push(t);
    }
    return tasks;
}

function getModelOrder() {
    return Object.entries(DATA.models)
        .sort((a, b) => a[1].order - b[1].order)
        .map(([name]) => name);
}

function getModelsWithTask(task) {
    return getModelOrder().filter(m => DATA.available[m] && DATA.available[m].includes(task));
}

function globalProbs(model, task) {
    // Sum across layers → array of length num_experts
    const layers = DATA.data[model][task];
    const nExperts = layers[0].length;
    const sums = new Array(nExperts).fill(0);
    for (const layer of layers) {
        for (let e = 0; e < nExperts; e++) sums[e] += layer[e];
    }
    return sums;
}

function totalMassForExperts(model, task, expertIds) {
    if (!DATA.data[model] || !DATA.data[model][task]) return null;
    const layers = DATA.data[model][task];
    let total = 0;
    for (const layer of layers) {
        for (const eid of expertIds) {
            if (eid < layer.length) total += layer[eid];
        }
    }
    return total;
}

function populateSelect(selectId, options, defaultIdx) {
    const sel = document.getElementById(selectId);
    sel.innerHTML = '';
    options.forEach((opt, i) => {
        const o = document.createElement('option');
        if (typeof opt === 'object') { o.value = opt.value; o.textContent = opt.label; }
        else { o.value = opt; o.textContent = opt; }
        if (i === (defaultIdx || 0)) o.selected = true;
        sel.appendChild(o);
    });
}

function getStageModels(pipeline) {
    if (pipeline === 'math') return ['Base (128 experts)', 'Math Ext (init)', 'Math Ext (trained)'];
    return ['Base (128 experts)', 'Code Ext (init)', 'Code Ext (trained)'];
}

const MODEL_COLORS = {
    'Base (128 experts)': '#4ea8de',
    'Math Ext (init)': '#ffd166',
    'Math Ext (trained)': '#e94560',
    'Code Ext (init)': '#06d6a0',
    'Code Ext (trained)': '#533483',
    'Merged': '#888',
};

// ── Tab switching ────────────────────────────────────────────────────────
document.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', () => {
        document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(tc => tc.classList.remove('active'));
        tab.classList.add('active');
        document.getElementById('tab-' + tab.dataset.tab).classList.add('active');
    });
});

// =====================================================================
// TAB 1: HEATMAPS
// =====================================================================
function renderHeatmaps() {
    const task = document.getElementById('heatmap-task').value;
    const grid = document.getElementById('heatmap-grid');
    grid.innerHTML = '';

    const models = getModelsWithTask(task);
    // Find global min/max for shared color scale
    let globalMin = Infinity, globalMax = -Infinity;
    for (const m of models) {
        for (const layer of DATA.data[m][task]) {
            for (const v of layer) {
                if (v < globalMin) globalMin = v;
                if (v > globalMax) globalMax = v;
            }
        }
    }

    const orderedModels = getModelOrder();
    for (const m of orderedModels) {
        const box = document.createElement('div');
        box.className = 'chart-box';
        if (m === 'Merged') {
            box.classList.add('pending');
            box.innerHTML = `<h3>${m} (${DATA.models[m].num_experts} experts)</h3><div style="height:350px"></div>`;
            grid.appendChild(box);
            continue;
        }
        if (!models.includes(m)) continue;

        box.innerHTML = `<h3>${m} (${DATA.models[m].num_experts} experts)</h3><div id="hm-${m.replace(/[^a-zA-Z0-9]/g,'_')}" class="plotly-chart"></div>`;
        grid.appendChild(box);

        const layers = DATA.data[m][task];
        const nExperts = layers[0].length;
        const xLabels = Array.from({length: nExperts}, (_, i) => i);
        const yLabels = Array.from({length: layers.length}, (_, i) => `L${i}`);
        const hovertext = layers.map((row, li) => row.map((v, ei) => `Expert ${ei}<br>Layer ${li}<br>Prob: ${pct(v)}`));

        const traces = [{
            z: layers,
            x: xLabels,
            y: yLabels,
            type: 'heatmap',
            colorscale: 'Viridis',
            zmin: globalMin,
            zmax: globalMax,
            hovertext: hovertext,
            hoverinfo: 'text',
            colorbar: { title: 'Prob', titleside: 'right', thickness: 12 },
        }];

        const shapes = [];
        if (nExperts > 128) {
            shapes.push({
                type: 'line', x0: 127.5, x1: 127.5, y0: -0.5, y1: layers.length - 0.5,
                line: { color: '#e94560', width: 2, dash: 'dash' },
            });
        }

        const layout = {
            ...PLOTLY_LAYOUT_DEFAULTS,
            xaxis: { title: 'Expert ID', dtick: nExperts > 128 ? 16 : 16 },
            yaxis: { title: 'Layer', autorange: 'reversed' },
            shapes,
            height: 380,
        };

        Plotly.newPlot(`hm-${m.replace(/[^a-zA-Z0-9]/g,'_')}`, traces, layout, PLOTLY_CONFIG);
    }
}

// =====================================================================
// TAB 2: TOP EXPERTS COMPARISON
// =====================================================================
function renderTopExperts() {
    const task = document.getElementById('topexp-task').value;
    const K = parseInt(document.getElementById('topexp-k').value);
    const container = document.getElementById('topexp-charts');
    container.innerHTML = '';

    const models = getModelsWithTask(task);
    const traces = [];

    for (const m of models) {
        const gp = globalProbs(m, task);
        const indexed = gp.map((v, i) => ({v, i})).sort((a, b) => b.v - a.v).slice(0, K);
        const colors = indexed.map(d => d.i >= 128 ? '#e94560' : '#4ea8de');

        traces.push({
            x: indexed.map(d => `E${d.i}`),
            y: indexed.map(d => d.v),
            type: 'bar',
            name: m,
            marker: { color: colors },
            text: indexed.map(d => d.i >= 128 ? 'NEW' : ''),
            textposition: 'outside',
            hovertext: indexed.map(d => `${m}<br>Expert ${d.i}<br>Sum prob: ${pct(d.v)}<br>${d.i >= 128 ? '(NEW)' : '(original)'}`),
            hoverinfo: 'text',
        });
    }

    // One subplot per model
    for (let mi = 0; mi < models.length; mi++) {
        const m = models[mi];
        const gp = globalProbs(m, task);
        const indexed = gp.map((v, i) => ({v, i})).sort((a, b) => b.v - a.v).slice(0, K);
        const colors = indexed.map(d => d.i >= 128 ? '#e94560' : '#4ea8de');

        const div = document.createElement('div');
        div.className = 'chart-box';
        div.innerHTML = `<h3>${m}</h3><div id="topexp-chart-${mi}" class="plotly-chart"></div>`;
        container.appendChild(div);

        Plotly.newPlot(`topexp-chart-${mi}`, [{
            x: indexed.map(d => `E${d.i}`),
            y: indexed.map(d => d.v),
            type: 'bar',
            marker: { color: colors },
            hovertext: indexed.map(d => `Expert ${d.i}<br>Sum prob: ${pct(d.v)}<br>${d.i >= 128 ? '(NEW)' : '(original)'}`),
            hoverinfo: 'text',
        }], {
            ...PLOTLY_LAYOUT_DEFAULTS,
            xaxis: { title: 'Expert (ranked)', tickangle: -45 },
            yaxis: { title: 'Sum probability (across layers)' },
            height: 320,
            showlegend: false,
            annotations: indexed.filter(d => d.i >= 128).map(d => ({
                x: `E${d.i}`, y: d.v, text: 'NEW', showarrow: false, yshift: 10,
                font: { color: '#e94560', size: 10, weight: 'bold' },
            })),
        }, PLOTLY_CONFIG);
    }
}

// =====================================================================
// TAB 3: EXPERT TRACKER
// =====================================================================
function renderExpertTracker() {
    const task = document.getElementById('tracker-task').value;
    const pipeline = document.getElementById('tracker-pipeline').value;
    const expertStr = document.getElementById('tracker-experts').value;
    const expertIds = expertStr.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n));

    const stages = getStageModels(pipeline);
    const stageLabels = stages.map(s => s);

    const traces = [];
    for (const eid of expertIds) {
        const yVals = [];
        const texts = [];
        for (const m of stages) {
            if (!DATA.data[m] || !DATA.data[m][task]) {
                yVals.push(null);
                texts.push('No data');
            } else {
                const layers = DATA.data[m][task];
                let total = 0;
                for (const layer of layers) {
                    if (eid < layer.length) total += layer[eid];
                }
                yVals.push(total);
                texts.push(`Expert ${eid}<br>${m}<br>Total prob: ${pct(total)}`);
            }
        }
        traces.push({
            x: stageLabels,
            y: yVals,
            mode: 'lines+markers',
            name: `Expert ${eid}${eid >= 128 ? ' (NEW)' : ''}`,
            line: { width: eid >= 128 ? 3 : 2, dash: eid >= 128 ? 'solid' : 'dot' },
            marker: { size: 8, color: eid >= 128 ? '#e94560' : undefined },
            hovertext: texts,
            hoverinfo: 'text',
        });
    }

    Plotly.newPlot('tracker-chart', traces, {
        ...PLOTLY_LAYOUT_DEFAULTS,
        xaxis: { title: 'Model Stage' },
        yaxis: { title: 'Total probability mass (sum across layers)' },
        height: 500,
        legend: { orientation: 'h', y: -0.2 },
        title: { text: `Expert Tracking: ${task}`, font: { size: 14 } },
    }, PLOTLY_CONFIG);
}

// =====================================================================
// TAB 4: NEW EXPERT ANALYSIS
// =====================================================================
function renderNewExpertAnalysis() {
    const task = document.getElementById('newexp-task').value;
    const statsDiv = document.getElementById('newexp-stats');
    const chartDiv = document.getElementById('newexp-chart');
    statsDiv.innerHTML = '';
    chartDiv.innerHTML = '';

    const extModels = getModelOrder().filter(m => DATA.models[m].num_experts > 128 && DATA.data[m] && DATA.data[m][task]);
    if (extModels.length === 0) {
        chartDiv.innerHTML = '<p style="color:var(--text-dim)">No extended model data for this task.</p>';
        return;
    }

    // Grouped bar: per layer, each new expert's prob, per model
    const newExperts = [128, 129, 130, 131];
    const traces = [];

    for (const m of extModels) {
        const layers = DATA.data[m][task];
        const nLayers = layers.length;
        const layerLabels = Array.from({length: nLayers}, (_, i) => `L${i}`);

        // Total new expert mass per layer
        const massPerLayer = layers.map(layer => newExperts.reduce((s, eid) => s + (eid < layer.length ? layer[eid] : 0), 0));

        traces.push({
            x: layerLabels,
            y: massPerLayer,
            type: 'bar',
            name: m,
            marker: { color: MODEL_COLORS[m] || '#888' },
            hovertext: massPerLayer.map((v, li) => {
                const details = newExperts.map(eid => `  E${eid}: ${pct(layers[li][eid] || 0)}`).join('<br>');
                return `${m}<br>Layer ${li}<br>New expert total: ${pct(v)}<br>${details}`;
            }),
            hoverinfo: 'text',
        });
    }

    Plotly.newPlot(chartDiv, traces, {
        ...PLOTLY_LAYOUT_DEFAULTS,
        barmode: 'group',
        xaxis: { title: 'Layer' },
        yaxis: { title: 'New expert prob mass (sum of 128-131)' },
        height: 450,
        legend: { orientation: 'h', y: -0.15 },
        title: { text: `New Expert (128-131) Activation by Layer: ${task}`, font: { size: 14 } },
    }, PLOTLY_CONFIG);

    // Summary stats
    for (const m of extModels) {
        const layers = DATA.data[m][task];
        const massPerLayer = layers.map(layer => newExperts.reduce((s, eid) => s + (eid < layer.length ? layer[eid] : 0), 0));
        const avg = massPerLayer.reduce((a, b) => a + b, 0) / massPerLayer.length;
        const peakLayer = massPerLayer.indexOf(Math.max(...massPerLayer));

        // Rank of new experts globally
        const gp = globalProbs(m, task);
        const sorted = gp.map((v, i) => ({v, i})).sort((a, b) => b.v - a.v);
        const ranks = {};
        sorted.forEach((d, r) => { ranks[d.i] = r + 1; });

        const card = document.createElement('div');
        card.className = 'stat-card';
        card.innerHTML = `
            <div class="label">${m}</div>
            <div class="value">${pct(avg)}</div>
            <div class="label">avg mass/layer | peak: L${peakLayer}</div>
            <div class="label" style="margin-top:4px">Ranks: ${newExperts.map(e => `E${e}=#${ranks[e]||'?'}`).join(', ')}</div>
        `;
        statsDiv.appendChild(card);
    }
}

// =====================================================================
// TAB 5: CROSS-ACTIVATION
// =====================================================================
function renderCrossActivation() {
    const groups = Object.keys(DATA.task_groups);
    const models = getModelOrder().filter(m => DATA.models[m].num_experts > 128 && DATA.data[m]);
    const newExperts = [128, 129, 130, 131];

    // Build matrix: model x group → avg mass across tasks in group
    const zValues = [];
    const hoverTexts = [];
    const modelLabels = [];

    for (const m of models) {
        const row = [];
        const hrow = [];
        modelLabels.push(m);
        for (const group of groups) {
            const tasks = DATA.task_groups[group];
            let total = 0, count = 0;
            const details = [];
            for (const t of tasks) {
                const mass = totalMassForExperts(m, t, newExperts);
                if (mass !== null) {
                    total += mass;
                    count++;
                    details.push(`  ${t}: ${pct(mass)}`);
                }
            }
            const avg = count > 0 ? total / count : null;
            row.push(avg);
            hrow.push(avg !== null ? `${m}<br>${group}<br>Avg new expert mass: ${pct(avg)}<br>${details.join('<br>')}` : 'No data');
        }
        zValues.push(row);
        hoverTexts.push(hrow);
    }

    Plotly.newPlot('cross-activation-chart', [{
        z: zValues,
        x: groups,
        y: modelLabels,
        type: 'heatmap',
        colorscale: [[0, '#16213e'], [0.5, '#ffd166'], [1, '#e94560']],
        hovertext: hoverTexts,
        hoverinfo: 'text',
        colorbar: { title: 'Avg mass', titleside: 'right', thickness: 12 },
    }], {
        ...PLOTLY_LAYOUT_DEFAULTS,
        xaxis: { title: 'Task Group' },
        yaxis: { autorange: 'reversed' },
        height: 300,
        title: { text: 'New Expert (128-131) Cross-Activation by Domain', font: { size: 14 } },
    }, PLOTLY_CONFIG);

    // Table view
    let html = '<table class="data-table" style="margin-top:16px"><thead><tr><th>Model</th>';
    for (const g of groups) html += `<th>${g}</th>`;
    html += '</tr></thead><tbody>';
    for (let i = 0; i < models.length; i++) {
        html += `<tr><td>${models[i]}</td>`;
        for (let j = 0; j < groups.length; j++) {
            const v = zValues[i][j];
            const bg = v !== null ? `rgba(233,69,96,${Math.min(v * 30, 0.6)})` : 'transparent';
            html += `<td style="background:${bg}">${v !== null ? pct(v) : '—'}</td>`;
        }
        html += '</tr>';
    }
    html += '</tbody></table>';
    document.getElementById('cross-activation-table').innerHTML = html;
}

// =====================================================================
// TAB 6: REDISTRIBUTION
// =====================================================================
function renderRedistribution() {
    const task = document.getElementById('redist-task').value;
    const compare = document.getElementById('redist-compare').value;
    const chartDiv = document.getElementById('redist-chart');
    const tableDiv = document.getElementById('redist-table');
    chartDiv.innerHTML = '';
    tableDiv.innerHTML = '';

    const baseModel = 'Base (128 experts)';
    const compareMap = {
        'math_init': 'Math Ext (init)',
        'math_trained': 'Math Ext (trained)',
        'code_init': 'Code Ext (init)',
        'code_trained': 'Code Ext (trained)',
    };
    const targetModel = compareMap[compare];

    if (!DATA.data[baseModel] || !DATA.data[baseModel][task] || !DATA.data[targetModel] || !DATA.data[targetModel][task]) {
        chartDiv.innerHTML = '<p style="color:var(--text-dim)">Data not available for this combination.</p>';
        return;
    }

    const baseGP = globalProbs(baseModel, task);
    const targetGP = globalProbs(targetModel, task);

    // Only compare original experts (0-127)
    const deltas = [];
    for (let e = 0; e < 128; e++) {
        deltas.push({ expert: e, base: baseGP[e], target: targetGP[e], delta: targetGP[e] - baseGP[e] });
    }

    const colors = deltas.map(d => d.delta < 0 ? '#e94560' : '#06d6a0');

    Plotly.newPlot(chartDiv, [{
        x: deltas.map(d => d.expert),
        y: deltas.map(d => d.delta),
        type: 'bar',
        marker: { color: colors },
        hovertext: deltas.map(d =>
            `Expert ${d.expert}<br>Base: ${pct(d.base)}<br>${targetModel}: ${pct(d.target)}<br>Δ: ${d.delta >= 0 ? '+' : ''}${pct(d.delta)}`
        ),
        hoverinfo: 'text',
    }], {
        ...PLOTLY_LAYOUT_DEFAULTS,
        xaxis: { title: 'Expert ID (0-127)' },
        yaxis: { title: 'Δ probability (sum across layers)', zeroline: true, zerolinecolor: '#666' },
        height: 450,
        title: { text: `Prob Redistribution: ${baseModel} → ${targetModel} | ${task}`, font: { size: 13 } },
        showlegend: false,
    }, PLOTLY_CONFIG);

    // Table of biggest losers
    const losers = [...deltas].sort((a, b) => a.delta - b.delta).slice(0, 15);
    const gainers = [...deltas].sort((a, b) => b.delta - a.delta).slice(0, 15);

    let html = '<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:16px">';
    html += '<div><h3 style="color:var(--accent);margin-bottom:8px">Biggest Losers (prob decreased)</h3><table class="data-table"><thead><tr><th>Expert</th><th>Base</th><th>After</th><th>Δ</th></tr></thead><tbody>';
    for (const d of losers) {
        html += `<tr><td>E${d.expert}</td><td>${pct(d.base)}</td><td>${pct(d.target)}</td><td style="color:#e94560">${pct(d.delta)}</td></tr>`;
    }
    html += '</tbody></table></div>';

    html += '<div><h3 style="color:#06d6a0;margin-bottom:8px">Biggest Gainers (prob increased)</h3><table class="data-table"><thead><tr><th>Expert</th><th>Base</th><th>After</th><th>Δ</th></tr></thead><tbody>';
    for (const d of gainers) {
        html += `<tr><td>E${d.expert}</td><td>${pct(d.base)}</td><td>${pct(d.target)}</td><td style="color:#06d6a0">+${pct(d.delta)}</td></tr>`;
    }
    html += '</tbody></table></div></div>';
    tableDiv.innerHTML = html;
}

// =====================================================================
// TAB 7: SPECIALIZATION
// =====================================================================
function renderSpecialization() {
    const model = document.getElementById('spec-model').value;
    const chartDiv = document.getElementById('spec-chart');
    const overlapDiv = document.getElementById('spec-overlap');
    chartDiv.innerHTML = '';
    overlapDiv.innerHTML = '';

    if (!DATA.data[model]) {
        chartDiv.innerHTML = '<p style="color:var(--text-dim)">No data for this model.</p>';
        return;
    }

    const groups = Object.keys(DATA.task_groups);
    const nExperts = DATA.models[model].num_experts;

    // Per expert: avg prob per task group (averaged across tasks in group)
    const expertGroupProbs = []; // [expertId][groupIdx]
    for (let e = 0; e < nExperts; e++) {
        const groupProbs = [];
        for (const group of groups) {
            const tasks = DATA.task_groups[group];
            let total = 0, count = 0;
            for (const t of tasks) {
                if (DATA.data[model][t]) {
                    let sum = 0;
                    for (const layer of DATA.data[model][t]) {
                        if (e < layer.length) sum += layer[e];
                    }
                    total += sum;
                    count++;
                }
            }
            groupProbs.push(count > 0 ? total / count : 0);
        }
        expertGroupProbs.push(groupProbs);
    }

    // Specialization index = variance across groups
    const specScores = expertGroupProbs.map(gp => {
        const mean = gp.reduce((a, b) => a + b, 0) / gp.length;
        return gp.reduce((s, v) => s + (v - mean) ** 2, 0) / gp.length;
    });

    // Dominant domain per expert
    const domainColors = { 'General': '#4ea8de', 'Math': '#e94560', 'Code': '#06d6a0' };
    const dominantDomain = expertGroupProbs.map(gp => {
        const maxIdx = gp.indexOf(Math.max(...gp));
        return groups[maxIdx];
    });

    Plotly.newPlot(chartDiv, [{
        x: Array.from({length: nExperts}, (_, i) => i),
        y: specScores,
        mode: 'markers',
        marker: {
            size: 6,
            color: dominantDomain.map(d => domainColors[d]),
            opacity: 0.7,
        },
        hovertext: specScores.map((s, i) => {
            const details = groups.map((g, gi) => `  ${g}: ${pct(expertGroupProbs[i][gi])}`).join('<br>');
            return `Expert ${i}${i >= 128 ? ' (NEW)' : ''}<br>Specialization: ${s.toExponential(2)}<br>Dominant: ${dominantDomain[i]}<br>${details}`;
        }),
        hoverinfo: 'text',
    }], {
        ...PLOTLY_LAYOUT_DEFAULTS,
        xaxis: { title: 'Expert ID' },
        yaxis: { title: 'Specialization index (variance)' },
        height: 420,
        title: { text: `Expert Specialization: ${model}`, font: { size: 14 } },
        showlegend: false,
        shapes: nExperts > 128 ? [{
            type: 'line', x0: 127.5, x1: 127.5, y0: 0, y1: Math.max(...specScores) * 1.1,
            line: { color: '#e94560', width: 1, dash: 'dash' },
        }] : [],
        annotations: [{
            x: 0.01, y: 0.99, xref: 'paper', yref: 'paper', showarrow: false,
            text: `<span style="color:${domainColors.General}">● General</span>  <span style="color:${domainColors.Math}">● Math</span>  <span style="color:${domainColors.Code}">● Code</span>`,
            font: { size: 11 },
        }],
    }, PLOTLY_CONFIG);

    // Domain overlap: cosine similarity of expert activation vectors across task groups
    const groupVectors = {};
    for (const group of groups) {
        const vec = new Array(nExperts).fill(0);
        const tasks = DATA.task_groups[group];
        let count = 0;
        for (const t of tasks) {
            if (DATA.data[model][t]) {
                const gp = globalProbs(model, t);
                for (let e = 0; e < gp.length; e++) vec[e] += gp[e];
                count++;
            }
        }
        if (count > 0) for (let e = 0; e < nExperts; e++) vec[e] /= count;
        groupVectors[group] = vec;
    }

    function cosineSim(a, b) {
        let dot = 0, na = 0, nb = 0;
        for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; na += a[i] ** 2; nb += b[i] ** 2; }
        return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-12);
    }

    const simMatrix = groups.map(g1 => groups.map(g2 => cosineSim(groupVectors[g1], groupVectors[g2])));

    Plotly.newPlot(overlapDiv, [{
        z: simMatrix,
        x: groups,
        y: groups,
        type: 'heatmap',
        colorscale: 'Blues',
        zmin: 0.9,
        zmax: 1,
        text: simMatrix.map(row => row.map(v => v.toFixed(4))),
        texttemplate: '%{text}',
        hoverinfo: 'text',
        hovertext: simMatrix.map((row, i) => row.map((v, j) => `${groups[i]} vs ${groups[j]}<br>Cosine similarity: ${v.toFixed(4)}`)),
    }], {
        ...PLOTLY_LAYOUT_DEFAULTS,
        title: { text: `Domain Overlap (cosine similarity of expert activation): ${model}`, font: { size: 13 } },
        height: 300,
        xaxis: { title: '' },
        yaxis: { autorange: 'reversed' },
    }, PLOTLY_CONFIG);
}

// =====================================================================
// TAB 8: RANK MIGRATION (Bump Chart)
// =====================================================================
function renderRankMigration() {
    const task = document.getElementById('rank-task').value;
    const pipeline = document.getElementById('rank-pipeline').value;
    const topN = parseInt(document.getElementById('rank-topn').value);
    const chartDiv = document.getElementById('rank-chart');
    chartDiv.innerHTML = '';

    const stages = getStageModels(pipeline);
    const availableStages = stages.filter(m => DATA.data[m] && DATA.data[m][task]);

    if (availableStages.length < 2) {
        chartDiv.innerHTML = '<p style="color:var(--text-dim)">Need at least 2 stages with data for this task.</p>';
        return;
    }

    // Get rankings per stage
    const rankings = {};
    for (const m of availableStages) {
        const gp = globalProbs(m, task);
        const sorted = gp.map((v, i) => ({v, i})).sort((a, b) => b.v - a.v);
        rankings[m] = {};
        sorted.forEach((d, r) => { rankings[m][d.i] = r + 1; });
    }

    // Collect union of top-N experts from all stages
    const trackedExperts = new Set();
    for (const m of availableStages) {
        const gp = globalProbs(m, task);
        const sorted = gp.map((v, i) => ({v, i})).sort((a, b) => b.v - a.v);
        for (let r = 0; r < topN && r < sorted.length; r++) {
            trackedExperts.add(sorted[r].i);
        }
    }

    const traces = [];
    for (const eid of [...trackedExperts].sort((a, b) => a - b)) {
        const yVals = availableStages.map(m => rankings[m][eid] || null);
        const isNew = eid >= 128;
        traces.push({
            x: availableStages,
            y: yVals,
            mode: 'lines+markers',
            name: `E${eid}${isNew ? '*' : ''}`,
            line: {
                width: isNew ? 3 : 1.5,
                color: isNew ? '#e94560' : undefined,
                dash: isNew ? 'solid' : 'dot',
            },
            marker: { size: isNew ? 8 : 5 },
            opacity: isNew ? 1 : 0.6,
            hovertext: availableStages.map(m => `Expert ${eid}${isNew ? ' (NEW)' : ''}<br>${m}<br>Rank: #${rankings[m][eid] || '?'}`),
            hoverinfo: 'text',
        });
    }

    Plotly.newPlot(chartDiv, traces, {
        ...PLOTLY_LAYOUT_DEFAULTS,
        xaxis: { title: 'Model Stage' },
        yaxis: { title: 'Rank (1 = highest prob)', autorange: 'reversed', range: [0.5, topN + 5] },
        height: 550,
        legend: { orientation: 'v', x: 1.02, y: 1, font: { size: 9 } },
        title: { text: `Expert Rank Migration: ${task} (${pipeline} pipeline)`, font: { size: 14 } },
    }, PLOTLY_CONFIG);
}

// =====================================================================
// TAB 9: MERGE READINESS
// =====================================================================
function renderMergeReadiness() {
    const mathTask = document.getElementById('merge-math-task').value;
    const codeTask = document.getElementById('merge-code-task').value;
    const chartDiv = document.getElementById('merge-chart');
    const newDiv = document.getElementById('merge-new-expert-chart');
    chartDiv.innerHTML = '';
    newDiv.innerHTML = '';

    const mathModel = 'Math Ext (trained)';
    const codeModel = 'Code Ext (trained)';

    if (!DATA.data[mathModel] || !DATA.data[mathModel][mathTask] || !DATA.data[codeModel] || !DATA.data[codeModel][codeTask]) {
        chartDiv.innerHTML = '<p style="color:var(--text-dim)">Need both Math Ext (trained) and Code Ext (trained) data.</p>';
        return;
    }

    const mathGP = globalProbs(mathModel, mathTask);
    const codeGP = globalProbs(codeModel, codeTask);

    // Scatter: original experts (0-127) math prob vs code prob
    const origExperts = Array.from({length: 128}, (_, i) => i);
    const mathProbs = origExperts.map(e => mathGP[e]);
    const codeProbs = origExperts.map(e => codeGP[e]);

    // Identify "conflict" experts — high in both
    const mathThreshold = mathProbs.reduce((a, b) => a + b, 0) / 128 * 2;
    const codeThreshold = codeProbs.reduce((a, b) => a + b, 0) / 128 * 2;
    const colors = origExperts.map(e =>
        mathProbs[e] > mathThreshold && codeProbs[e] > codeThreshold ? '#e94560' :
        mathProbs[e] > mathThreshold ? '#ffd166' :
        codeProbs[e] > codeThreshold ? '#06d6a0' : '#4ea8de'
    );

    Plotly.newPlot(chartDiv, [{
        x: mathProbs,
        y: codeProbs,
        mode: 'markers',
        marker: { size: 5, color: colors, opacity: 0.7 },
        text: origExperts.map(e => `E${e}`),
        hovertext: origExperts.map(e =>
            `Expert ${e}<br>Math prob: ${pct(mathProbs[e])}<br>Code prob: ${pct(codeProbs[e])}<br>${
                mathProbs[e] > mathThreshold && codeProbs[e] > codeThreshold ? 'POTENTIAL CONFLICT' : ''}`
        ),
        hoverinfo: 'text',
    }], {
        ...PLOTLY_LAYOUT_DEFAULTS,
        xaxis: { title: `Prob in Math Ext (trained) on ${mathTask}` },
        yaxis: { title: `Prob in Code Ext (trained) on ${codeTask}` },
        height: 500,
        title: { text: 'Merge Conflict Analysis: Original Expert Usage', font: { size: 14 } },
        shapes: [
            { type: 'line', x0: mathThreshold, x1: mathThreshold, y0: 0, y1: Math.max(...codeProbs) * 1.1,
              line: { color: '#ffd166', width: 1, dash: 'dash' } },
            { type: 'line', x0: 0, x1: Math.max(...mathProbs) * 1.1, y0: codeThreshold, y1: codeThreshold,
              line: { color: '#06d6a0', width: 1, dash: 'dash' } },
        ],
        annotations: [
            { x: 0.99, y: 0.99, xref: 'paper', yref: 'paper', showarrow: false,
              text: '<span style="color:#e94560">● Conflict</span>  <span style="color:#ffd166">● Math-heavy</span>  <span style="color:#06d6a0">● Code-heavy</span>  <span style="color:#4ea8de">● Low</span>',
              font: { size: 10 } },
        ],
    }, PLOTLY_CONFIG);

    // New expert analysis: do math's new experts activate on code tasks and vice versa?
    const newExperts = [128, 129, 130, 131];
    const mathOnMath = newExperts.map(e => totalMassForExperts(mathModel, mathTask, [e]));
    const mathOnCode = newExperts.map(e => {
        if (DATA.data[mathModel] && DATA.data[mathModel][codeTask]) return totalMassForExperts(mathModel, codeTask, [e]);
        return null;
    });
    const codeOnCode = newExperts.map(e => totalMassForExperts(codeModel, codeTask, [e]));
    const codeOnMath = newExperts.map(e => {
        if (DATA.data[codeModel] && DATA.data[codeModel][mathTask]) return totalMassForExperts(codeModel, mathTask, [e]);
        return null;
    });

    const barTraces = [];
    const xLabels = newExperts.map(e => `E${e}`);

    barTraces.push({
        x: xLabels, y: mathOnMath, type: 'bar', name: `Math new exp → ${mathTask}`,
        marker: { color: '#e94560' },
    });
    if (mathOnCode.some(v => v !== null)) {
        barTraces.push({
            x: xLabels, y: mathOnCode, type: 'bar', name: `Math new exp → ${codeTask}`,
            marker: { color: '#ffd166' },
        });
    }
    barTraces.push({
        x: xLabels, y: codeOnCode, type: 'bar', name: `Code new exp → ${codeTask}`,
        marker: { color: '#06d6a0' },
    });
    if (codeOnMath.some(v => v !== null)) {
        barTraces.push({
            x: xLabels, y: codeOnMath, type: 'bar', name: `Code new exp → ${mathTask}`,
            marker: { color: '#533483' },
        });
    }

    Plotly.newPlot(newDiv, barTraces, {
        ...PLOTLY_LAYOUT_DEFAULTS,
        barmode: 'group',
        xaxis: { title: 'New Expert ID' },
        yaxis: { title: 'Total probability mass' },
        height: 350,
        title: { text: 'New Expert Cross-Domain Activation', font: { size: 13 } },
        legend: { orientation: 'h', y: -0.2 },
    }, PLOTLY_CONFIG);
}

// ── Initialization ───────────────────────────────────────────────────────
function init() {
    const allTasks = getAllTasks();

    // Populate all task dropdowns
    const taskSelects = ['heatmap-task', 'topexp-task', 'tracker-task', 'newexp-task', 'redist-task', 'rank-task'];
    for (const id of taskSelects) {
        populateSelect(id, allTasks);
    }

    // Populate model selects
    const modelNames = getModelOrder().filter(m => DATA.data[m]);
    populateSelect('spec-model', modelNames);

    // Merge readiness: math/code task selects
    const mathTasks = DATA.task_groups['Math'] || [];
    const codeTasks = DATA.task_groups['Code'] || [];
    populateSelect('merge-math-task', mathTasks);
    populateSelect('merge-code-task', codeTasks);

    // Wire up event listeners
    document.getElementById('heatmap-task').addEventListener('change', renderHeatmaps);
    document.getElementById('topexp-task').addEventListener('change', renderTopExperts);
    document.getElementById('topexp-k').addEventListener('change', renderTopExperts);
    document.getElementById('newexp-task').addEventListener('change', renderNewExpertAnalysis);
    document.getElementById('redist-task').addEventListener('change', renderRedistribution);
    document.getElementById('redist-compare').addEventListener('change', renderRedistribution);
    document.getElementById('spec-model').addEventListener('change', renderSpecialization);
    document.getElementById('rank-task').addEventListener('change', renderRankMigration);
    document.getElementById('rank-pipeline').addEventListener('change', renderRankMigration);
    document.getElementById('rank-topn').addEventListener('change', renderRankMigration);
    document.getElementById('merge-math-task').addEventListener('change', renderMergeReadiness);
    document.getElementById('merge-code-task').addEventListener('change', renderMergeReadiness);

    // Tab change triggers re-render
    document.querySelectorAll('.tab').forEach(tab => {
        tab.addEventListener('click', () => {
            setTimeout(() => {
                const tabId = tab.dataset.tab;
                const renderers = {
                    'heatmaps': renderHeatmaps,
                    'top-experts': renderTopExperts,
                    'expert-tracker': renderExpertTracker,
                    'new-expert': renderNewExpertAnalysis,
                    'cross-activation': renderCrossActivation,
                    'redistribution': renderRedistribution,
                    'specialization': renderSpecialization,
                    'rank-migration': renderRankMigration,
                    'merge-readiness': renderMergeReadiness,
                };
                if (renderers[tabId]) renderers[tabId]();
            }, 50);
        });
    });

    // Initial render
    renderHeatmaps();
}

document.addEventListener('DOMContentLoaded', init);
</script>
</body>
</html>
"""


def main():
    print("Loading router probability data...")
    data, available = load_all_data()

    print(f"\nLoaded data for {len(data)} models")
    for model, tasks in available.items():
        print(f"  {model}: {len(tasks)} tasks")

    # Build and embed JSON
    embedded_json = build_embedded_json(data, available)
    html = HTML_TEMPLATE.replace("__DATA_PLACEHOLDER__", embedded_json)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        f.write(html)

    print(f"\nDashboard written to: {OUTPUT_FILE}")
    print(f"  File size: {os.path.getsize(OUTPUT_FILE) / 1024:.1f} KB")
    print("Open in a browser to view.")


if __name__ == "__main__":
    main()
