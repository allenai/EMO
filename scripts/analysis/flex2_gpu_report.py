#!/usr/bin/env python3
"""Build a weekly, per-user GPU occupancy report for ai2/flex2.

The report uses Beaker's experiment history.  "Utilization" here means allocated
GPU occupancy (requested GPUs multiplied by running wall-clock time), not device
SM/Tensor Core telemetry.
"""

import argparse
import base64
import html
import json
import math
import re
import struct
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import quote
from zoneinfo import ZoneInfo


WORKSPACE = "ai2/flex2"
WORKSPACE_API_REF = "ai2%252Fflex2"
PAGE_SIZE = 100
COLORS = [
    "#2563eb", "#7c3aed", "#0891b2", "#059669", "#d97706", "#dc2626",
    "#db2777", "#4f46e5", "#0f766e", "#65a30d", "#ea580c", "#9333ea",
]


def parse_time(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    # Python 3.9's ISO parser rejects some variable-width fractional seconds
    # emitted by Beaker, so normalize them to exactly six digits.
    match = re.fullmatch(r"(.{19})(?:\.(\d+))?(Z|[+-]\d\d:\d\d)?", value)
    if not match:
        raise ValueError(f"Unsupported Beaker timestamp: {value!r}")
    base, fraction, offset = match.groups()
    normalized = f"{base}.{((fraction or '') + '000000')[:6]}"
    parsed = datetime.strptime(normalized, "%Y-%m-%dT%H:%M:%S.%f")
    if not offset or offset == "Z":
        return parsed.replace(tzinfo=timezone.utc)
    sign = 1 if offset[0] == "+" else -1
    hours, minutes = map(int, offset[1:].split(":"))
    return parsed.replace(tzinfo=timezone(sign * timedelta(hours=hours, minutes=minutes)))


def cursor_for_offset(offset: int) -> str:
    # Cursor travels in the query string; Beaker accepts URL-safe base64.
    return base64.urlsafe_b64encode(struct.pack("<Q", offset)).decode("ascii")


def fetch_page(offset: int) -> Tuple[int, Dict[str, Any]]:
    cursor = "" if offset == 0 else cursor_for_offset(offset)
    endpoint = f"workspaces/{WORKSPACE_API_REF}/experiments?cursor={quote(cursor, safe='')}&q="
    proc = subprocess.run(
        ["beaker", "api", endpoint, "--format", "json"],
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(proc.stdout)
    if not isinstance(payload, dict) or "data" not in payload:
        raise RuntimeError(f"Unexpected Beaker response at offset {offset}: {payload!r}")
    return offset, payload


def iter_recent_experiments(scan_start: datetime, workers: int) -> Iterable[Dict[str, Any]]:
    """Fetch newest-first pages until a whole batch predates scan_start."""
    offset = 0
    while True:
        offsets = [offset + PAGE_SIZE * i for i in range(workers)]
        pages: Dict[int, Dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(fetch_page, item) for item in offsets]
            for future in as_completed(futures):
                page_offset, payload = future.result()
                pages[page_offset] = payload

        batch: List[Dict[str, Any]] = []
        for page_offset in offsets:
            batch.extend(pages[page_offset].get("data", []))
        if not batch:
            return

        for experiment in batch:
            yield experiment

        created = [parse_time(item.get("created")) for item in batch]
        valid_created = [item for item in created if item is not None]
        if len(batch) < PAGE_SIZE * workers or (
            valid_created and max(valid_created) < scan_start
        ):
            return
        offset += PAGE_SIZE * workers


def job_interval(job: Dict[str, Any], report_end: datetime) -> Optional[Tuple[datetime, datetime]]:
    status = job.get("status") or {}
    start = parse_time(status.get("started"))
    if start is None:
        return None
    end = (
        parse_time(status.get("exited"))
        or parse_time(status.get("canceled"))
        or parse_time(status.get("finalized"))
        or report_end
    )
    end = min(end, report_end)
    return (start, end) if end > start else None


def gpu_count(job: Dict[str, Any]) -> int:
    requested = (job.get("requests") or {}).get("gpuCount")
    if requested is not None:
        return int(requested)
    return len((job.get("limits") or {}).get("gpus") or [])


def sparkline(values: List[float], width: int = 126, height: int = 28) -> str:
    maximum = max(values, default=0.0)
    if maximum <= 0:
        points = f"0,{height - 2} {width},{height - 2}"
    else:
        step = width / max(1, len(values) - 1)
        points = " ".join(
            f"{i * step:.1f},{height - 2 - (value / maximum) * (height - 6):.1f}"
            for i, value in enumerate(values)
        )
    return (
        f'<svg class="spark" viewBox="0 0 {width} {height}" aria-hidden="true">'
        f'<polyline points="{points}" fill="none" stroke="currentColor" '
        'stroke-width="2" vector-effect="non-scaling-stroke"/></svg>'
    )


def stacked_chart(weeks: List[Dict[str, Any]], users: List[str], colors: Dict[str, str]) -> str:
    width, height = 1120, 360
    left, right, top, bottom = 54, 18, 20, 54
    plot_w, plot_h = width - left - right, height - top - bottom
    bar_slot = plot_w / len(weeks)
    bar_w = bar_slot * 0.72
    chunks = [f'<svg viewBox="0 0 {width} {height}" class="chart" role="img" aria-label="Weekly GPU utilization stacked by user">']
    for pct in (0, 25, 50, 75, 100):
        y = top + plot_h * (1 - pct / 100)
        chunks.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" class="grid"/>')
        chunks.append(f'<text x="{left-9}" y="{y+4:.1f}" text-anchor="end" class="axis">{pct}%</text>')
    for i, week in enumerate(weeks):
        x = left + i * bar_slot + (bar_slot - bar_w) / 2
        y_bottom = top + plot_h
        for user in users:
            pct = 100 * week["by_user"].get(user, 0.0) / week["capacity_gpu_hours"]
            if pct <= 0:
                continue
            h = plot_h * pct / 100
            y_bottom -= h
            title = html.escape(f'{week["label"]} — {user}: {pct:.1f}%')
            chunks.append(
                f'<rect x="{x:.1f}" y="{y_bottom:.1f}" width="{bar_w:.1f}" height="{h:.1f}" '
                f'fill="{colors[user]}"><title>{title}</title></rect>'
            )
        label = html.escape(week["short_label"])
        chunks.append(f'<text x="{x+bar_w/2:.1f}" y="{height-29}" text-anchor="middle" class="axis week-label">{label}</text>')
    chunks.append('</svg>')
    return "".join(chunks)


def render_html(report: Dict[str, Any]) -> str:
    weeks = report["weeks"]
    users = report["users"]
    top_users = [item["user"] for item in users[:11]]
    chart_users = top_users + (["Other"] if len(users) > len(top_users) else [])
    colors = {user: COLORS[i % len(COLORS)] for i, user in enumerate(chart_users)}

    chart_weeks = []
    for week in weeks:
        by_user = {user: week["by_user"].get(user, 0.0) for user in top_users}
        if "Other" in chart_users:
            by_user["Other"] = sum(
                value for user, value in week["by_user"].items() if user not in top_users
            )
        chart_weeks.append({**week, "by_user": by_user})

    avg = report["summary"]["average_utilization_pct"]
    peak = report["summary"]["peak_week"]
    total = report["summary"]["used_gpu_hours"]
    legend = "".join(
        f'<span><i style="background:{colors[user]}"></i>{html.escape(user)}</span>'
        for user in chart_users
    )
    weekly_rows = "".join(
        "<tr>"
        f'<td><strong>{html.escape(week["label"])}</strong></td>'
        f'<td>{week["used_gpu_hours"]:,.0f}</td>'
        f'<td>{week["average_gpus"]:.1f}</td>'
        f'<td><div class="meter"><b style="width:{min(100, week["utilization_pct"]):.2f}%"></b></div></td>'
        f'<td class="pct">{week["utilization_pct"]:.1f}%</td>'
        "</tr>"
        for week in weeks
    )
    user_rows = "".join(
        "<tr>"
        f'<td><span class="rank">{index}</span><strong>{html.escape(user["user"])}</strong></td>'
        f'<td>{user["gpu_hours"]:,.0f}</td>'
        f'<td>{user["share_pct"]:.1f}%</td>'
        f'<td>{user["average_gpus"]:.1f}</td>'
        f'<td>{user["peak_week_pct"]:.1f}%</td>'
        f'<td>{sparkline(user["weekly_gpu_hours"])}</td>'
        "</tr>"
        for index, user in enumerate(users, 1)
    )
    generated = datetime.fromisoformat(report["generated_at"]).astimezone(ZoneInfo(report["timezone"]))

    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Flex2 GPU Utilization Report</title><meta name="robots" content="noindex,nofollow,noarchive">
<style>
:root{{--ink:#172033;--muted:#64748b;--line:#e2e8f0;--paper:#fff;--wash:#f4f7fb;--blue:#2563eb;--navy:#15274a}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--wash);color:var(--ink);font:14px/1.5 Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
.wrap{{max-width:1200px;margin:auto;padding:34px 26px 56px}} header{{background:linear-gradient(125deg,#102246,#1e4b88);color:white;border-radius:18px;padding:34px 38px;box-shadow:0 14px 36px #18376624}}
.eyebrow{{margin:0 0 5px;text-transform:uppercase;letter-spacing:.13em;font-size:12px;font-weight:750;color:#b9d6ff}} h1{{margin:0;font-size:34px;letter-spacing:-.035em}} header p{{margin:8px 0 0;color:#d7e6fb;font-size:15px}} .cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:18px 0}}
.card,.panel{{background:var(--paper);border:1px solid var(--line);border-radius:14px;box-shadow:0 4px 18px #1f35500a}} .card{{padding:19px 20px}} .card small{{display:block;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);font-weight:700;font-size:11px}} .card strong{{display:block;font-size:27px;letter-spacing:-.03em;margin-top:4px}} .card em{{font-style:normal;color:var(--muted)}}
.panel{{padding:24px;margin-top:16px}} h2{{font-size:19px;margin:0 0 3px;letter-spacing:-.015em}} .sub{{color:var(--muted);margin:0 0 18px}} .chart{{width:100%;height:auto;overflow:visible}} .grid{{stroke:#e7edf5;stroke-width:1}} .axis{{fill:#718096;font-size:11px}} .week-label{{font-size:10px}} .legend{{display:flex;flex-wrap:wrap;gap:8px 16px;margin-top:7px;color:#475569;font-size:12px}} .legend span{{white-space:nowrap}} .legend i{{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:5px}}
.cols{{display:grid;grid-template-columns:1fr 1.15fr;gap:16px}} table{{border-collapse:collapse;width:100%}} th{{text-align:left;text-transform:uppercase;letter-spacing:.07em;color:var(--muted);font-size:10px;padding:9px 10px;border-bottom:1px solid var(--line)}} td{{padding:10px;border-bottom:1px solid #edf1f6;white-space:nowrap}} tr:last-child td{{border-bottom:0}} td:not(:first-child),th:not(:first-child){{text-align:right}} .meter{{height:8px;width:100px;background:#edf2f7;border-radius:9px;overflow:hidden}} .meter b{{display:block;height:100%;background:linear-gradient(90deg,#60a5fa,#2563eb);border-radius:9px}} .pct{{font-weight:700}} .rank{{display:inline-grid;place-items:center;width:22px;height:22px;margin-right:8px;border-radius:7px;background:#eef2ff;color:#4f46e5;font-size:11px;font-weight:750}} .spark{{width:126px;height:28px;color:#2563eb}}
.note{{display:grid;grid-template-columns:auto 1fr;gap:12px;margin-top:17px;padding:15px 17px;background:#eff6ff;border:1px solid #bfdbfe;border-radius:11px;color:#344a67}} .note b{{color:#1d4ed8}} footer{{color:var(--muted);font-size:12px;margin-top:16px;text-align:center}} code{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.92em}}
@media(max-width:850px){{.cards{{grid-template-columns:1fr 1fr}}.cols{{grid-template-columns:1fr}}.panel{{overflow-x:auto}}}} @media(max-width:520px){{.wrap{{padding:14px}}header{{padding:25px}}h1{{font-size:27px}}.cards{{grid-template-columns:1fr}}}}
</style></head><body><main class="wrap">
<header><p class="eyebrow">Beaker workspace · ai2/flex2</p><h1>GPU Utilization Report</h1><p>13 completed weeks · {html.escape(report["window_label"])} · 192-GPU full capacity</p></header>
<section class="cards">
<div class="card"><small>Average utilization</small><strong>{avg:.1f}%</strong><em>{avg*1.92:.1f} of 192 GPUs</em></div>
<div class="card"><small>Peak week</small><strong>{peak["utilization_pct"]:.1f}%</strong><em>{html.escape(peak["label"])}</em></div>
<div class="card"><small>Allocated GPU-hours</small><strong>{total:,.0f}</strong><em>across the report window</em></div>
<div class="card"><small>Active users</small><strong>{len(users)}</strong><em>{report["summary"]["job_count"]:,} GPU jobs</em></div>
</section>
<section class="panel"><h2>Weekly occupancy by user</h2><p class="sub">Share of the 192-GPU fleet reserved by running Flex2 jobs. Hover a segment for detail.</p>{stacked_chart(chart_weeks, chart_users, colors)}<div class="legend">{legend}</div></section>
<div class="cols"><section class="panel"><h2>Week-by-week totals</h2><p class="sub">One full week has 32,256 available GPU-hours.</p><table><thead><tr><th>Week</th><th>GPU-h</th><th>Avg GPUs</th><th></th><th>Use</th></tr></thead><tbody>{weekly_rows}</tbody></table></section>
<section class="panel"><h2>User summary</h2><p class="sub">Ranked by allocated GPU-hours; sparkline shows weekly usage.</p><table><thead><tr><th>User</th><th>GPU-h</th><th>Share</th><th>Avg GPUs</th><th>Peak fleet</th><th>Trend</th></tr></thead><tbody>{user_rows}</tbody></table></section></div>
<aside class="note"><b>Definition</b><span>This report measures <strong>allocated GPU occupancy</strong>: requested GPUs × running time, apportioned across week boundaries. It does not measure device-level compute activity within an allocation. Capacity is fixed at 192 GPUs. Batch experiment jobs in <code>ai2/flex2</code> are included; storage and other resources are not.</span></aside>
<footer>Generated {generated.strftime("%B %-d, %Y at %-I:%M %p %Z")} · Source: Beaker experiment history · Window uses America/Los_Angeles week boundaries</footer>
</main></body></html>'''


def build_report(args: argparse.Namespace) -> Dict[str, Any]:
    tz = ZoneInfo(args.timezone)
    report_end_local = datetime.fromisoformat(args.end).replace(tzinfo=tz)
    report_start_local = report_end_local - timedelta(weeks=args.weeks)
    report_start = report_start_local.astimezone(timezone.utc)
    report_end = report_end_local.astimezone(timezone.utc)
    scan_start = report_start - timedelta(days=args.lookback_days)

    week_bounds = [report_start_local + timedelta(weeks=i) for i in range(args.weeks + 1)]
    used = [defaultdict(float) for _ in range(args.weeks)]
    job_ids = set()
    job_count = 0
    experiments_scanned = 0

    for experiment in iter_recent_experiments(scan_start, args.workers):
        experiments_scanned += 1
        for job in experiment.get("jobs") or []:
            job_id = job.get("id")
            if not job_id or job_id in job_ids or job.get("kind") != "execution":
                continue
            job_ids.add(job_id)
            ngpu = gpu_count(job)
            interval = job_interval(job, report_end)
            if ngpu <= 0 or interval is None:
                continue
            start, end = interval
            if end <= report_start or start >= report_end:
                continue
            author = ((job.get("author") or {}).get("name") or (experiment.get("author") or {}).get("name") or "unknown")
            counted = False
            for i in range(args.weeks):
                left = week_bounds[i].astimezone(timezone.utc)
                right = week_bounds[i + 1].astimezone(timezone.utc)
                seconds = (min(end, right) - max(start, left)).total_seconds()
                if seconds > 0:
                    used[i][author] += seconds * ngpu / 3600
                    counted = True
            if counted:
                job_count += 1

    weeks = []
    user_weekly = defaultdict(lambda: [0.0] * args.weeks)
    for i in range(args.weeks):
        left, right = week_bounds[i], week_bounds[i + 1]
        capacity = args.capacity * (right.astimezone(timezone.utc) - left.astimezone(timezone.utc)).total_seconds() / 3600
        by_user = dict(sorted(used[i].items(), key=lambda item: (-item[1], item[0])))
        total = sum(by_user.values())
        for user, gpu_hours in by_user.items():
            user_weekly[user][i] = gpu_hours
        weeks.append({
            "start": left.isoformat(), "end": right.isoformat(),
            "label": f'{left.strftime("%b %-d")}–{(right - timedelta(days=1)).strftime("%b %-d")}',
            "short_label": left.strftime("%b %-d"),
            "capacity_gpu_hours": capacity, "used_gpu_hours": total,
            "average_gpus": total / (capacity / args.capacity),
            "utilization_pct": 100 * total / capacity, "by_user": by_user,
        })

    total_capacity = sum(week["capacity_gpu_hours"] for week in weeks)
    total_used = sum(week["used_gpu_hours"] for week in weeks)
    users = []
    for user, values in user_weekly.items():
        user_total = sum(values)
        users.append({
            "user": user, "gpu_hours": user_total,
            "share_pct": 100 * user_total / total_used if total_used else 0,
            "average_gpus": user_total / (total_capacity / args.capacity),
            "peak_week_pct": max(100 * values[i] / weeks[i]["capacity_gpu_hours"] for i in range(args.weeks)),
            "weekly_gpu_hours": values,
        })
    users.sort(key=lambda item: (-item["gpu_hours"], item["user"]))
    peak = max(weeks, key=lambda item: item["utilization_pct"])
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(), "workspace": WORKSPACE,
        "timezone": args.timezone, "capacity_gpus": args.capacity,
        "window_label": f'{report_start_local.strftime("%B %-d")}–{(report_end_local - timedelta(days=1)).strftime("%B %-d, %Y")}',
        "report_start": report_start_local.isoformat(), "report_end": report_end_local.isoformat(),
        "method": "allocated_gpu_hours", "experiments_scanned": experiments_scanned,
        "scan_lookback_days": args.lookback_days, "weeks": weeks, "users": users,
        "summary": {
            "used_gpu_hours": total_used, "capacity_gpu_hours": total_capacity,
            "average_utilization_pct": 100 * total_used / total_capacity,
            "job_count": job_count,
            "peak_week": {"label": peak["label"], "utilization_pct": peak["utilization_pct"]},
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--end", default="2026-08-03T00:00:00", help="exclusive local-time end")
    parser.add_argument("--weeks", type=int, default=13)
    parser.add_argument("--capacity", type=int, default=192)
    parser.add_argument("--timezone", default="America/Los_Angeles")
    parser.add_argument("--lookback-days", type=int, default=45)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("reports/flex2-gpu-report-2026.html"))
    parser.add_argument("--data-output", type=Path, default=Path("reports/flex2-gpu-report-2026.json"))
    args = parser.parse_args()
    if args.weeks <= 0 or args.capacity <= 0 or args.workers <= 0:
        parser.error("weeks, capacity, and workers must be positive")

    report = build_report(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.data_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_html(report), encoding="utf-8")
    args.data_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "html": str(args.output), "data": str(args.data_output),
        "weeks": len(report["weeks"]), "users": len(report["users"]),
        "jobs": report["summary"]["job_count"],
        "average_utilization_pct": round(report["summary"]["average_utilization_pct"], 2),
    }, indent=2))


if __name__ == "__main__":
    main()
