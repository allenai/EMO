#!/bin/bash
# Publish all experiment reports to Cloudflare Pages:
#
#   https://emo-reports.pages.dev/                  (index)
#   https://emo-reports.pages.dev/<experiment>.html (one page per experiment)
#
#   bash scripts/publish_reports.sh
#
# Deploys are direct uploads (pre-built assets; no build quota consumed) and
# the CDN cache is invalidated on deploy, so updates are visible in seconds.
#
# Registering an experiment: add a "name|blurb" line to EXPERIMENTS below.
# Each experiment <name> must provide scripts/<name>/build_report.py that
# writes claude_outputs/<name>/report.html (self-contained HTML, images
# embedded, <25 MiB). Experiments whose report is missing are skipped with a
# warning.
#
# Credentials: CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN are read from
# the env or from /root/.config/cloudflare/emo_reports.env (never committed —
# this repo is public). Wrangler runs off the Node install in
# /root/.local/node (both live on weka, so they persist across compute nodes).
set -euo pipefail
cd "$(dirname "$0")/.."

EXPERIMENTS=(
    "models_sizescaling|expert specialization vs pool size (32/64/96/128 experts)"
    "models_fullextend|ghost-expert pretraining so new experts can be added post-training (sweep in progress)"
)

PROJECT="emo-reports"
SITE="https://${PROJECT}.pages.dev"

if [ -z "${CLOUDFLARE_API_TOKEN:-}" ]; then
    source /root/.config/cloudflare/emo_reports.env
fi
command -v wrangler >/dev/null || export PATH="/root/.local/node/bin:$PATH"

stage="$(mktemp -d)"
trap 'rm -rf "$stage"' EXIT

entries=""
for spec in "${EXPERIMENTS[@]}"; do
    name="${spec%%|*}"
    blurb="${spec#*|}"
    builder="scripts/${name}/build_report.py"
    report="claude_outputs/${name}/report.html"
    if [ -f "$builder" ]; then
        python "$builder"
    fi
    if [ ! -f "$report" ]; then
        echo "WARNING: ${report} not found, skipping ${name}" >&2
        continue
    fi
    cp "$report" "${stage}/${name}.html"
    entries+="<li><a href=\"/${name}.html\">${name}</a>
— ${blurb} <span class=\"date\">updated $(date -r "$report" +%F)</span></li>
"
done

cat > "${stage}/index.html" <<EOF
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"><title>EMO extension — experiment reports</title>
<style>
body { font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
       max-width:760px; margin:48px auto; padding:0 24px; color:#1e293b; line-height:1.6; }
h1 { font-size:22px; } li { margin:8px 0; }
.date { color:#64748b; font-size:13px; }
</style>
</head>
<body>
<h1>EMO extension — experiment reports</h1>
<ul>
${entries}</ul>
</body>
</html>
EOF

wrangler pages deploy "$stage" --project-name "$PROJECT" --branch main
echo "Published: ${SITE}/"
