#!/bin/bash
# Publish all experiment reports to the (secret) reports gist, served rendered
# at stable gist.githack.com URLs (index page + one page per experiment).
#
#   bash scripts/publish_reports.sh
#
# Registering an experiment: add a "name|blurb" line to EXPERIMENTS below.
# Each experiment <name> must provide scripts/<name>/build_report.py that
# writes claude_outputs/<name>/report.html (self-contained HTML, images
# embedded). The report is published as <name>.html; experiments whose report
# is missing are skipped with a warning.
#
# The gist ID is deliberately NOT committed (this repo is public; committing
# it would expose the unlisted URL). It is read from the untracked file
# claude_outputs/.report_gist_id, or the GIST_ID env var. The gist history is
# kept at a single commit (amend + force-push) so it never grows.
#
# Stable URLs (always latest push):
#   https://gist.githack.com/<user>/<gist_id>/raw/index.html
#   https://gist.githack.com/<user>/<gist_id>/raw/<experiment>.html
set -euo pipefail
cd "$(dirname "$0")/.."

EXPERIMENTS=(
    "models_sizescaling|expert specialization vs pool size (32/64/96/128 experts)"
)

GIST_ID="${GIST_ID:-$(cat claude_outputs/.report_gist_id)}"
GH_USER="$(gh api user --jq .login)"

workdir="$(mktemp -d)"
trap 'rm -rf "$workdir"' EXIT
git clone --quiet "https://$(gh auth token)@gist.github.com/${GIST_ID}.git" "$workdir/gist"

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
    cp "$report" "$workdir/gist/${name}.html"
    entries+="<li><a href=\"https://gist.githack.com/${GH_USER}/${GIST_ID}/raw/${name}.html\">${name}</a>
— ${blurb} <span class=\"date\">updated $(date -r "$report" +%F)</span></li>
"
done

cat > "$workdir/gist/index.html" <<EOF
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

cd "$workdir/gist"
git add -A
git -c user.name="$GH_USER" -c user.email="${GH_USER}@users.noreply.github.com" \
    commit --quiet --amend -m "publish reports"
git push --quiet --force
echo "Published: https://gist.githack.com/${GH_USER}/${GIST_ID}/raw/index.html"
