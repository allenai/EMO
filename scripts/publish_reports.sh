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
# claude_outputs/.report_gist_id, or the GIST_ID env var.
#
# URL freshness: the stable URLs below sit behind ~5-10 min of stacked
# GitHub-raw + githack caching. To make new results reachable immediately,
# each publish makes TWO commits — reports first, then an index that links
# sha-pinned snapshot URLs (gistcdn.githack.com/.../raw/<sha>/<file>), which
# are immutable and therefore never stale. The script prints both URL kinds
# and pre-warms the snapshot URLs. History is force-pushed back to these two
# commits every publish, so the gist never grows.
#
# Stable URLs (always latest publish, may lag a few minutes after a push):
#   https://gist.githack.com/<user>/<gist_id>/raw/index.html
#   https://gist.githack.com/<user>/<gist_id>/raw/<experiment>.html
set -euo pipefail
cd "$(dirname "$0")/.."

EXPERIMENTS=(
    "models_sizescaling|expert specialization vs pool size (32/64/96/128 experts)"
)

GIST_ID="${GIST_ID:-$(cat claude_outputs/.report_gist_id)}"
GH_USER="$(gh api user --jq .login)"
GITC=(git -c user.name="$GH_USER" -c user.email="${GH_USER}@users.noreply.github.com")

workdir="$(mktemp -d)"
trap 'rm -rf "$workdir"' EXIT
git clone --quiet "https://$(gh auth token)@gist.github.com/${GIST_ID}.git" "$workdir/gist"

names=() blurbs=() dates=()
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
    names+=("$name"); blurbs+=("$blurb"); dates+=("$(date -r "$report" +%F)")
done

cd "$workdir/gist"

# Commit 1: report pages only, collapsed onto the root commit so history
# stays bounded. Its sha anchors the immutable snapshot URLs.
git reset --quiet --soft "$(git rev-list --max-parents=0 HEAD)"
git add -A
"${GITC[@]}" commit --quiet --amend -m "reports"
sha="$(git rev-parse HEAD)"

# Commit 2: index linking both the stable and the sha-pinned URLs.
entries=""
for i in "${!names[@]}"; do
    n="${names[$i]}"
    entries+="<li><a href=\"https://gist.githack.com/${GH_USER}/${GIST_ID}/raw/${n}.html\">${n}</a>
— ${blurbs[$i]} <span class=\"date\">updated ${dates[$i]}</span>
<a class=\"snap\" href=\"https://gistcdn.githack.com/${GH_USER}/${GIST_ID}/raw/${sha}/${n}.html\">[latest snapshot]</a></li>
"
done
cat > index.html <<EOF
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"><title>EMO extension — experiment reports</title>
<style>
body { font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
       max-width:760px; margin:48px auto; padding:0 24px; color:#1e293b; line-height:1.6; }
h1 { font-size:22px; } li { margin:8px 0; }
.date { color:#64748b; font-size:13px; }
.snap { font-size:13px; }
</style>
</head>
<body>
<h1>EMO extension — experiment reports</h1>
<ul>
${entries}</ul>
<p class="date">Main links always show the latest publish but can lag ~5-10 min
behind a push; [latest snapshot] links are pinned to this publish and never stale.</p>
</body>
</html>
EOF
git add index.html
"${GITC[@]}" commit --quiet -m "index"
git push --quiet --force

echo "Published (index may take ~5-10 min to refresh if recently viewed):"
echo "  index:  https://gist.githack.com/${GH_USER}/${GIST_ID}/raw/index.html"
for n in "${names[@]}"; do
    snap="https://gistcdn.githack.com/${GH_USER}/${GIST_ID}/raw/${sha}/${n}.html"
    # Pre-warm so the snapshot is cached before the upstream sha can be
    # discarded by a future force-push, and verify it serves.
    code="$(curl -sL -o /dev/null -w '%{http_code}' "$snap")"
    echo "  ${n}:"
    echo "    stable:           https://gist.githack.com/${GH_USER}/${GIST_ID}/raw/${n}.html"
    echo "    instant snapshot: ${snap}  (HTTP ${code})"
done
