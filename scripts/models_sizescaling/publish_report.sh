#!/bin/bash
# Rebuild the sizescaling report and publish it to the (secret) reports gist,
# served rendered at a stable URL via gist.githack.com.
#
#   bash scripts/models_sizescaling/publish_report.sh
#
# The gist ID is deliberately NOT committed (this repo is public; committing it
# would expose the unlisted URL). It is read from the untracked file
# claude_outputs/.report_gist_id, or the GIST_ID env var. The gist history is
# kept at a single commit (amend + force-push) so it never grows.
#
# Stable URLs (always latest push):
#   https://gist.githack.com/<user>/<gist_id>/raw/index.html
#   https://gist.githack.com/<user>/<gist_id>/raw/models_sizescaling.html
set -euo pipefail
cd "$(dirname "$0")/../.."

GIST_ID="${GIST_ID:-$(cat claude_outputs/.report_gist_id)}"
GH_USER="$(gh api user --jq .login)"

python scripts/models_sizescaling/build_report.py

workdir="$(mktemp -d)"
trap 'rm -rf "$workdir"' EXIT
git clone --quiet "https://$(gh auth token)@gist.github.com/${GIST_ID}.git" "$workdir/gist"

cp claude_outputs/models_sizescaling/report.html "$workdir/gist/models_sizescaling.html"
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
<li><a href="https://gist.githack.com/${GH_USER}/${GIST_ID}/raw/models_sizescaling.html">models_sizescaling</a>
— expert specialization vs pool size (32/64/96/128 experts)
<span class="date">updated $(date +%F)</span></li>
</ul>
</body>
</html>
EOF

cd "$workdir/gist"
git add -A
git -c user.name="$GH_USER" -c user.email="${GH_USER}@users.noreply.github.com" \
    commit --quiet --amend -m "publish reports"
git push --quiet --force
echo "Published: https://gist.githack.com/${GH_USER}/${GIST_ID}/raw/index.html"
