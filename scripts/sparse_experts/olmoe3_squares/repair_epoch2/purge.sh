#!/usr/bin/env bash
# Delete the pre-repair (epoch-2-tainted) copies that launch.sh / downstream.sh set aside: every *_epoch2 directory under
# sparse_experts/ and every *.epoch2.json under sparse_experts/ + claude_outputs/olmoe3_routing/. User-approved 2026-09-18
# ("you can delete them"). Re-run as the downstream driver sets more aside.   bash .../purge.sh [--dry-run]
set -u; cd "$(git rev-parse --show-toplevel)"; DRY="${1:-}"
mapfile -t dirs < <(find sparse_experts -maxdepth 3 -type d -name '*_epoch2' -prune 2>/dev/null | sort)
mapfile -t files < <(find sparse_experts claude_outputs/olmoe3_routing -maxdepth 4 -type f -name '*.epoch2.json' 2>/dev/null | sort)
[ ${#dirs[@]} -eq 0 ] && [ ${#files[@]} -eq 0 ] && { echo "$(date -u +%H:%M) nothing to purge"; exit 0; }
[ ${#dirs[@]} -gt 0 ] && du -sh "${dirs[@]}" 2>/dev/null
printf '%s\n' "${files[@]}"
[ "$DRY" = --dry-run ] && exit 0
[ ${#dirs[@]} -gt 0 ] && rm -rf "${dirs[@]}"; [ ${#files[@]} -gt 0 ] && rm -f "${files[@]}"
echo "$(date -u +%H:%M) purged ${#dirs[@]} dirs, ${#files[@]} files; free now: $(df -h /root/EMO | awk 'NR==2{print $4}')"
