#!/bin/bash
# Usage: ./get_mix_size.sh <mix_file.txt> <base_dir>
# Totals file sizes for all .npy paths listed in a mix file.

set -euo pipefail

mix_file="$1"
base_dir="$2"

total=0
while IFS= read -r line; do
  # skip blank lines and comments
  [[ -z "$line" || "$line" == \#* ]] && continue
  path="${line#*,}"
  full="$base_dir/$path"
  if [[ -f "$full" ]]; then
    size=$(stat -f%z "$full" 2>/dev/null || stat -c%s "$full" 2>/dev/null)
    total=$((total + size))
    echo "$size  $path"
  else
    echo "MISSING  $path" >&2
  fi
done < "$mix_file"

echo ""
echo "Total: $total bytes ($(echo "scale=2; $total / 1073741824" | bc) GB)"
