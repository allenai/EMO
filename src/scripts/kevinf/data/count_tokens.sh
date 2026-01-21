#!/bin/bash

BASE="/data/input/ai2-llm"
TOKENIZER="allenai/dolma2-tokenizer"

if [[ -z "$1" ]]; then
  echo "Usage: $0 <path-to-txt-file>" >&2
  exit 1
fi

if [[ ! -f "$1" ]]; then
  echo "ERROR: Input file not found: $1" >&2
  exit 1
fi

# Create a temp file to store subset,size pairs
tmpfile=$(mktemp)
missing_file=$(mktemp)
trap "rm -f $tmpfile $missing_file" EXIT

# Process each line and collect subset name with file size
grep -v "^#" "$1" | grep -v "^$" | while IFS=',' read -r subset path; do
  path=$(echo "$path" | sed "s|{TOKENIZER}|$TOKENIZER|g")
  full="$BASE/$path"
  if [[ -f "$full" ]]; then
    size=$(stat -c%s "$full")
    echo "$subset,$size"
  else
    echo "MISSING: $full" >&2
    echo "$full" >> "$missing_file"
  fi
done > "$tmpfile"

# Check if any files were missing
if [[ -s "$missing_file" ]]; then
  missing_count=$(wc -l < "$missing_file")
  echo "" >&2
  echo "ERROR: $missing_count file(s) not found. Aborting." >&2
  exit 1
fi

# Print per-subset summary
echo "=== Tokens by Subset ==="
awk -F',' '
{
  subset[$1] += $2
  count[$1]++
  total += $2
  total_count++
}
END {
  for (s in subset) {
    pct = (total > 0) ? (subset[s] / total * 100) : 0
    printf "%-30s %5d files, %8.4f TB, %10.2f B tokens (%5.1f%%)\n", s, count[s], subset[s]/1e12, subset[s]/4/1e9, pct
  }
  print ""
  print "=== Total ==="
  printf "%-30s %5d files, %8.4f TB, %10.2f B tokens (100.0%%)\n", "TOTAL", total_count, total/1e12, total/4/1e9
}
' "$tmpfile"
