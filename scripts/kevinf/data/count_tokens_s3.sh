#!/bin/bash

TOKENIZER="allenai/dolma2-tokenizer"

if [[ -z "$1" || -z "$2" ]]; then
  echo "Usage: $0 <s3-base-path> <path-to-txt-file>" >&2
  echo "Example: $0 s3://ai2-llm src/olmo_core/data/mixes/OLMo-mix-0625.txt" >&2
  exit 1
fi

BASE="$1"
MIXFILE="$2"

if [[ ! -f "$MIXFILE" ]]; then
  echo "ERROR: Input file not found: $MIXFILE" >&2
  exit 1
fi

# Create temp files
s3_cache=$(mktemp)
mix_resolved=$(mktemp)
prefixes_file=$(mktemp)
trap "rm -f $s3_cache $mix_resolved $prefixes_file $s3_cache.results" EXIT

# Resolve all paths and save to temp file: "subset,resolved_path"
echo "Resolving paths..." >&2
grep -v "^#" "$MIXFILE" | grep -v "^$" | while IFS=',' read -r subset path; do
  resolved=$(echo "$path" | sed "s|{TOKENIZER}|$TOKENIZER|g")
  echo "$subset,$resolved"
done > "$mix_resolved"

total_files=$(wc -l < "$mix_resolved")
echo "Total files in mix: $total_files" >&2

# Extract unique dataset prefixes
# Strategy: group at the tokenizer directory level or one below
# e.g., preprocessed/.../allenai/dolma2-tokenizer/ or preprocessed/.../allenai/dolma2-tokenizer/all-dressed-snazzy2/
echo "Finding dataset prefixes..." >&2

cut -d',' -f2 "$mix_resolved" | while read -r path; do
  # Find the tokenizer in the path and take everything up to one level after it
  # This handles both:
  #   .../tokenizer/file.npy -> .../tokenizer/
  #   .../tokenizer/subdir/subsubdir/file.npy -> .../tokenizer/subdir/
  
  dir=$(dirname "$path")
  
  # Check if tokenizer is in path
  if [[ "$dir" == *"$TOKENIZER"* ]]; then
    # Get path up to and including tokenizer, plus one more level if exists
    before_tok="${dir%%$TOKENIZER*}"
    after_tok="${dir#*$TOKENIZER}"
    after_tok="${after_tok#/}"  # Remove leading slash
    
    # Get first directory after tokenizer (if any)
    first_subdir="${after_tok%%/*}"
    
    if [[ -n "$first_subdir" ]]; then
      echo "${before_tok}${TOKENIZER}/${first_subdir}"
    else
      echo "${before_tok}${TOKENIZER}"
    fi
  else
    # Fallback: use the full directory
    echo "$dir"
  fi
done | sort -u > "$prefixes_file"

num_prefixes=$(wc -l < "$prefixes_file")
echo "Found $num_prefixes unique dataset prefixes" >&2

# List each prefix separately and combine into cache
echo "Listing S3 files for each dataset prefix..." >&2
current=0

while read -r prefix; do
  ((current++))
  echo "[$current/$num_prefixes] Listing: $BASE/$prefix" >&2
  
  # List files under this prefix and append to cache
  # Output format: "PATH SIZE"
  aws s3 ls --recursive "$BASE/$prefix/" 2>/dev/null | \
    awk '{print $4, $3}' >> "$s3_cache"
done < "$prefixes_file"

cache_count=$(wc -l < "$s3_cache")
echo "Cached $cache_count total files from S3" >&2
echo "Looking up sizes..." >&2

# Use awk to join mix file with S3 cache (hash lookup - very fast!)
awk '
  # First pass: load S3 cache into associative array
  NR==FNR {
    path = $1
    size = $2
    sizes[path] = size
    next
  }
  
  # Second pass: process mix file
  {
    split($0, parts, ",")
    subset = parts[1]
    path = parts[2]
    
    if (path in sizes) {
      print subset "," sizes[path]
    } else {
      print "MISSING: " path > "/dev/stderr"
      missing++
    }
  }
  
  END {
    if (missing > 0) {
      print "" > "/dev/stderr"
      print "ERROR: " missing " file(s) not found." > "/dev/stderr"
      exit 1
    }
  }
' "$s3_cache" "$mix_resolved" > "$s3_cache.results"

if [[ $? -ne 0 ]]; then
  echo "Aborting due to missing files." >&2
  exit 1
fi

# Print per-subset summary (grouped by top-level label prefix)
echo "=== Tokens by Subset (Grouped) ==="
awk -F',' '
{
  # Get top-level label (before first underscore, or full label if no underscore)
  label = $1
  # Check for common prefixes like snazzy2_*, proofpile-2-*, etc.
  if (match(label, /^[a-zA-Z0-9-]+/)) {
    prefix = substr(label, RSTART, RLENGTH)
  } else {
    prefix = label
  }
  
  # Group by prefix
  grouped[prefix] += $2
  grouped_count[prefix]++
  
  # Also track individual subsets
  subset[$1] += $2
  count[$1]++
  
  total += $2
  total_count++
}
END {
  # Print grouped summary first
  for (g in grouped) {
    pct = (total > 0) ? (grouped[g] / total * 100) : 0
    printf "%-30s %5d files, %8.4f TB, %10.2f B tokens (%5.1f%%)\n", g, grouped_count[g], grouped[g]/1e12, grouped[g]/4/1e9, pct
  }
  print ""
  print "=== Total ==="
  printf "%-30s %5d files, %8.4f TB, %10.2f B tokens (100.0%%)\n", "TOTAL", total_count, total/1e12, total/4/1e9
  
  # Print detailed breakdown
  print ""
  print "=== Detailed Breakdown ==="
  for (s in subset) {
    pct = (total > 0) ? (subset[s] / total * 100) : 0
    printf "  %-28s %5d files, %8.4f TB, %10.2f B tokens (%5.1f%%)\n", s, count[s], subset[s]/1e12, subset[s]/4/1e9, pct
  }
}
' "$s3_cache.results"
