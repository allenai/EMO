BASE="/data/input/ai2-llm"
TOKENIZER="allenai/dolma2-tokenizer"

if [[ -z "$1" ]]; then
  echo "Usage: $0 <path-to-txt-file>" >&2
  exit 1
fi

grep -v "^#" "$1" | grep -v "^$" | cut -d',' -f2 | \
    sed "s|{TOKENIZER}|$TOKENIZER|g" | \
    while read path; do
      full="$BASE/$path"
      if [[ -f "$full" ]]; then
        stat -c%s "$full"
      else
        echo "MISSING: $full" >&2
      fi
    done | awk '{sum += $1; count++} END {print count " files, " sum/1e12 " TB, " sum/4/1e9 " B tokens"}'