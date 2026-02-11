#!/bin/bash
# =============================================================================
# Count tokens in tokenized data
# =============================================================================
#
# Counts tokens in Dolma-tokenized .npy files (uint32 = 4 bytes per token).
#
# Usage:
#   ./count_tokens.sh /path/to/tokenized/data
#
# Examples:
#   ./count_tokens.sh /data/croissant/tokenized/french_303b_1
#   ./count_tokens.sh /data/croissant/tokenized  # counts all subdirs
#
# Output:
#   Directory: /data/croissant/tokenized
#   Bytes:     120000000000
#   Tokens:    30000000000
#   Billions:  30.00B
#
# NOTE: Only works for tokenized .npy/.bin files, NOT for JSONL/text files.
#       For JSONL, the byte count doesn't equal tokens - tokenize first.
# =============================================================================

DIR="${1:-.}"

if [ ! -d "$DIR" ]; then
    echo "Error: $DIR is not a directory"
    exit 1
fi

# Count only tokenized files (.npy, .bin)
BYTES=$(find "$DIR" -type f \( -name "*.npy" -o -name "*.bin" \) -print0 2>/dev/null | xargs -0 du -cb 2>/dev/null | tail -1 | cut -f1)

if [ -z "$BYTES" ] || [ "$BYTES" -eq 0 ]; then
    echo "Directory: $DIR"
    echo "No tokenized files (.npy/.bin) found."
    echo ""
    echo "If this is JSONL data, you need to tokenize it first with dolma."
    echo "Raw byte count (not tokens):"
    du -sh "$DIR"
    exit 0
fi

TOKENS=$((BYTES / 4))
BILLIONS=$(echo "scale=2; $TOKENS / 1000000000" | bc)

echo "Directory: $DIR"
echo "Bytes:     $BYTES"
echo "Tokens:    $TOKENS"
echo "Billions:  ${BILLIONS}B"
