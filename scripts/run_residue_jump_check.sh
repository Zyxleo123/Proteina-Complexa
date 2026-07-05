#!/usr/bin/env bash
set -euo pipefail

SCRIPT="$(dirname "$0")/check_pdb_residue_jumps.py"
DIR="$1"
LABEL="$2"

echo "================================================================================"
echo "DIRECTORY: $LABEL ($DIR)"
echo "================================================================================"

total=0
with_issues=0
real_breaks=0
numbering_only=0
clean=0

while IFS= read -r -d '' pdb; do
    total=$((total + 1))
    name=$(basename "$pdb")
    out=$(python3 "$SCRIPT" "$pdb" 2>&1 || true)

    file_real=$(echo "$out" | grep -c "^REAL_BREAK:" || true)
    file_num=$(echo "$out" | grep -c "^NUMBERING_ONLY:" || true)

    if [[ "$file_real" -gt 0 || "$file_num" -gt 0 ]]; then
        with_issues=$((with_issues + 1))
        echo ""
        echo ">>> $name"
        echo "$out" | grep -E "^(REAL_BREAK|NUMBERING_ONLY|Chain )" || true
    else
        clean=$((clean + 1))
    fi

    real_breaks=$((real_breaks + file_real))
    numbering_only=$((numbering_only + file_num))
done < <(find "$DIR" -type f \( -name '*.pdb' -o -name '*.pdb.gz' \) -print0 | sort -z)

echo ""
echo "--- Summary for $LABEL ---"
echo "Files scanned:        $total"
echo "Files with gaps:      $with_issues"
echo "Files clean:          $clean"
echo "REAL_BREAK events:    $real_breaks"
echo "NUMBERING_ONLY events: $numbering_only"
