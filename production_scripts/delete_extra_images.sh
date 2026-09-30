#!/usr/bin/env bash
# Usage: ./delete_extra_images.sh [dry-run|move|delete]
#
# Example:
# chmod +x delete_extra_images.sh
# ./delete_extra_images.sh              # dry run
# ./delete_extra_images.sh move         # reversible
# # after checking that nothing is broken:
# rm -rf /path/to/output/images_extra_quarantine    # or ./delete_extra_images.sh delete
set -euo pipefail

IMAGES_DIR="/home/thebaulm/split/31/2025/rasters/images"
LIST="$(realpath /home/thebaulm/split/31/2025/rasters/images/extra_images.txt)"
QUARANTINE="/home/thebaulm/images_extra_quarantine/split/31/2025/rasters/images/"
MODE="${1:-dry-run}"

[[ -d "$IMAGES_DIR" ]] || { echo "Images dir not found: $IMAGES_DIR"; exit 1; }
[[ -f "$LIST" ]]       || { echo "List not found: $LIST"; exit 1; }

# Safety: only plain .jpg file names, no paths, no odd characters
if grep -qvE '^[A-Za-z0-9_-]+\.jpg$' "$LIST"; then
    echo "Unexpected lines in $LIST (paths or odd names):"
    grep -vnE '^[A-Za-z0-9_-]+\.jpg$' "$LIST" | head
    exit 1
fi

TOTAL=$(wc -l < "$LIST")
cd "$IMAGES_DIR"

# How many listed files really exist
EXISTING=$(xargs -a "$LIST" -d '\n' ls -d 2>/dev/null | wc -l || true)
echo "Listed: $TOTAL | existing in $IMAGES_DIR: $EXISTING | mode: $MODE"

case "$MODE" in
  dry-run)
    echo "First 10 files that would be affected:"
    head -n 10 "$LIST"
    echo "Nothing was changed. Re-run with 'move' or 'delete'."
    ;;
  move)
    mkdir -p "$QUARANTINE"
    xargs -a "$LIST" -d '\n' mv -t "$QUARANTINE" --
    echo "Moved to $QUARANTINE"
    ;;
  delete)
    read -r -p "Permanently delete $EXISTING files from $IMAGES_DIR? Type 'yes': " ANSWER
    [[ "$ANSWER" == "yes" ]] || { echo "Aborted."; exit 1; }
    xargs -a "$LIST" -d '\n' rm -f --
    echo "Deleted."
    ;;
  *)
    echo "Unknown mode: $MODE (use dry-run, move or delete)"; exit 1 ;;
esac