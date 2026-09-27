#!/usr/bin/env bash
# The resume PDF used to be hand-uploaded to assets/, and that URL is already out
# in the world on sent applications. Copy the generated PDF over the top of it so
# those links keep resolving to the current version instead of a frozen one.
set -euo pipefail

PDF="Olivia_Jackson_Lambert_Resume.pdf"
SRC="${QUARTO_PROJECT_OUTPUT_DIR:-_site}/${PDF}"
DEST_DIR="${QUARTO_PROJECT_OUTPUT_DIR:-_site}/assets"

# Single-file renders of other pages will not have produced the PDF; skip quietly.
if [[ ! -f "$SRC" ]]; then
  exit 0
fi

mkdir -p "$DEST_DIR"
cp "$SRC" "$DEST_DIR/$PDF"

# Keep the copy checked into assets/ current too, so a fresh clone and a plain
# `quarto render` both publish the same file.
cp "$SRC" "assets/$PDF"

echo "synced $PDF -> $DEST_DIR/ and assets/"
