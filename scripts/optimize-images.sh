#!/bin/sh
# Runs the image optimiser with the portfolio venv (Pillow) when present.
PY=/Users/oliviajackson/Documents/portfolio/.venv/bin/python
[ -x "$PY" ] || PY=python3
exec "$PY" "$(dirname "$0")/optimize-images.py"
