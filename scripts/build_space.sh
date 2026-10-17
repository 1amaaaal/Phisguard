#!/usr/bin/env bash
# Assemble a ready-to-upload Hugging Face Space in build/space/.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=build/space
# Flat layout (no folders), because the Hugging Face browser upload flattens
# folders anyway. app.py knows how to load the package from this layout.
rm -rf "$OUT" && mkdir -p "$OUT"
cp huggingface/app.py huggingface/requirements.txt huggingface/README.md "$OUT/"
cp phishguard/__init__.py phishguard/model.py phishguard/features.py phishguard/urltools.py "$OUT/"
cp models/phishguard.joblib "$OUT/"
echo "Space files ready in $OUT/:"
find "$OUT" -type f | sort
