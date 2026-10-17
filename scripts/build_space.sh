#!/usr/bin/env bash
# Assemble a ready-to-upload Hugging Face Space in build/space/.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=build/space
rm -rf "$OUT" && mkdir -p "$OUT/models"
cp huggingface/app.py huggingface/requirements.txt huggingface/README.md "$OUT/"
mkdir -p "$OUT/phishguard"
cp phishguard/__init__.py phishguard/model.py phishguard/features.py phishguard/urltools.py "$OUT/phishguard/"
cp models/phishguard.joblib "$OUT/models/"
echo "Space files ready in $OUT/:"
find "$OUT" -type f | sort
