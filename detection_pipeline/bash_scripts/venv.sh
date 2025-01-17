#!/bin/bash

module load pytorch/2.2.0-rocm5.7.3

bash << 'EOF'

cd $MYSOFTWARE/ML-Fireballs
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
mkdir -p $MYSCRATCH/tmp
export TMPDIR="$MYSCRATCH/tmp"
mkdir -p "$TMPDIR"
pip install --cache-dir="$TMPDIR" -r requirements.txt
pip install --cache-dir="$TMPDIR" onnxruntime

EOF