#!/bin/bash

echo "loading pytorch module"
module load pytorch/2.2.0-rocm5.7.3

echo "entering subshell"
bash << 'EOF'

cd $MYSOFTWARE/ML-Fireballs

echo "creating virtual environment"
python3 -m venv .venv --system-site-packages
source .venv/bin/activate

echo "installing dependencies"

mkdir -p $MYSCRATCH/tmp
export TMPDIR="$MYSCRATCH/tmp"
mkdir -p "$TMPDIR"

pip install --cache-dir="$TMPDIR" -r requirements.txt
pip install --cache-dir="$TMPDIR" onnxruntime

EOF