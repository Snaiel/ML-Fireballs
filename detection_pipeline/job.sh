#!/bin/bash

#SBATCH --job-name=fireball_detection
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=01:00:00
#SBATCH --partition=work


if [ -z "$FOLDER_PATH" ]; then
    echo "Error: FOLDER_PATH is not defined."
    exit 1
fi

if [ -z "$OUTPUT_PATH" ]; then
    echo "Error: OUTPUT_PATH is not defined."
    exit 1
fi

if [ -z "$MODEL_PATH" ]; then
    echo "Error: MODEL_PATH is not defined."
    exit 1
fi


module load pytorch/2.2.0-rocm5.7.3

# Don't load other python modules, it messes with the
# package versions that pytorch needs...

# definitely not these
# module load py-pandas/2.1.2
# module load py-scikit-learn/1.3.2

# these might be fine (at your own risk)
# module load py-pip/23.1.2-py3.11.6
# module load py-matplotlib/3.8.1

# important! start bash subshell in the singularity container
bash << 'EOF'

cd $MYSOFTWARE/ML-Fireballs

if [ ! -d ".venv" ]; then
    echo ".venv does not exist. Creating virtual environment and installing dependencies..."
    python3 -m venv .venv --system-site-packages
    source .venv/bin/activate
    mkdir -p $MYSCRATCH/tmp
    export TMPDIR="$MYSCRATCH/tmp"
    mkdir -p "$TMPDIR"
    pip install --cache-dir="$TMPDIR" -r requirements.txt
    pip install --cache-dir="$TMPDIR" onnxruntime
else
    source .venv/bin/activate
fi

python3 -m detection_pipeline.main \
    --folder_path "$FOLDER_PATH" \
    --output_path "$OUTPUT_PATH" \
    --model_path "$MODEL_PATH" \
    --detector ONNX \
    --processes "$SLURM_CPUS_PER_TASK"

EOF
