#!/bin/bash
#SBATCH --job-name=multiprocessing_example
#SBATCH --output=job_output.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=work

if [ -z "$FOLDER_PATH" ] || [ -z "$MODEL_PATH" ]; then
    echo "Error: FOLDER_PATH or MODEL_PATH is not defined."
    exit 1
fi

module unload python
module load pytorch/2.2.0-rocm5.7.3

module load py-matplotlib/3.8.1
module load py-numpy/1.26.1
module load py-pandas/2.1.2
module load py-scikit-learn/1.3.2
module load py-pip/23.1.2-py3.11.6

bash

cd $MYSOFTWARE/ML-Fireballs

if [ ! -d ".venv" ]; then
    echo ".venv does not exist. Creating virtual environment and installing dependencies..."
    python3 -m venv .venv --system-site-packages
    source .venv/bin/activate
    mkdir -p $MYSCRATCH/tmp
    TMPDIR=$MYSCRATCH/tmp pip install -r requirements.txt
else
    source .venv/bin/activate
fi

python3 -m detection_pipeline.main \
    --folder_path "$FOLDER_PATH" \
    --model_path "$MODEL_PATH" \
    --detector ONNX \
    --processes "$SLURM_CPUS_PER_TASK"