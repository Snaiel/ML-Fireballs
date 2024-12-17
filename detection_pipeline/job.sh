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
bash

cd $MYSOFTWARE/ML-Fireballs

source .venv/bin/activate

python3 -m detection_pipeline.main \
    --folder_path "$FOLDER_PATH" \
    --model_path "$MODEL_PATH" \
    --detector ONNX \
    --processes "$SLURM_CPUS_PER_TASK"