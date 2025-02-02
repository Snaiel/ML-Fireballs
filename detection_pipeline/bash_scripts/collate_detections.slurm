#!/bin/bash

#SBATCH --job-name=collate_detections
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --partition=work


if [ -z "$OUTPUT_PATH" ]; then
    echo "Error: OUTPUT_PATH is not defined."
    exit 1
fi


module load pytorch/2.2.0-rocm5.7.3


bash << 'EOF'

cd $MYSOFTWARE/ML-Fireballs

source .venv/bin/activate

python3 -m detection_pipeline.core.collate_detections "$OUTPUT_PATH"

EOF
