#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <input_folder> <output_folder> <model_path>"
    exit 1
fi

input_folder="$1"
output_folder="$2"
model_path="$3"

# Check if the input folder exists
if [ ! -d "$input_folder" ]; then
    echo "Error: Input folder '$input_folder' does not exist."
    exit 1
fi

# Check if the model file exists
if [ ! -f "$model_path" ]; then
    echo "Error: Model file '$model_path' does not exist."
    exit 1
fi

# Create the output folder with the basename of the input folder
input_basename=$(basename "$input_folder")
output_path="$output_folder/$input_basename"
mkdir -p "$output_path"

echo "Created output folder: $output_path"

# Submit a job for each sub folder (assumed to be a folder for a camera's images that night)
for subfolder in "$input_folder"/*; do
    if [ -d "$subfolder" ]; then
        subfolder_basename=$(basename "$subfolder")
        echo "Submitting job for: $input_basename/$subfolder_basename"
        sbatch \
            --export=ALL,FOLDER_PATH="$subfolder",OUTPUT_PATH="$output_path",MODEL_PATH="$model_path" \
            --output="$output_path/slurm-%j-$input_basename-$subfolder_basename.out"\
            "$MYSOFTWARE/ML-Fireballs/detection_pipeline/job.sh"
    fi
done