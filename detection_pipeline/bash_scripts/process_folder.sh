#!/bin/bash

# input_folder is the path to the folder of images

# DFNSMALL07 <- this one
#   0001.thumb.jpg
#   0002.thumb.jpg
#   0003.thumb.jpg
#   ...

# a folder with the basename of input_folder will be created in output_folder
# model_path is the .onnx file to the differenced object detector


# Check if the correct number of arguments is provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <input_folder> <output_folder> <model_path>"
    exit 1
fi

input_folder=$(realpath "$1")
output_folder=$(realpath "$2")
model_path=$(realpath "$3")

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

# Check if the directory exists and delete it if it does
if [ -d "$output_path" ]; then
    echo "Deleting existing output folder: $output_path"
    rm -rf "$output_path"
fi

# Create the output folder
mkdir -p "$output_path"
echo "Created output folder: $output_path"

job_id=$(sbatch \
    --export=ALL,FOLDER_PATH="$input_folder",OUTPUT_PATH="$output_path",MODEL_PATH="$model_path" \
    --output="$output_path/slurm-%j-$input_basename-$subfolder_basename.out" \
    "$MYSOFTWARE/ML-Fireballs/detection_pipeline/bash_scripts/fireball_detection.slurm" | awk '{print $4}')

echo "$input_basename job id: $job_id"