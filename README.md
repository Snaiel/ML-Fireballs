# ML-Fireballs

Repository of work for [Desert Fireball Network](https://dfn.gfo.rocks/) research. Contains code for automated point pickings and YOLO fireball detection.

# Table of Contents

- [ML-Fireballs](#ml-fireballs)
- [Table of Contents](#table-of-contents)
- [Fireball detection on Setonix](#fireball-detection-on-setonix)
  - [Installation](#installation)
  - [Custom-Trained YOLO Model](#custom-trained-yolo-model)
  - [Running Jobs](#running-jobs)
    - [Processing a Single Folder (i.e a nights worth of images by a single camera)](#processing-a-single-folder-ie-a-nights-worth-of-images-by-a-single-camera)
    - [Processing Multiple Folders (i.e each camera of a given day)](#processing-multiple-folders-ie-each-camera-of-a-given-day)
    - [Example: Processing an Entire Month of Detections](#example-processing-an-entire-month-of-detections)
  - [Useful Setonix Commands](#useful-setonix-commands)
- [Working on the project](#working-on-the-project)
  - [Locally / On a Regular Server Like Nectar](#locally--on-a-regular-server-like-nectar)
  - [Working On The Project In Setonix](#working-on-the-project-in-setonix)
- [Usage](#usage)
- [Directories](#directories)
- [Detection Pipeline](#detection-pipeline)
  - [Logs](#logs)
- [Fireball Detection](#fireball-detection)
- [Object Detection](#object-detection)
  - [What’s the Deal with object\_detection.detectors?](#whats-the-deal-with-object_detectiondetectors)
  - [Training a Model for the Detection Pipeline](#training-a-model-for-the-detection-pipeline)
    - [Requirements](#requirements)
    - [Creating Differenced Images](#creating-differenced-images)
    - [Generating the Dataset of Tiles](#generating-the-dataset-of-tiles)
    - [Training a YOLO Model on the Tiles Dataset](#training-a-yolo-model-on-the-tiles-dataset)
    - [Validating Model](#validating-model)
    - [Converting Model to ONNX Format](#converting-model-to-onnx-format)
- [Point Pickings](#point-pickings)

<br>

# Fireball detection on Setonix

## Installation

`ssh` into setonix.

https://pawsey.atlassian.net/wiki/spaces/US/pages/51925858/Connecting+to+a+Supercomputer.

You might want to make sure `$PAWSEY_PROJECT` is pointing to the correct one.

```sh
echo $PAWSEY_PROJECT
```

If it's not correct, you might have to change it within your bash config or setup stuff that handles environment variables.

*Note: there's some hardcoded stuff in the `detection_pipeline/bash_scripts/` bash scripts that assigns scheduled jobs to the account referred to by `$PAWSEY_PROJECT`.*

Go to your personal software folder.

```sh
cd $MYSOFTWARE
```

*Note: there's some hardcoded stuff in the `detection_pipeline/bash_scripts/` bash scripts that assumes that the `ML-Fireballs` repo is in your `$MYSOFTWARE` folder.*

Clone the repository.

```sh
git clone https://github.com/Snaiel/ML-Fireballs.git
```

Enter the directory.

```sh
cd ML-Fireballs/
```

Allow execution of the bash scripts in `detection_pipeline/bash_scripts/`.

```sh
chmod +x detection_pipeline/bash_scripts/*.sh
```

Run the virtual environment setup script.

```sh
./detection_pipeline/bash_scripts/venv.sh
```

<br>

## Custom-Trained YOLO Model

You need to have the custom-trained YOLO model to do the detections! Either download from https://github.com/Snaiel/ML-Fireballs/releases or train a model yourself (Refer to [Training a Model for the Detection Pipeline](#training-a-model-for-the-detection-pipeline))

If you just want to do detections and don't care about the "integrity of detection rate comparisons", just download `2015-trained-entire-year_differenced-norm-tiles_yolov8s-pt.onnx`. It is trained on (mostly) all of the 2015 fireballs (available data at the time) and uses the `.onnx` format for CPU inference on Setonix.

If you want to do testing on the first half of 2015 using Setonix for a proper detection rate comparison, use `2015-trained-jul-to-dec_differenced-norm-tiles_yolov8s-pt.onnx`.

The respective `.pt` models are essentially the base that which the `.onnx` models were converted from. They can be used for testing, during development, or exporting to other formats like `.engine` which is optimised for Nvidia GPUs.

We need to copy the models onto Setonix. We can use `scp` for that. Refer to https://pawsey.atlassian.net/wiki/spaces/US/pages/51925882/Transferring+Files+in+out+Pawsey+Filesystems.

The following is from your local machine, not within Setonix.

```sh
scp path/to/model.onnx [username]@data-mover.pawsey.org.au:/software/projects/[project]/[username]/ML-Fireballs/data/
```

For convenience, just put it in `$MYSOFTWARE/ML-Fireballs/data/`.

<br>

## Running Jobs

We'll refer to this folder structure moving forwards:

```
/scratch/$PAWSEY_PROJECT/acacia_JPGs/
```

```txt
acacia_JPGs
    dfn-l0-20150101 
        DFNSMALL09 
            09_2015-01-01_105658_DSC_0049.thumb.jpg
            09_2015-01-01_105728_DSC_0050.thumb.jpg
            09_2015-01-01_105759_DSC_0051.thumb.jpg
            09_2015-01-01_113258_DSC_0052.thumb.jpg
            09_2015-01-01_113328_DSC_0053.thumb.jpg
            ...
        DFNSMALL15
        DFNSMALL16
        DFNSMALL18
        DFNSMALL20
        ...
    dfn-l0-20150102
    dfn-l0-20150103
    dfn-l0-20150104
    dfn-l0-20150105
    ...
```

The following commands should be run while in the `ML-Fireballs/` folder within Setonix (`ssh` back into it if needed).

<br>

### Processing a Single Folder (i.e a nights worth of images by a single camera)

If you want to process a single folder containing images (e.g. `/scratch/$PAWSEY_PROJECT/acacia_JPGS/dfn-l0-20150101/DFNSMALL09`), run:

```sh
./detection_pipeline/bash_scripts/process_folder <input_folder> <output_folder> <model_path> [save_erroneous]
```

Where

- `input_folder` is the path to the folder containing images (e.g. `/scratch/$PAWSEY_PROJECT/acacia_JPGS/dfn-l0-20150101/DFNSMALL09/`)
- `output_folder` is the path to where the outputs will be. A subfolder will be created in this output folder (e.g. `$MYSCRATCH`).
- `model_path` is the path to the yolo `.onnx` model (e.g. `$MYSOFTWARE/ML-Fireballs/data/2015-trained-entire-year_differenced-norm-tiles_yolov8s-pt.onnx`)
- `save_erroneous` is an optional argument where if you put `true`, it will save the outputs of all detections, even if they were recognised as erroneous. This is useful for debugging or checking the streak lines afterwards. You can leave this out if you're just wanting to do detections though.

So the final command may look like:

```sh
./detection_pipeline/bash_scripts/process_folder /scratch/$PAWSEY_PROJECT/acacia_JPGS/dfn-l0-20150101/DFNSMALL09/ $MYSCRATCH $MYSOFTWARE/ML-Fireballs/data/2015-trained-entire-year_differenced-norm-tiles_yolov8s-pt.onnx
```

From the above command, a folder `$MYSCRATCH/DFNSMALL09/` will be created containing the outputs of the program.

<br>

### Processing Multiple Folders (i.e each camera of a given day)

If you want to process a folder containing subfolders of images (e.g. `/scratch/$PAWSEY_PROJECT/acacia_JPGS/dfn-l0-20150101/`), run:

```sh
./detection_pipeline/bash_scripts/process_subfolders <input_folder> <output_folder> <model_path> [save_erroneous]
```

See above for argument explanations.

The final command may look like:

```sh
./detection_pipeline/bash_scripts/process_subfolders /scratch/$PAWSEY_PROJECT/acacia_JPGS/dfn-l0-20150101/ $MYSCRATCH $MYSOFTWARE/ML-Fireballs/data/2015-trained-entire-year_differenced-norm-tiles_yolov8s-pt.onnx
```

From the above command, a folder `$MYSCRATCH/dfn-l0-20150101/` will be created containing the outputs of the program.

<br>

### Example: Processing an Entire Month of Detections

`acacia_JPGs/` may have folders of days from multiple months.

If we want to process the folders from January 2015, first make a folder to house all the outputs.

```sh
mkdir $MYSCRATCH/dfn-2015-01-candidates/
```

The following command will submit jobs to process the folders from January 2015:

```sh
find /scratch/$PAWSEY_PROJECT/acacia_JPGs/ -maxdepth 1 -type d -name "*201501*" | sort | while read dir; do
	./detection_pipeline/bash_scripts/process_subfolders.sh $dir $MYSCRATCH/dfn-2015-01-candidates/ $MYSOFTWARE/ML-Fireballs/data/2015-trained-entire-year_differenced-norm-tiles_yolov8s-pt.onnx
done
```

<br>

## Useful Setonix Commands

**View Jobs in Queue**

```sh
squeue -u $USER
```

**View Total, Running, Pending Jobs in Queue**

```sh
echo "Total: $(squeue -u $USER | tail -n +2 | wc -l), Running: $(squeue -u $USER --state=R | tail -n +2 | wc -l), Pending: $(squeue -u $USER --state=PD | tail -n +2 | wc -l)"
```

**View Queued Jobs of Other Users**

```sh
squeue --format="%u" --noheader | sort | uniq -c | sort -nr
```

**Cancel All Jobs**

```sh
scancel -u $USER
```

**View Jobs History**

```sh
sacct -u $USER
```

**View Usage**

```sh
pawseyAccountBalance -u
```

**Find Paths of Cropped Detections**

```sh
find $MYSCRATCH/dfn-2015-01-candidates -type f -regex '.*/.*_[0-9]+-[0-9]+-[0-9]+-[0-9]+\.jpg'
```

**Copy Cropped Detections to Folder**

```sh
find $MYSCRATCH/dfn-2015-01-candidates -type f -regex '.*/.*_[0-9]+-[0-9]+-[0-9]+-[0-9]+\.jpg' -exec cp {} dfn-2015-01-candidates-cropped/ \;
```

Make sure `dfn-2015-01-candidates-cropped/` exists first!

**Count Images Captured**

```sh
find /scratch/$PAWSEY_PROJECT/acacia_JPGs/dfn-l0-201501* -type f | wc -l
```

**Count Number of Detections**

```sh
for folder in $MYSCRATCH/dfn-2015-01-candidates/dfn-l0-2015*; do
    if [[ -d "$folder" ]]; then
        json_file="$folder/$(basename "$folder").json"
        if [[ -f "$json_file" ]]; then
            jq -r '.final_detections[]' "$json_file"
        else
            echo "Skipping: $json_file (JSON file not found)"
        fi
    fi
done | wc -l
```

These point to the same detections as [Find Paths of Cropped Detections](#find-paths-of-cropped-detections), just in a different format.

You can swap out `'.final_detections[]'` for

- `'.detections[]'`
- `'.erroneous_detections[]'`

**Count Number of Images with Detections**

```sh
for folder in $MYSCRATCH/dfn-2015-01-candidates/dfn-l0-2015*; do
    if [[ -d "$folder" ]]; then
        json_file="$folder/$(basename "$folder").json"
        if [[ -f "$json_file" ]]; then
            jq -r '.final_detections[]' "$json_file"
        else
            echo "Skipping: $json_file (JSON file not found)"
        fi
    fi
done | awk -F'/' '{print $2}' | sort | uniq | wc -l
```

<br>

# Working on the project

Here are the steps if you are going to be tinkering around or doing development.

## Locally / On a Regular Server Like Nectar

Clone repository.

```sh
git clone https://github.com/Snaiel/ML-Fireballs.git
```

Enter directory.

```sh
cd ML-Fireballs/
```

Create virtual environment.

```sh
python3 -m venv .vev
```

Activate virtual environment.

```sh
source .venv/bin/activate
```

Install dependencies.

```sh
pip install -r requirements.txt
```

<br>

## Working On The Project In Setonix

oh boy.

Install the project. Refer to the [Installation](#installation) section in [Fireball detection on Setonix](#fireball-detection-on-setonix).

Make changes locally, push commits. Then in Setonix pull changes. If you don't want to be committing code, then I guess just tinker while in Setonix.

To run code in setonix. Ask for an interactive slurm session.

```sh
salloc -N 1 -n 1 -c 8
```

This will provide access to one node, beings used for 1 task (this isn't really that important in this case, we're using the regular Python multiprocessing so it still basically counts as 1 task), using 8 cores. Change the amount of cores `-c` accordingly (if testing the detection pipeline use the max, `128`).

**This next part is important to make sure the environment is right when running code. This is the intuition behind `detection_pipeline/bash_scripts/venv.sh`**

The following was learned after deciphering https://pawsey.atlassian.net/wiki/spaces/US/pages/51931230/PyTorch.

Load the PyTorch module.

```sh
module load pytorch/2.2.0-rocm5.7.3
```

*Note: PyTorch is very important for the entire YOLO object detection part. Even if the `ultralytics` package is not being directly used, running inference on the models requires math to be done by PyTorch.*

Enter the subshell. **THIS IS IMPORTANT!!!**

```sh
bash
```

*Note: Pawsey uses a container software called Singularity. I have no idea how it works. But what I do know is that you have to enter the subshell that runs the PyTorch module.*

The shell should look like

```sh
Singularity>
```

Navigate to the project if not already.

```sh
cd $MYSOFTWARE/ML-Fireballs/
```

Activate the virtual envinroment.

```
source .venv/bin/activate
```

Now be free, prosper, run the code, and hopefully you won't crash the supercomputer. Just remember you're in an interactive slurm session, so there's probably a one hour time limit. You could always provide a longer time limit when doing `salloc` but that'll be left as an exercise for the reader.

If you haven't created the virtual environment for whatever reason or you want to know how things work, read ahead...

This is basically what `detection_pipeline/bash_scripts/venv.sh` does.

Assuming you're in the subshell and in the `ML-Fireballs/` directory, create the virtual environment.

```sh
python3 -m venv .venv --system-site-packages
```

*Note: `--system-site-packages` is very important!!! This takes in the packages from the subshell, pulling in the needed environment for PyTorch.*

Activate the virtual environment.

```sh
source .venv/bin/activate
```

Install dependencies.

```sh
mkdir -p $MYSCRATCH/tmp
export TMPDIR="$MYSCRATCH/tmp"
mkdir -p "$TMPDIR"

pip install --cache-dir="$TMPDIR" -r requirements.txt
pip install --cache-dir="$TMPDIR" onnxruntime
```

`$TMPDIR` is used because the home user folder can fill up quickly.

`onnxruntime` is used for running inference on the model using CPUs.

<br>

# Usage

Everything is designed to work (hopefully) from the root project directory (`ML-Fireballs/`). If a particular script/file is meant to be run directly, you run it as a module. For example, to run

```
point_pickings/misc/scikit_blobs.py
```

use the following command:

```sh
python3 -m point_pickings.misc.scikit_blobs
```

Notice how `.` is used for package separators and `.py` is omitted. Tab completion isn't available which does suck when typing things out...

<br>

# Directories

`detection_pipeline`: performing detections on a folder containing a sequence of images.

`fireball_detection`: tiling and performing detections on standalone full-sized images.

`object_detection`: training and testing a YOLO object detection model for fireballs in tiles.

`point_pickings`: automating the point pickings process.

<br>

# Detection Pipeline

Building off of the paper: [Fireball streak detection with minimal CPU processing requirements for the Desert Fireball Network data processing pipeline
](https://doi.org/10.1017/pasa.2019.48) by Towner et al. (2020).

**Overview of the new detection pipeline**

![Drawing 2025-01-25 19 31 21 excalidraw](https://github.com/user-attachments/assets/320a4d2c-4264-45c1-8e80-4d162420ee5a)

This folder contains code for detecting fireballs in a folder of images. Functionality such as checking brightness of whole images, image differencing, tile pixel thresholds, and streak line analysis.

`detection_pipeline.main` is the main program where you give it a folder of images, an optional output destination, and a trained yolo model and it will run the detection pipeline on the input folder using the model, then generate outputs in the designated destination.

## Logs

The logs made by the pipeline use the **JSON Lines** format. You can use the `jq` command to view them in a human readable way.

```sh
jq . $MYSCRATCH/dfn-2015-01-candidates/dfn-l0-20150101/DFNSMALL09/dfn-l0-20150101_DFNSMALL09.log | less
```

Piping it to `less` makes it navigable. Up and down arrows, page up and page down, `g` to go to the start, `shift + g` to go to the end, `<num> + g` to go that line e.g. to go to line 11, type`11g`. `q` to quit.

`jq` works well when parsing through it. For example, to output the final detections by a camera:

```sh
jq -r 'select(.final_detections != null) | .final_detections[]' "$MYSCRATCH/dfn-2015-01-candidates/dfn-l0-20150101/DFNSMALL09/dfn-l0-20150101_DFNSMALL09.log"
```

<br>

# Fireball Detection

`fireball_detection` contains code for splitting an input image into tiles, running a yolo model on tiles, then repositioning and merging detections together.

**This only deals with detecting a fireball in a single image. It is better to use `detection_pipeline` for proper fireball detection on a folder of images since it uses image differencing and does analysis on the detected streak lines.**

`fireball_detection.detect` has the detection system implemented as a function where you call `detect_fireballs` with an image and it returns bounding boxes and confidence of detected fireballs.

Running the module also shows a sample detection.

```sh
python3 -m fireball_detection.detect
```

Animation source: https://github.com/Snaiel/Manim-Fireball-Detection

https://github.com/user-attachments/assets/a7e529c7-e998-486c-b863-5cc67f60fd0a

<br>

# Object Detection

This deals with YOLOv8 stuff.

## What’s the Deal with object_detection.detectors?

YOLOv8 is made through the `ultralytics` Python package. It makes things very convenient by doing a lot of the work for you. But when doing detections on Setonix, more control over the inference using `onnxruntime` was required. So the custom `Detector` abstract class was made for this project, along with `ONNXDetector` and `UltralyticsDetector`. `ONNXDetector` just has some configurations that make things smoother for CPU inference in general, not just for Setonix. `UltralyticsDetector` basically just feeds the input into `ultralytics`, used in development for the convenience.

<br>

## Training a Model for the Detection Pipeline

oh boy.

### Requirements

What you'll need:

- A folder containaing original JPGs of fireballs, along with their before and after images.

For example:

```
ML-Fireballs/data/fireballs_before_after/
    07_2015-03-18_140459_DSC_0350.thumb.jpg
    07_2015-03-18_140529_DSC_0351.thumb.jpg
    07_2015-03-18_140559_DSC_0352.thumb.jpg
    07_2015-03-21_153258_DSC_0497.thumb.jpg
    07_2015-03-21_154558_DSC_0498.thumb.jpg
    07_2015-03-21_154628_DSC_0499.thumb.jpg
    07_2015-04-11_164128_DSC_0670.thumb.jpg
    07_2015-04-11_164158_DSC_0671.thumb.jpg
    07_2015-04-11_164228_DSC_0672.thumb.jpg
    ...
```

In this case, the fireball images are:

```
    07_2015-03-18_140529_DSC_0351.thumb.jpg
    07_2015-03-21_154558_DSC_0498.thumb.jpg
    07_2015-04-11_164158_DSC_0671.thumb.jpg
    ...
```

<br>

- A folder containing corresponding point pickings CSVs.

**IMPORTANT: Unfortunately, the required path of these CSVs are hardcoded throughout the project. Please ensure that the CSVs are located in the following folder:**

```
ML-Fireballs/data/GFO_fireball_object_detection_training_set/point_pickings_csvs/
```

For example:

```
ML-Fireballs/data/GFO_fireball_object_detection_training_set/point_pickings_csvs/
    07_2015-03-18_140529_DSC_0351.csv
    07_2015-03-21_154558_DSC_0498.csv
    07_2015-04-11_164158_DSC_0671.csv
```

Make sure they are in the following format:

```
07_2015-03-18_140529_DSC_0351.csv
```

```csv
x_image_thumb,y_image_thumb
5170.27009161,4224.71460824
5151.98774013,4252.4533484
5135.5966664,4276.40953309
5099.66238936,4329.36530978
5088.62993589,4345.44117056
...
```

Since this is used throughout the project, it might useful just to put all the known fireball CSVs in this folder.

<br>

### Creating Differenced Images

The first step is to create the differenced images. Make sure you've got the virtual environment activated (refer to installing [Locally / On a Regular Server Like Nectar](#locally--on-a-regular-server-like-nectar))! The following is executed in the `ML-Fireballs/` directory.

We will use `detection_pipeline.image_differencing.create_differenced_images`

```sh
python3 -m detection_pipeline.image_differencing.create_differenced_images -h
```

For example:

```sh
python3 -m detection_pipeline.image_differencing.create_differenced_images data/fireballs_before_after/
```

This will create the folder `data/fireballs_before_after/differenced_images/`.

<br>

### Generating the Dataset of Tiles

The second step will be to generate all the tiles from these differenced images.

We will use `object_detection.dataset.differenced.generate_all_tiles`

```sh
python3 -m object_detection.dataset.differenced.generate_all_tiles -h
```

For example:

```sh
python3 -m object_detection.dataset.differenced.generate_all_tiles data/fireballs_before_after/ data/fireballs_before_after/differenced_images/
```

This will create the folder `data/object_detection/differenced/all/`.

```
ML-Fireballs/data/object_detection/differenced/all/
    images/
    labels/
    data.yaml
    fireballs.txt
```

You can rename `data/object_detection/differenced/` to something more recognisable or if you want to differentiate between datasets.

This is enough to train a model on all of this data, but if you want to create a validation and training split, you'll have to modify `object_detection.dataset.differenced.create_2015_halved` or write another python file that mimics it, sorry!

Running `object_detection.dataset.differenced.create_2015_halved` will create the training set of fireballs from July to December 2015 and a validation set of fireballs from January to June 2015. Coincidentally, this creates a convenient 80/20 train/validation split.

```sh
python3 -m object_detection.dataset.differenced.create_2015_halved -h
```

For example

```sh
python3 -m object_detection.dataset.differenced.create_2015_halved --all_folder_path data/object_detection/differenced/all/
```

This will created the folder `data/object_detection/differenced/halved/`

`negative_ratio` refers to the ratio of tiles containing no fireballs to the fireball tiles. The default value of `-1` means it will use all the available negative samples in the dataset. **It was found through testing that having all available negative samples in the dataset provides the best performance.** A custom negative ratio may not result in the exact amount since there may not be enough negative samples.

<br>

### Training a YOLO Model on the Tiles Dataset

The third step is to train a YOLO model on created dataset. We will be using `object_detection.model.train`

**It is important to use a GPU-enabled device. Nectar VMs worked nicely!**

```sh
python3 -m object_detection.model.train -h
```

For example:

```sh
python3 -m object_detection.model.train --data_yaml_path data/object_detection/differenced/all/data.yaml --yolo_model yolov8s.pt --batch_size 0.8
```

Outputs of YOLO training go to `ML-Fireballs/runs/detect/`.

The above command should result in a folder `ML-Fireballs/runs/detect/object_detection-differenced-all-yolov8s.pt/`

If you point towards an "all" folder, there won't be a validation set to compare against as training progresses. In the generated `data.yaml` file, the validation set points towards the same training set in this case, so `ultralytics` will do one validation test at the very end on it but the usefuless of that is up to you.

Using `yolov8s.pt` means we are using the pre-trained YOLOv8 small model. YOLOv8 over YOLO11, the `small` model size, and the pre-trained model were found to be better.

Be careful about `--batch_size`, probably start high then work your way down from `0.8`, you'll know if it's too high because it'll crash... Needs good amount of GPU VRAM and also regular RAM.

For a system like (RTX 3060 12GB VRAM + 32GB RAM), a batch size of `0.5` was achievable. For a Nectar VM with an A100 40GB VRAM (I forgot how much regular RAM it had), it worked well with `0.8`.

A high batch size is preferrable because it considers more samples during a training pass, which makes training faster and it provides slightly more performance as it supposedly considers a bigger sample when modifying the network's weights (supported somewhat by the literature).

<br>

### Validating Model

Ultralytic's inbuilt validation isn't that useful for us since its metrics aren't reliable and we want to know the detection rate of fireballs as a whole instead of just on the tiles.

We will use `object_detection.val.val_tiles`.

```sh
python3 -m object_detection.val.val_tiles -h
```

For example:

```sh
python3 -m object_detection.val.val_tiles --yolo_pt_path runs/detect/object_detection-differenced-all-yolov8s.pt/weights/last.pt --data_yaml_path data/object_detection/differenced/all/data.yaml 
```

Validating on an "all" dataset might not allow for a reliable comparison because the model has already seen the data in the training set but it enables a view of the expected fireballs to be detected when rerunning detections on already-trained data.

If for example we did create a validation set, the program will test accordingly and provide a better judgement of the model on unseen data.

Example output:

```
(.venv) ubuntu@fireballs-detection:/data/ML-Fireballs$ python3 -m object_detection.val.val_tiles --data_yaml_path data/object_detection/2015_differenced_norm_tiles/halved/data.yaml --yolo_pt_path runs/detect/object_detection-2015_differenced_norm_tiles-halved-yolov8s.pt/weights/last.pt 

args: {
    "border_size": 5,
    "data_yaml_path": "data/object_detection/2015_differenced_norm_tiles/halved/data.yaml",
    "yolo_pt_path": "runs/detect/object_detection-2015_differenced_norm_tiles-halved-yolov8s.pt/weights/last.pt",
    "samples": "both",
    "metric": "iom",
    "threshold": 0.5,
    "show_false_negatives": false,
    "save_false_negatives": false
} 

runs/detect/object_detection-2015_differenced_norm_tiles-halved-yolov8s.pt/weights/last.pt 

kfold folder: data/object_detection/2015_differenced_norm_tiles/halved/data.yaml 

Total samples:                 11901
Positive samples:              2007
Negative samples:              9894

running predictions: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 11901/11901 [02:02<00:00, 97.28it/s]

Detected samples:              1808
False negative samples:        199
Recall on individual samples:  0.90085

Total fireballs:               248
Detected fireballs:            244
False negative fireballs:      4
Recall on entire fireballs:    0.98387

Missed Fireballs:
10_2015-06-12_140028_DSC_0509
20_2015-04-03_132758_DSC_0042
34_2015-03-29_133259_DSC_0385
35_2015-06-26_155628_DSC_0811

Total boxes:                   2121
True positives:                2027
False positives:               94
Precision:                     0.95568
```

<br>

### Converting Model to ONNX Format

If we want to run the model on Setonix, we need to convert to ONNX, a format that is optimised for CPU inference.

We will use `object_detection.model.export`

```sh
python3 -m object_detection.model.export -h
```

For example:

```sh
python3 -m object_detection.model.export --yolo_pt_path runs/detect/object_detection-differenced-all-yolov8s.pt/weights/last.pt
```

This will create the following file: `runs/detect/object_detection-differenced-all-yolov8s.pt/weights/last.onnx`.

Rename and move where needed.

If you want to export to any other formats, just modify `object_detection.model.export` accordingly (Refer to https://docs.ultralytics.com/modes/export/).

<br>

Congratulations, you've trained a model! It can now be used for fireball detection.

<br>

# Point Pickings

`point_pickings.process` has the automated point picking system implemented as a function where you call `retrieve_fireball` with an image (assumed to be cropped) and returns all the processed position and timing information.

Running the module shows a sample with the distances between segments being labelled.

```sh
python3 -m point_pickings.process
```

<br>


Running `point_pickings.compare` shows a comparison between the automated system and the manual point pickings.

```sh
python3 -m point_pickings.compare
```

<br>
