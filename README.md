# Automating Fireballs

Repository of work for [Desert Fireball Network](https://dfn.gfo.rocks/) research. Contains code for automated point pickings and YOLO fireball detection.

<br>

## Installation

Clone repository.

```sh
git clone https://github.com/Snaiel/Automating-Fireballs.git
```

Enter directory.

```sh
cd Automating-Fireballs/
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

## Directories

`object_detection` contains code for training and testing a YOLO object detection model for fireballs in tiles.

`fireball_detection` contains code for tiling and performing detections on full-sized images.

`point_pickings` contains code for automating the point pickings process.

<br>

## Usage

Everything is designed to work (hopefully) from the project directory. If a particular script/file is meant to be run directly, you run it as a module. For example, to run

```
point_pickings/misc/scikit_blobs.py
```

use the following command:

```sh
python3 -m point_pickings.misc.scikit_blobs
```

Notice how `.` is used for package separators and `.py` is omitted. Tab completion isn't available which does suck when typing things out...

<br>

## Object Detection

### Dataset Creation

`object_detection.dataset.generate_dataset` generates a dataset of 400x400 pixel tiles from the full sized images.

Running the module generates the dataset with a specified positive to negative tile sample ratio.

```sh
python3 -m object_detection.dataset.generate_dataset
```

<br>

`object_detection.dataset.create_kfold_dataset` creates a 5-fold kfold dataset using the tiles generated from the `generate_dataset` script.

Running the module creates the different splits based on the dataset with the specified negative tile sample ratio.

```sh
python3 -m object_detection.dataset.create_kfold_dataset
```

<br>

### Model Training

`object_detection.model.train` trains a YOLO model using the generated dataset.

Running the module performs the training process with a specified dataset.

```sh
python3 -m object_detection.model.train
```

<br>

## Fireball Detection

`fireball_detection.detect` has the detection system implemented as a function where you call `detect_fireballs` with an image and it returns bounding boxes and confidence of detected fireballs.

Running the module also shows a sample detection.

```sh
python3 -m fireball_detection.detect
```

<br>

## Point Pickings

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
