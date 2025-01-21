import argparse
import json
import os
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import skimage.io as io
from sklearn.metrics import fbeta_score, precision_score, recall_score
from skopt import gp_minimize
from skopt.plots import plot_evaluations, plot_objective
from skopt.space import Integer
from tqdm import tqdm

from fireball_detection.tiling.included import retrieve_included_coordinates
from utils.constants import (GFO_PICKINGS, GFO_THUMB_EXT,
                             MAX_PIXEL_TOTAL_THRESHOLD,
                             MIN_PIXEL_TOTAL_THRESHOLD, MIN_POINTS_IN_TILE,
                             PIXEL_BRIGHTNESS_THRESHOLD, RANDOM_SEED,
                             SQUARE_SIZE)

import matplotlib.pyplot as plt

included_coordinates = retrieve_included_coordinates()


FBETA_BETA = 10.0


def process_fireball(fireball_name: str) -> list:
    points = pd.read_csv(Path(GFO_PICKINGS, fireball_name + ".csv"))
    fireball_ground_truth = []

    for tile_pos in included_coordinates:
        points_in_tile = []
        for point in points.itertuples(False, None):
            if (
                tile_pos[0] <= point[0] < tile_pos[0] + SQUARE_SIZE and
                tile_pos[1] <= point[1] < tile_pos[1] + SQUARE_SIZE
            ):
                points_in_tile.append(point)
        
        fireball_ground_truth.append(len(points_in_tile) >= MIN_POINTS_IN_TILE)
    
    return fireball_ground_truth


def evaluate_thresholds(differenced_image_path: Path, pixel_threshold: int = PIXEL_BRIGHTNESS_THRESHOLD, min_pixel: int = MIN_PIXEL_TOTAL_THRESHOLD, max_pixel: int = MAX_PIXEL_TOTAL_THRESHOLD):
    
    differenced_image = io.imread(differenced_image_path)
    predictions = []

    for tile_pos in included_coordinates:
        differenced_tile = differenced_image[tile_pos[1] : tile_pos[1] + SQUARE_SIZE, tile_pos[0] : tile_pos[0] + SQUARE_SIZE]
        pixels_over_threshold = np.sum(differenced_tile > pixel_threshold)
        prediction = min_pixel < pixels_over_threshold < max_pixel
        predictions.append(prediction)
    
    return predictions


def evaluate_with_params(args):
    img, pixel_threshold, min_pixel, max_pixel = args
    return evaluate_thresholds(img, pixel_threshold, min_pixel, max_pixel)


def optimise_thresholds(ground_truth, differenced_images):
    def objective(params):
        pixel_threshold, min_pixel, max_pixel = params
        map_args = [(img, pixel_threshold, min_pixel, max_pixel) for img in differenced_images]
        predictions = []
        with Pool() as pool:
            results = pool.map(evaluate_with_params, map_args)
        for result in results:
            predictions.extend(result)

        f_beta = fbeta_score(ground_truth, predictions, beta=FBETA_BETA)
        return -f_beta

    space = [
        Integer(1, 255, name="pixel_threshold"),
        Integer(1, 1000, name="min_pixel"),
        Integer(1000, 100000, name="max_pixel")
    ]

    result = gp_minimize(
        func=objective,
        dimensions=space,
        random_state=RANDOM_SEED,
        verbose=True
    )

    return result


def main() -> None:
    @dataclass
    class Args:
        differenced_images_folder: str

    parser = argparse.ArgumentParser(
        description="Generate dataset for object detection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--differenced_images_folder', type=str, required=True, help='Folder containing differenced images.')

    args = Args(**vars(parser.parse_args()))
    print("\nargs:", json.dumps(vars(args), indent=4), "\n")

    fireballs = list(map(lambda x: x.replace(GFO_THUMB_EXT, ""), sorted(os.listdir(args.differenced_images_folder))))

    ground_truth = []
    with Pool() as pool:
        results = list(tqdm(pool.imap(process_fireball, fireballs), total=len(fireballs), desc="Generating ground truth labels"))
    for result in results:
        ground_truth.extend(result)

    print()
    print("Total:", len(ground_truth))
    print("Count of True:", ground_truth.count(True))
    print("Count of False:", ground_truth.count(False))
    print()

    # Load differenced images
    differenced_images = list(map(lambda x: Path(args.differenced_images_folder, x), sorted(os.listdir(args.differenced_images_folder))))

    # Optimize thresholds
    print("Starting Bayesian optimization...")
    result = optimise_thresholds(ground_truth, differenced_images)

    print(result)

    pixel_threshold = result.x[0]
    min_pixel = result.x[1]
    max_pixel = result.x[2]

    # Output the best thresholds
    print()
    print("Best thresholds found:")
    print(f"Pixel Threshold: {pixel_threshold}")
    print(f"Min Pixel: {min_pixel}")
    print(f"Max Pixel: {max_pixel}")
    print(f"Best F1 Score: {-result.fun}")
    print()

    plot_evaluations(result)
    plot_objective(result)

    # pixel_threshold = 21
    # min_pixel = 151
    # max_pixel = 84119

    # pixel_threshold = PIXEL_BRIGHTNESS_THRESHOLD
    # min_pixel = MIN_PIXEL_TOTAL_THRESHOLD
    # max_pixel = MAX_PIXEL_TOTAL_THRESHOLD

    map_args = [(img, pixel_threshold, min_pixel, max_pixel) for img in differenced_images]

    predictions = []
    with Pool() as pool:
        results = list(tqdm(pool.imap(evaluate_with_params, map_args), total=len(differenced_images), desc="Evaluating thresholds"))
    for result in results:
        predictions.extend(result)

    print()
    print("Precision:", precision_score(ground_truth, predictions))
    print("Recall:", recall_score(ground_truth, predictions))
    print("F10:", fbeta_score(ground_truth, predictions, beta=FBETA_BETA))

    plt.show()


if __name__ == "__main__":
    main()