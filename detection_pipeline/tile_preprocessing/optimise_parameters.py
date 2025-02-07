"""
This uses bayesian optimisation to optimise the tiles
preprocessing thresholds.

*sigh* this tangent was kinda interesting. I learnt a lot
about bayesian optimisation. But this was kinda useless in
the end. My initial guesses turned out to be perfect. It
validated my assumptions though which is a win.
"""

import argparse
import json
import os
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import skimage.io as io
from sklearn.metrics import fbeta_score, precision_score, recall_score
from skopt import gp_minimize
from skopt.plots import plot_evaluations, plot_objective
from skopt.space import Integer
from tqdm import tqdm

from detection_pipeline.tile_preprocessing import (TilePreprocessingThresholds,
                                                   satisfies_thresholds)
from fireball_detection.tiling import (get_image_tile,
                                       retrieve_included_coordinates)
from utils.constants import MIN_POINTS_IN_TILE, RANDOM_SEED, SQUARE_SIZE
from utils.paths import GFO_PICKINGS, GFO_THUMB_EXT


included_coordinates = retrieve_included_coordinates()


FBETA_BETA = 10 

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


def evaluate_thresholds(differenced_image_path: Path, thresholds: TilePreprocessingThresholds):
    
    differenced_image = io.imread(differenced_image_path)
    predictions = []

    for tile_pos in included_coordinates:
        differenced_tile = get_image_tile(differenced_image, tile_pos)
        prediction = satisfies_thresholds(differenced_tile, thresholds)
        predictions.append(prediction)
    
    return predictions


def evaluate_with_params(args):
    return evaluate_thresholds(*args)


def optimise_thresholds(ground_truth, differenced_images):
    def objective(params):
        thresholds = TilePreprocessingThresholds(*params)
        map_args = [(img, thresholds) for img in differenced_images]
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
        Integer(1000, 160000, name="max_pixel"),
        Integer(1, 1000, name="variance_threshold")
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
        ground_truth_results = list(tqdm(pool.imap(process_fireball, fireballs), total=len(fireballs), desc="Generating ground truth labels"))
    for result in ground_truth_results:
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

    plot_evaluations(result)
    plot_objective(result)

    thresholds = TilePreprocessingThresholds(*result.x)

    # thresholds = TilePreprocessingThresholds(
    #     11,
    #     105,
    #     100000,
    #     50
    # )

    # thresholds = None

    print(thresholds)
    print()

    map_args = [(img, thresholds) for img in differenced_images]

    predictions = []
    full_images_kept_at_least_half = 0
    full_images_kept_at_least_one = 0
    
    with Pool() as pool:
        predictions_results = list(tqdm(pool.imap(evaluate_with_params, map_args), total=len(differenced_images), desc="Evaluating thresholds"))
    for result in predictions_results:
        predictions.extend(result)
    
    print()
    print("Missed fireballs:")
    for i, g_tiles, p_tiles in zip(range(len(ground_truth_results)), ground_truth_results, predictions_results):
        true_count = sum(1 for g, p in zip(g_tiles, p_tiles) if g and p)
        if true_count > 0:
            full_images_kept_at_least_one += 1
            if true_count > sum(1 for g in g_tiles if g) // 2:
                full_images_kept_at_least_half += 1
        else:
            print(differenced_images[i].name)
    
    print()
    print(f"Tiles removed: {predictions.count(False)}/{len(predictions)} ({predictions.count(False)/len(predictions)})")
    print(f"Fireballs kept (At least half of tiles detected): {full_images_kept_at_least_half}/{len(ground_truth_results)} ({full_images_kept_at_least_half / len(ground_truth_results)})")
    print(f"Fireballs kept (At least one tile detected): {full_images_kept_at_least_one}/{len(ground_truth_results)} ({full_images_kept_at_least_one / len(ground_truth_results)})")
    print()
    print("Precision:", precision_score(ground_truth, predictions))
    print("Recall:", recall_score(ground_truth, predictions))
    print(f"F{int(FBETA_BETA)}:", fbeta_score(ground_truth, predictions, beta=FBETA_BETA))

    plt.show()


if __name__ == "__main__":
    main()