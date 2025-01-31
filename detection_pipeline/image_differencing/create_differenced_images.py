import argparse
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from detection_pipeline.image_differencing import difference_images


SKIP_IMAGES = {
    "09_2015-04-29_122428_DSC_0364.thumb.jpg",
    "18_2015-04-29_122428_DSC_0561.thumb.jpg"
}


def process_image_chunk(args: list):
    n, images, differenced_images, folder_2015_before_after, folder_differenced_images = args

    fireball_image = images[n+1]

    if fireball_image in differenced_images:
        return

    image1 = cv2.imread(Path(folder_2015_before_after, images[n]), cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(Path(folder_2015_before_after, fireball_image), cv2.IMREAD_GRAYSCALE)
    image3 = cv2.imread(Path(folder_2015_before_after, images[n+2]), cv2.IMREAD_GRAYSCALE)

    differenced_image_pair1 = difference_images(image2, image1)
    differenced_image_pair2 = difference_images(image2, image3)

    brightness_image_pair1 = np.mean(differenced_image_pair1)
    brightness_image_pair2 = np.mean(differenced_image_pair2)

    # print(fireball_image)
    # print(f"    Pair 1: {brightness_image_pair1:.5f}")
    # print(f"    Pair 2: {brightness_image_pair2:.5f}")

    image_to_save = differenced_image_pair1 if brightness_image_pair1 < brightness_image_pair2 else differenced_image_pair2

    cv2.imwrite(
        Path(folder_differenced_images, fireball_image),
        image_to_save,
        [cv2.IMWRITE_JPEG_QUALITY, 100]
    )


@dataclass
class Args:
    images_folder: str
    output_folder: str | None
    overwrite: bool


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create differenced images from before, current, and after images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--images_folder", type=str, required=True, help="Path to the folder containing before, current, and after images")
    parser.add_argument("--output_folder", type=str, required=False, default=None,
        help="The folder that will be created to contain differenced images. If None, will be under <images_folder>/differenced_images/")
    parser.add_argument('--overwrite', action='store_true', default=False, help='Overwrite the output directory if it exists.')
    
    args = Args(**vars(parser.parse_args()))
    print("\nargs:", json.dumps(vars(args), indent=4), "\n")

    folder_2015_before_after = Path(args.images_folder)
    if args.output_folder:
        folder_differenced_images = Path(args.output_folder)
    else:
        folder_differenced_images = Path(folder_2015_before_after, "differenced_images")

    if folder_differenced_images.exists():
        if args.overwrite:
            print("removing existing folder...\n")
            shutil.rmtree(folder_differenced_images)
        else:
            print(f"\"{folder_differenced_images}\" already exists. include --overwrite option to overwrite folders.")
            return
    else:
        os.makedirs(folder_differenced_images, exist_ok=True)

    differenced_images = set(os.listdir(folder_differenced_images))
    images = [i for i in sorted(os.listdir(folder_2015_before_after)) if Path(folder_2015_before_after, i).is_file()]

    print(len(images))

    for i in SKIP_IMAGES:
        images.remove(i)

    print(len(images))

    args_list = [
        (i * 3, images, differenced_images, folder_2015_before_after, folder_differenced_images)
        for i in range(len(images) // 3)
    ]

    with ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(process_image_chunk, args_list), total=len(args_list)))


if __name__ == "__main__":
    main()