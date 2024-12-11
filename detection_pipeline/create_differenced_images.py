import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from skimage import io
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

    image1 = io.imread(Path(folder_2015_before_after, images[n]))
    image2 = io.imread(Path(folder_2015_before_after, fireball_image))
    image3 = io.imread(Path(folder_2015_before_after, images[n+2]))

    differenced_image_pair1 = difference_images(image2, image1)
    differenced_image_pair2 = difference_images(image2, image3)

    brightness_image_pair1 = np.mean(differenced_image_pair1)
    brightness_image_pair2 = np.mean(differenced_image_pair2)

    # print(fireball_image)
    # print(f"    Pair 1: {brightness_image_pair1:.5f}")
    # print(f"    Pair 2: {brightness_image_pair2:.5f}")

    image_to_save = differenced_image_pair1 if brightness_image_pair1 < brightness_image_pair2 else differenced_image_pair2

    io.imsave(
        Path(folder_differenced_images, fireball_image),
        image_to_save,
        check_contrast=False,
        quality=100
    )


def main() -> None:

    folder_2015_before_after = Path("data/2015_before_after")
    folder_differenced_images = Path(folder_2015_before_after, "differenced_images")

    if not folder_differenced_images.exists():
        os.mkdir(folder_differenced_images)

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