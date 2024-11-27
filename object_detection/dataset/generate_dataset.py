import argparse
import multiprocessing as mp
import os
import shutil
import signal
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Full

from tqdm import tqdm

from object_detection.dataset import DATA_FOLDER, DATA_YAML, GFO_JPEGS
from object_detection.dataset.split_tiles import SplitTilesFireball


# Sentinel value to indicate the end of the queue processing
_SENTINEL = None


def _generate_tiles(fireball_name: str, negative_ratio: int, images_folder: Path, labels_folder: Path) -> None:
    """
    Generate tile images and labels for a given fireball image.

    Parameters:
    - fireball_name (str): The name of the fireball image to process.
    - negative_ratio (int): The ratio of negative examples to positive examples.
    - images_folder (Path): Path to save the generated images.
    - labels_folder (Path): Path to save the generated labels.
    """
    fireball = SplitTilesFireball(fireball_name, negative_ratio)
    fireball.save_images(images_folder)
    fireball.save_labels(labels_folder)


def _run_generate_tiles(names_queue: mp.Queue, bar_queue: mp.Queue, negative_ratio: int, images_folder: Path, labels_folder: Path) -> None:
    """
    Process fireball names from the queue to generate tiles in a multiprocessing context.

    It will put progress signals in `bar_queue` after processing each item.

    Parameters are similar to `generate_tiles` with additional queue handling.
    """
    try:
        while True:
            fireball_name = names_queue.get()
            if fireball_name is _SENTINEL:
                break
            _generate_tiles(fireball_name, negative_ratio, images_folder, labels_folder)
            bar_queue.put_nowait(1)
    except (Full, Empty) as e:
        print(type(e))
        return


def _update_bar(bar_queue: mp.Queue, total: int) -> None:
    """
    Update a tqdm progress bar based on signals from the queue.

    Parameters:
    - bar_queue (mp.Queue): A queue to receive progress signals.
    - total (int): Total number of tasks/items to process.
    """
    pbar = tqdm(total=total, desc="generating tiles")
    while True:
        bar_queue.get(True)
        pbar.update(1)


def _create_tiles(num_processes: int, negative_ratio: int, images_folder: Path, labels_folder: Path) -> None:
    """
    Organize multiprocessing to generate tiles for fireball images.

    Parameters are used to control the ratio and destination paths for images and labels.
    """
    fireball_images = sorted(os.listdir(GFO_JPEGS))

    names_queue = mp.Queue()
    for fireball_image in fireball_images:
        names_queue.put_nowait(fireball_image.split(".")[0])
    
    # Adding sentinel values to signal the processes they can stop
    for _ in range(num_processes):
        names_queue.put(_SENTINEL)

    bar_queue = mp.Queue()
    # Process for updating the progress bar
    bar_process = mp.Process(target=_update_bar, args=(bar_queue, len(fireball_images)), daemon=True)
    bar_process.start()

    print()

    processes: list[mp.Process] = []
    # Starting worker processes
    for _ in range(num_processes):
        process = mp.Process(target=_run_generate_tiles, args=(names_queue, bar_queue, negative_ratio, images_folder, labels_folder))
        processes.append(process)
        process.start()

    try:
        for process in processes:
            process.join()
    except KeyboardInterrupt:
        # Graceful shutdown in case of an interruption
        names_queue.close()
        bar_queue.close()
        for process in processes:
            process.terminate()
            process.join()
        os.kill(os.getpid(), signal.SIGTERM)


def main():
    """
    Main function to parse arguments and initiate the dataset generation process.
    """
    @dataclass
    class Args:
        """
        Data class to store command-line arguments.
        """
        negative_ratio: int = 1
        overwrite: bool = False
        num_processes: int = 8

    parser = argparse.ArgumentParser(
        description="Generate dataset for object detection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--negative_ratio', type=int, default=1, required=True, 
                        help='Ratio of negative examples to positive examples.')
    parser.add_argument('--overwrite', action='store_true', 
                        help='Overwrite the output directory if it exists.')
    parser.add_argument('--num_processes', type=int, default=8,
                        help='Number of processes to use for multiprocessing.')


    args = Args(**vars(parser.parse_args()))
    print(f"\nArgs: {vars(args)}")

    object_detection_folder_name = f"object_detection_1_to_{args.negative_ratio}"
    object_detection_folder = Path(DATA_FOLDER, object_detection_folder_name)
    all_folder = Path(object_detection_folder, "all")
    all_images_folder = Path(all_folder, "images")
    all_labels_folder = Path(all_folder, "labels")

    if Path(object_detection_folder).exists():
        if args.overwrite:
            print("\nremoving existing folder...")
            shutil.rmtree(object_detection_folder)
        else:
            print(f"\"{object_detection_folder}\" already exists. include --overwrite option to overwrite folders.")
            return
    
    os.mkdir(object_detection_folder)
    os.mkdir(all_folder)
    os.mkdir(all_images_folder)
    os.mkdir(all_labels_folder)

    shutil.copy(DATA_YAML, all_folder)
    with open(Path(all_folder, "data.yaml"), "r+") as yaml_file:
        content = yaml_file.read()
        content = content.replace("object_detection", f"{object_detection_folder_name}/all")
        content = content.replace("images/train", "images")
        content = content.replace("images/val", "images")
        yaml_file.seek(0)
        yaml_file.write(content)
        yaml_file.truncate()

    with open(Path(all_folder, "fireballs.txt"), "w") as fireballs_file:
        fireballs_file.write(
            "\n".join(map(lambda x: x.replace(".thumb.jpg", ""), sorted(os.listdir(GFO_JPEGS))))
        )

    _create_tiles(args.num_processes, args.negative_ratio, all_images_folder, all_labels_folder)


if __name__ == "__main__":
    main()