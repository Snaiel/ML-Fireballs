import multiprocessing as mp
import os
import signal
from pathlib import Path
from queue import Empty, Full

from tqdm import tqdm

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


def _tiles_progress_bar(bar_queue: mp.Queue, total: int) -> None:
    """
    Create progress bar and update a based on signals from the queue.

    Parameters:
    - bar_queue (mp.Queue): A queue to receive progress signals.
    - total (int): Total number of tasks/items to process.
    """
    pbar = tqdm(total=total, desc="generating tiles")
    while True:
        bar_queue.get(True)
        pbar.update(1)


def create_tiles(num_processes: int, negative_ratio: int, fireballs: list[str], images_folder: Path, labels_folder: Path) -> None:
    """
    Organize multiprocessing to generate tiles for fireball images.

    Parameters are used to control the ratio and destination paths for images and labels.
    """

    names_queue = mp.Queue()
    for fireball in fireballs:
        names_queue.put_nowait(fireball)
    
    # Adding sentinel values to signal the processes they can stop
    for _ in range(num_processes):
        names_queue.put(_SENTINEL)

    bar_queue = mp.Queue()
    # Process for updating the progress bar
    bar_process = mp.Process(target=_tiles_progress_bar, args=(bar_queue, len(fireballs)), daemon=True)
    bar_process.start()

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