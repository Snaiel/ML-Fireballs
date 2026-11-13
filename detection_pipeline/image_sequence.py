"""Select temporal neighbours independently of queues, logging and inference."""

import re
from dataclasses import dataclass
from datetime import datetime

from utils.constants import MAX_TIME_DIFFERENCE


@dataclass
class ImagesTriple:
    index: int
    before: str | None
    current: str
    after: str | None


def filter_camera_images(images: list[str], camera_folder: str):
    """Reject mismatched numeric camera prefixes in explicit DFN camera folders.

    Generic folder names do not imply a camera ID and are left unchanged.
    """
    camera = re.fullmatch(r"DFN(?:SMALL|KIT)(\d+)", camera_folder)
    if camera is None:
        return list(images), []
    expected = int(camera.group(1))
    accepted, rejected = [], []
    for image in images:
        prefix = image.split("_", 1)[0]
        matches = prefix.isdigit() and int(prefix) == expected
        (accepted if matches else rejected).append(image)
    return accepted, rejected


def build_image_triples(images: list[str], max_difference: int = MAX_TIME_DIFFERENCE):
    """Return processing triples and orphan filenames in chronological input order.

    Input images must already have passed brightness checks. Use full dates so
    adjacency across midnight does not produce a negative time-of-day interval.
    """
    times = []
    for image in images:
        match = re.search(r"\d{4}-\d{2}-\d{2}_\d{6}", image)
        if match is None:
            raise ValueError(f"Missing image timestamp: {image}")
        times.append(datetime.strptime(match.group(), "%Y-%m-%d_%H%M%S"))
    triples = []
    skipped = []
    for i, current in enumerate(images):
        before = images[i-1] if i > 0 and 0 <= (times[i]-times[i-1]).total_seconds() <= max_difference else None
        after = images[i+1] if i+1 < len(images) and 0 <= (times[i+1]-times[i]).total_seconds() <= max_difference else None
        if before is None and after is None:
            skipped.append(current)
        else:
            triples.append(ImagesTriple(len(triples), before, current, after))
    return triples, skipped
