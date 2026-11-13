"""Scientific-coordinate regression checks; no weights or inference required."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from skimage import io

from fireball_detection import Tile
from fireball_detection.boxes.fireball_boxes import get_absolute_fireball_boxes
from fireball_detection.detect import detect_tiles_common
from object_detection.val.val_tiles import Args, val_split


class BorderedDetector:
    """An oracle returns a known box in the input-image coordinate system."""

    def __init__(self, border):
        self.border = max(0, border)

    def detect(self, image):
        assert image.shape == (400 + 2*self.border, 400 + 2*self.border, 3)
        # Include tile-edge coordinates, not only interior boxes.
        return [[self.border, self.border + 10, self.border + 400,
                 self.border + 390]], [0.9], [0]


class BorderCoordinateTests(unittest.TestCase):
    def test_tile_to_full_image_coordinates(self):
        for border in (0, 5, -1):
            for origin in ((0, 0), (200, 600), (6960, 4512)):
                with self.subTest(border=border, origin=origin):
                    tile = Tile(origin, np.zeros((400, 400, 3), dtype=np.uint8))
                    detected = detect_tiles_common(BorderedDetector(border), border, [tile])
                    self.assertEqual(detected[0].boxes, [(0., 10., 400., 390.)])
                    full = get_absolute_fireball_boxes(detected)[0]
                    self.assertEqual(full.box, (origin[0], origin[1]+10,
                                                origin[0]+400, origin[1]+390))
                    self.assertEqual(full.conf, 0.9)

    def test_empty_detections_stay_empty(self):
        detector = SimpleNamespace(detect=lambda image: ([], [], []))
        tile = Tile((200, 600), np.zeros((400, 400, 3), dtype=np.uint8))
        self.assertEqual(detect_tiles_common(detector, 5, [tile]), [])

    def test_saved_january_prediction(self):
        # Saved tile prediction: 2015-01-01 / DFNSMALL09 / DSC_0098.
        box = [385.0273742675781, 381.1101379394531,
               402.5876770019531, 402.5210876464844]
        detector = SimpleNamespace(detect=lambda image: ([box], [0.33635181188583374], [0]))
        tile = Tile((5600, 2800), np.zeros((400, 400, 3), dtype=np.uint8))
        detected = detect_tiles_common(detector, 5, [tile])
        full = get_absolute_fireball_boxes(detected)[0]
        np.testing.assert_allclose(full.box, [5980.027374267578, 3176.110137939453,
                                             5997.587677001953, 3197.5210876464844])
        # The old projection was exactly +5 in both axes. Box size is unchanged.
        np.testing.assert_allclose(np.array(full.box) + 5,
                                   np.array(box) + [5600, 2800, 5600, 2800])

    def test_validation_matches_ground_truth_in_bordered_space(self):
        for border in (0, 5, -1):
            with self.subTest(border=border), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                for kind in ("images", "labels"):
                    (root / kind / "val").mkdir(parents=True)
                name = "09_2015-01-02_123000_DSC_0001_0"
                io.imsave(root / "images/val" / (name + ".jpg"),
                          np.zeros((400, 400, 3), dtype=np.uint8), check_contrast=False)
                # Original pixel box = [100, 100, 120, 120].
                (root / "labels/val" / (name + ".txt")).write_text("0 0.275 0.275 0.05 0.05")
                b = max(0, border)
                box = torch.tensor([[100+b, 100+b, 120+b, 120+b]], dtype=torch.float32)
                model = SimpleNamespace(predict=lambda *a, **k:
                                        [SimpleNamespace(boxes=SimpleNamespace(xyxy=box))])
                args = Args(border, str(root / "data.yaml"), "unused.pt", "positive",
                            "iou", 0.99, False, False)
                with patch("object_detection.val.val_tiles.YOLO", return_value=model):
                    stats = val_split(args)
                self.assertEqual(stats["detected_fireballs"], 1)
                self.assertEqual(stats["true_positives"], 1)
                self.assertEqual(stats["precision"], 1)


if __name__ == "__main__":
    unittest.main()
