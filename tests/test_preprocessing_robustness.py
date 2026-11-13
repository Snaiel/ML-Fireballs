"""Regression cases for failures found by the independent paper reviewer."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from detection_pipeline.image_differencing.difference import difference_images
from detection_pipeline.streak_lines.streak_line import StreakLine
from utils.constants import GAUSSIAN_BLUR_SIZE


class PreprocessingRobustnessTests(unittest.TestCase):
    def test_sparse_image_falls_back_to_unaligned_difference(self):
        target = np.zeros((256, 256), dtype=np.uint8)
        target[0, 0] = 255
        reference = np.zeros_like(target)
        expected = cv2.GaussianBlur(
            target, (GAUSSIAN_BLUR_SIZE, GAUSSIAN_BLUR_SIZE), 0)
        np.testing.assert_array_equal(difference_images(target, reference), expected)

    def test_failed_affine_estimate_falls_back(self):
        image = np.random.default_rng(42).integers(0, 256, (256, 256), dtype=np.uint8)
        reference = np.zeros_like(image)
        reference[4:, 3:] = image[:-4, :-3]
        kernel = (GAUSSIAN_BLUR_SIZE, GAUSSIAN_BLUR_SIZE)
        expected = cv2.subtract(cv2.GaussianBlur(image, kernel, 0),
                                cv2.GaussianBlur(reference, kernel, 0))
        with patch('detection_pipeline.image_differencing.difference.cv2.estimateAffinePartial2D',
                   return_value=(None, None)) as estimate:
            np.testing.assert_array_equal(difference_images(image, reference), expected)
        estimate.assert_called_once()

    def test_textured_self_reference_has_zero_difference(self):
        image = np.random.default_rng(17).integers(0, 256, (256, 256), dtype=np.uint8)
        np.testing.assert_array_equal(difference_images(image, image.copy()),
                                      np.zeros_like(image))

    def test_empty_blob_crop_is_invalid_without_crashing(self):
        with tempfile.TemporaryDirectory(prefix='fireballs-review-') as directory:
            path = Path(directory) / '01_2015-01-01_000000_DSC_0001_50_0000-0000-0100-0100.differenced.jpg'
            self.assertTrue(cv2.imwrite(str(path), np.zeros((100, 100), dtype=np.uint8)))
            line = StreakLine(path)
        self.assertFalse(line.is_valid)
        self.assertEqual(line.blobs.shape[0], 0)
        self.assertEqual(line.weights.size, 0)


if __name__ == '__main__':
    unittest.main()
