"""
    Blob Detection Script using OpenCV

    This script performs blob detection on a grayscale image using OpenCV's SimpleBlobDetector. 
    The script sets up the parameters for blob detection, detects blobs in the image, and visualizes the detected blobs.

    The main functionalities of this script include:
    - Loading a grayscale image from a specified file path.
    - Configuring blob detection parameters such as threshold, area, color, convexity, inertia, and minimum distance between blobs.
    - Initializing the blob detector based on the OpenCV version.
    - Detecting blobs in the image.
    - Drawing detected blobs on the image and displaying the result.

    Dependencies:
    - opencv-python

    Usage:
        Run the script to perform blob detection on the predefined sample image and display the detected blobs.

        Whilst in `points_processing/`, run:

        python3 blob_detection/opencv_blobs.py

        Ctrl-Z to exit.

    Note:
    OpenCV blob detection is unsophisticated and finnicky to set up.
"""

from pathlib import Path

import cv2
import numpy as np

image_path = Path(Path(__file__).parents[2], 'data', 'fireball_highlights', 'cropped', '071_2021-12-14_032259_E_DSC_0611-G_cropped.jpeg')

im = cv2.imread(
    str(image_path),
    cv2.IMREAD_GRAYSCALE
)

params = cv2.SimpleBlobDetector_Params()

params.thresholdStep = 10

params.minThreshold = 140
params.maxThreshold = 500

params.filterByArea = False
params.minArea = 5
params.maxArea = 200

params.filterByColor = True
params.blobColor = 255

params.filterByConvexity = False

params.filterByInertia = False

params.minDistBetweenBlobs = 1

for i in dir(params):
    if not str(i).startswith('_'):
        print(i, eval(f"params.{i}"))


ver = (cv2.__version__).split('.')
if int(ver[0]) < 3 :
    detector = cv2.SimpleBlobDetector(params)
else : 
    detector = cv2.SimpleBlobDetector_create(params)

# Detect blobs.
keypoints = detector.detect(im)


# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
im_with_keypoints = cv2.drawKeypoints(
    im,
    keypoints,
    np.array([]),
    (0,0,255),
    cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)


# Show keypoints
cv2.imshow("Keypoints", im_with_keypoints)
cv2.waitKey(0)