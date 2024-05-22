import cv2
import numpy as np
from pathlib import Path

file_path = Path(__file__)
image_path = Path(file_path.parents[1], 'fireball_images', 'cropped', '071_2021-12-14_032259_E_DSC_0611-G_cropped.jpeg')

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