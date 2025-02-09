RANDOM_SEED = 0

IMAGE_DIMENSIONS = (7360, 4912)

# Checking image brightness
# median brightness shouldnt be higher than this
MAX_THRESHOLD_MEDIAN_BRIGHTNESS = 165
# the maximum brightness shouldnt be lower than this
MIN_THRESHOLD_MAX_BRIGHTNESS = 40

# Maximum time in seconds between images to allow for image differencing
MAX_TIME_DIFFERENCE = 180

# when aligning images, if the transformation is larger than this,
# it has done too much and has not really aligned the stars at this point.
MAX_TRANSFORMATION_MAGNITUDE = 10
# blur size when image differencing
GAUSSIAN_BLUR_SIZE = 11

# Tiles
SQUARE_SIZE = 400
# The minimum number of point pickings in the tile to use it as
# a positive sample in the object detection dataset
MIN_POINTS_IN_TILE = 3
# min diagonal length of a detection box
MIN_DIAGONAL_LENGTH = 60
# bounding box padding for creating the object detection dataset
BB_PADDING = 0.05
# minimum dimension size for the object detection dataset bounding box
MIN_BB_DIM_SIZE = 20

# The minimum brightness 0-255 of a pixel to be counted towards the total
PIXEL_BRIGHTNESS_THRESHOLD = 10
# The minimum total number of pixels above brightness threshold
MIN_PIXEL_TOTAL_THRESHOLD = 200
# maximum total number of pixels above brightness threshold
MAX_PIXEL_TOTAL_THRESHOLD = 50000
# *check code where actually used*, but the intuition behind this is that
# if the pixel total is above the max threshold, the variance must be
# above a certain point. Tries to remove tiles that are bright but
# dont have any defining features.
VARIANCE_THRESHOLD = 50

# minimum confidence value for detections
DETECTOR_CONF = 0.2
# IoU used for NMS in detectors
DETECTOR_IOU = 0.5
# border size added to tile before going through model.
# experiments showed a border size of 5 gave the best performance.
TILE_BORDER_SIZE = 5

# arguments when blob detection to establish the streak line
STREAK_LINE_BLOB_DETECTION_KWARGS = {
    "min_sigma": 1,
    "max_sigma": 10, 
    "sigma_ratio": 5.0,
    "threshold_rel": 0.2,
    "threshold": None,
    "overlap": 0.5
}
# minimum amount of blobs for streak line to be valid
STREAK_LINE_MIN_BLOBS = 3
STREAK_LINE_WEIGHT_SIGMOID_STEEPNESS = 50
# arguments for RANSAC Linear regression on the blobs
STREAK_LINE_RANSAC_KWARGS = {
    "residual_threshold": 10,
    "min_samples": 0.5,
    "max_trials": 100,
    "random_state": RANDOM_SEED
}
# minimum number of inliers resulting from RANSAC
STREAK_LINE_MIN_INLIERS = 4

# Maximum allowable angle difference in degrees
SIMILAR_LINES_MAX_ANGLE_DIFFERENCE = 15
# Maximum midpoint distance as a fraction of longer streak length
SIMILAR_LINES_MAX_MIDPOINT_DISTANCE_RATIO = 0.25
# Minimum ratio of shorter streak length to longer streak length
SIMILAR_LINES_MIN_LENGTH_RATIO = 0.75

# Max angle difference per offset
SAME_TRAJECTORY_MAX_ANGLE_DIFFERENCE = 25
# Max midpoint distance per offset
SAME_TRAJECTORY_MAX_PARALLEL_DISTANCE = 1000
# Max projected distance per offset
SAME_TRAJECTORY_MAX_PERPENDICULAR_DISTANCE = 200
