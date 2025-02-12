RANDOM_SEED = 0

IMAGE_DIMENSIONS = (7360, 4912)

# Checking image brightness
# median brightness shouldnt be higher than this.
# Chosen from visual inspection of ORB alignment.
MAX_THRESHOLD_MEDIAN_BRIGHTNESS = 165
# the maximum brightness shouldnt be lower than this.
# Chosen from visual inspection of ORB alignment.
MIN_THRESHOLD_MAX_BRIGHTNESS = 40

# Maximum time in seconds between images to allow for image differencing.
# Taken from Towner et al. (2020)
MAX_TIME_DIFFERENCE = 180

# when aligning images, if the transformation is larger than this,
# it has done too much and has not really aligned the stars at this point.
# Chosen by looking at example images and their magnitudes and choosing an upper limit. 
MAX_TRANSFORMATION_MAGNITUDE = 10
# blur size when image differencing.
# Chosen by visual inspection "looks good enough".
GAUSSIAN_BLUR_SIZE = 11

# Tiles.
# Median fireball size lower than this.
# Also used by Towner et al. (2020) 
SQUARE_SIZE = 400
# The minimum number of point pickings in the tile to use it as
# a positive sample in the object detection dataset.
# seemed like a good number for being able to actually see the points.
MIN_POINTS_IN_TILE = 3
# min diagonal length of a detection box
MIN_DIAGONAL_LENGTH = 60
# bounding box padding for creating the object detection dataset
BB_PADDING = 0.05
# minimum dimension size for the object detection dataset bounding box
MIN_BB_DIM_SIZE = 20

# Following are chosen by looking at examples and visual inspection of pixels.
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

# minimum confidence value for detections.
# low confidence to detect streaks then worry about filtering them out.
# detecting satellites is valuable because it provides more chances for same trajectory filtering.
DETECTOR_CONF = 0.2
# IoU used for NMS in detectors
# This isued in the object detection model itself. Standard threshold.
DETECTOR_IOU = 0.5
# border size added to tile before going through model.
# experiments showed a border size of 5 gave the best performance.
TILE_BORDER_SIZE = 5
# The one used by ultralytics i think
TILE_BORDER_COLOUR = (114, 114, 114)

# arguments when blob detection to establish the streak line.
# Chosen from visual inspection of examples.
# sigma ratio seems high but it allows for really subtle blobs to be detected
# which is helpful when detecting points on a line which aren't blobs.
# a high sigma ratio would mean a bigger gaussian which would blur it more
# so i don't know why it results in detecting subtle changes but eh it works.
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
# High separation was better. Even 100 worked well but it felt too 'steep'
# which is not very scientific but eh it fine.
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

# How far to look forwards in images for detections.
# These are guesses on expected movements from some examples.
SAME_TRAJECTORY_MAX_OFFSET = 3
# Max angle difference per offset
SAME_TRAJECTORY_MAX_ANGLE_DIFFERENCE = 25
# Max midpoint distance per offset
SAME_TRAJECTORY_MAX_PARALLEL_DISTANCE = 1000
# Max projected distance per offset
SAME_TRAJECTORY_MAX_PERPENDICULAR_DISTANCE = 200
