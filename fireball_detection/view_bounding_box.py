import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from skimage import io
from pathlib import Path
from point_pickings_to_bounding_boxes import get_yolov8_label_from_point_pickings_csv, IMAGE_DIM

fireball_title = "03_2016-07-28_043558_K_DSC_8287"

file_path = Path(__file__)
parent_folder = file_path.parents[0]

GFO_DATASET_FOLDER = "GFO_fireball_object_detection_training_set"

fireball_image_path = Path(parent_folder, GFO_DATASET_FOLDER, "jpegs", fireball_title + ".thumb.jpg")
point_pickings_path = Path(parent_folder, GFO_DATASET_FOLDER, "point_pickings_csvs", fireball_title + ".csv")

bounding_box = get_yolov8_label_from_point_pickings_csv(point_pickings_path)

# Load the image using skimage
image = io.imread(fireball_image_path)

# Display the image using matplotlib
fig, ax = plt.subplots()
ax.imshow(image)

# Define the rectangle parameters: (x, y, width, height)
c_x, c_y, rect_width, rect_height = bounding_box
c_x *= IMAGE_DIM[0] 
c_y *= IMAGE_DIM[1] 
rect_width *= IMAGE_DIM[0] 
rect_height *= IMAGE_DIM[1]

rect_x = c_x - rect_width / 2
rect_y = c_y - rect_height / 2

# Create a rectangle patch
rect = Rectangle((rect_x, rect_y), rect_width, rect_height,
                 linewidth=2, edgecolor='r', facecolor='none')

# Add the rectangle to the plot
ax.add_patch(rect)

# Show the plot with the image and rectangle
plt.show()