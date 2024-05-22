import matplotlib.pyplot as plt
import pandas as pd
from skimage import io
from pathlib import Path

fireball_name = "005_2017-07-30_134712_E_DSC_0445"

file_path = Path(__file__)
root_folder = file_path.parents[2]

GFO_DATASET_FOLDER = Path(root_folder, "data", "GFO_fireball_object_detection_training_set")

# Load the image using skimage
image = io.imread(Path(GFO_DATASET_FOLDER, "jpegs", fireball_name + ".thumb.jpg"))

# Plot the image using matplotlib
plt.imshow(image)
plt.axis('off')  # Hide axes for better image display

# Read the CSV file using pandas
data = pd.read_csv(Path(GFO_DATASET_FOLDER, "point_pickings_csvs", fireball_name + ".csv"))

# Assume the CSV has columns named 'x' and 'y'
x_coords = data['x_image_thumb']
y_coords = data['y_image_thumb']

# Plot a dot on the image for every row in the data
plt.scatter(x_coords, y_coords, c='red', s=10)  # c for color, s for size of the dots

# Show the final plot with the image and points
plt.show()