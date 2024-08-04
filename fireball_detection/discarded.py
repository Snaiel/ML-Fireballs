import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
import pandas as pd

# Load the image
image_path = "../data/GFO_fireball_object_detection_training_set/jpegs/25_2016-08-22_202428_S_DSC_2787.thumb.jpg"
image = mpimg.imread(image_path)
img_h, img_w, _ = image.shape
print(img_h, img_w)

# Load the CSV file with coordinates (assumes the CSV is in the same directory or provide full path)
csv_path = "discard_overlap.csv"
coordinates = pd.read_csv(csv_path)

# Create a figure and axis
fig, ax = plt.subplots(figsize=(12, 12))

# Display the image
ax.imshow(image)

# Define the size of the square
square_size = 400

# Draw a red square at each (x, y) coordinate
print(coordinates)
for index, row in coordinates.iterrows():
    y = row[0]
    x = row[1]
    y_pos = img_h - square_size if y == -1 else y * 400
    x_pos = img_w - square_size if x == -1 else x * 400
    print(x_pos, y_pos)
    square = patches.Rectangle((x_pos, y_pos), square_size, square_size, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(square)

# Remove axes for a cleaner look
ax.axis('off')

# Adjust layout and display the image
plt.tight_layout()
plt.show()