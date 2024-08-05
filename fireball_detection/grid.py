import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Load the image
image = mpimg.imread("../data/GFO_fireball_object_detection_training_set/jpegs/25_2016-08-22_202428_S_DSC_2787.thumb.jpg")

# Define grid size (400x400 pixels)
cell_size = 400

# Calculate the number of rows and columns, including excess grids
rows = (image.shape[0] + cell_size - 1) // cell_size  # Round up to include excess grids
cols = (image.shape[1] + cell_size - 1) // cell_size  # Round up to include excess grids

# Create a figure and axis
fig, ax = plt.subplots()
ax.imshow(image)

# Draw the grid
for row in range(rows + 1):
    ax.axhline(y=row * cell_size + (cell_size / 2), color='white', linestyle='-')
for col in range(cols + 1):
    ax.axvline(x=col * cell_size + (cell_size / 2), color='white', linestyle='-')

# Number the cells
for row in range(rows):
    for col in range(cols):
        cell_number = row * cols + col
        ax.text(col * cell_size + (cell_size / 2) + cell_size / 3, row * cell_size + (cell_size / 2) + cell_size / 4, 
                f"({row + 0.5}, {col + 0.5})", color='white', fontsize=6, 
                ha='center', va='center')

# Remove axes
ax.axis('off')
plt.tight_layout()
# Show the image with the grid
plt.show()