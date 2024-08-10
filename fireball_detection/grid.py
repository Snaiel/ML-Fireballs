import matplotlib.pyplot as plt
from skimage import io
from pathlib import Path

# Load the image
image = io.imread(Path(Path(__file__).parents[1], "data/GFO_fireball_object_detection_training_set/jpegs/03_2019-06-03_073527_K_DSC_3233.thumb.jpg"))

# Define grid size (400x400 pixels)
cell_size = 400

# Calculate the number of rows and columns, including excess grids
rows = (image.shape[0] + cell_size - 1) // cell_size  # Round up to include excess grids
cols = (image.shape[1] + cell_size - 1) // cell_size  # Round up to include excess grids

# Create a figure and axis
fig, ax = plt.subplots()
ax.imshow(image)

# Draw the grid
for row in range(1, rows):
    ax.axhline(y=row * cell_size, color='white', linestyle='-', linewidth=1)
for col in range(1, cols):
    ax.axvline(x=col * cell_size, color='white', linestyle='-', linewidth=1)

# Number the cells
for row in range(rows):
    for col in range(cols):
        cell_number = row * cols + col
        ax.text(col * cell_size + (cell_size / 2) + cell_size / 3, row * cell_size + (cell_size / 2) + cell_size / 4, 
                f"({row}, {col})", color='white', fontsize=6, 
                ha='center', va='center')

ax.axis('off')
fig.tight_layout()
fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
# fig.savefig("yer.png", bbox_inches='tight', pad_inches=0, dpi=600)
plt.show()