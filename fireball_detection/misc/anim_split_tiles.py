import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation

from fireball_detection import SQUARE_SIZE
from fireball_detection.tiling.included import retrieve_included_coordinates
from skimage import io

from object_detection.dataset import GFO_JPEGS
from pathlib import Path

# Sample list of rectangle coordinates (top-left corner (x, y), width, height)
coordinates = sorted(retrieve_included_coordinates(), key=lambda x: x[1])

# Load the background image
image = io.imread(Path(GFO_JPEGS, "04_2023-06-25_114027_K_DSC_5220.thumb.jpg"))

# Create the plot
fig, ax = plt.subplots()
ax.imshow(image, cmap='gray')

ax.axis('off')
fig.tight_layout()
fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

# Initial rectangle
rect = patches.Rectangle((coordinates[0][0], coordinates[0][1]), 
                         SQUARE_SIZE, SQUARE_SIZE, 
                         linewidth=2, edgecolor='r', facecolor='none')

ax.add_patch(rect)

# Update function for animation
def update(frame):
    x, y = coordinates[frame]
    rect.set_xy((x, y))
    return [rect]

# Create animation
ani = FuncAnimation(fig, update, frames=len(coordinates), interval=200, blit=True, repeat=True)
ani.save(
    "split_tiles.gif",
    progress_callback = lambda i, n: print(f'Saving frame {i}/{n}'),
    savefig_kwargs={'bbox_inches': 'tight', 'pad_inches': 0}
)

# Show the plot
plt.show()
