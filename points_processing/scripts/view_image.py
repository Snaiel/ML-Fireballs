from pathlib import Path

import matplotlib.pyplot as plt
from skimage import io


image_path = Path(Path(__file__).parents[2], 'data', 'dfn_highlights/051_Kanandah/051_2021-12-11_182730_E_DSC_0884-G.jpeg')

# Load the image using skimage
image = io.imread(image_path)

# Plot the image using Matplotlib
plt.imshow(image, cmap='gray', vmin=80, vmax=255)
plt.axis('off')  # Turn off axis
plt.show()