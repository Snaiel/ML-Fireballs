import matplotlib.pyplot as plt
from skimage import io
from pathlib import Path

file_path = Path(__file__)
image_path = Path(file_path.parents[1], 'dfn_highlights/051_Kanandah/051_2021-12-11_182730_E_DSC_0884-G.jpeg')

# Load the image using skimage
image = io.imread(image_path)

# Plot the image using Matplotlib
plt.imshow(image, cmap='gray', vmin=80, vmax=255)
plt.axis('off')  # Turn off axis
plt.show()