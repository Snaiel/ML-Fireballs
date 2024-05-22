from math import sqrt
import skimage as ski
from skimage import data
from skimage.feature import blob_dog

import matplotlib.pyplot as plt
from pathlib import Path

file_path = Path(__file__)
image_path = Path(file_path.parents[1], 'fireball_images', 'cropped', '051_2021-12-11_182730_E_DSC_0884-G_cropped.jpeg')

image = ski.io.imread(image_path)

image_gray = image

blobs_dog = blob_dog(image_gray, max_sigma=20, threshold=.025)
blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)

fig, ax = plt.subplots(figsize=(6, 6))

ax.set_title('Difference of Gaussian')
ax.imshow(image, cmap='gray')

for blob in blobs_dog:
    y, x, r = blob
    c = plt.Circle((x, y), r, color='lime', linewidth=1, fill=False)
    ax.add_patch(c)

plt.axis('off')
plt.tight_layout()
plt.show()