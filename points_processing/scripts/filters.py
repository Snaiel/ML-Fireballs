import matplotlib.pyplot as plt
from skimage import io, filters

image = io.imread("../data/multi_tiered_test_data/500to1000at1280/images/test/04_2020-10-22_062427_K_DSC_7222.jpg", True)

fig, ax = filters.try_all_threshold(image, figsize=(10, 10), verbose=True)

fig2, ax2 = plt.subplots()
dog = filters.difference_of_gaussians(image, 1.5)
ax2.set_title("Difference of Gaussians")
ax2.imshow(dog)

plt.tight_layout()
plt.show()
