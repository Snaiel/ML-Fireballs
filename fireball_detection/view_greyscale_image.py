import cv2
from skimage.color import rgb2gray
import matplotlib.pyplot as plt

# Read the image using OpenCV
image = cv2.imread('yolov8_fireball_dataset/images/test/03_2020-05-16_071327_K_DSC_6175.jpg')

# Convert the image to grayscale using scikit-image
gray_image = rgb2gray(image)

# Plot the original and grayscale images using Matplotlib
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(gray_image, cmap='gray')
plt.title('Grayscale Image')
plt.axis('off')

plt.show()