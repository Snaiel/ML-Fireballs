import cv2
import matplotlib.pyplot as plt
import numpy as np


# Load the images
image1_path = "data/val_fireball_detection/2013-10-28/images/08_2013-10-28_173929_DSC_0836.NEF.thumb.jpg"
image2_path = "data/val_fireball_detection/2013-10-28/images/08_2013-10-28_173959_DSC_0837.NEF.thumb.jpg"

image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

# Step 1: Detect keypoints and descriptors using ORB
orb = cv2.ORB.create()
keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
keypoints2, descriptors2 = orb.detectAndCompute(image2, None)

# Step 2: Match keypoints using BFMatcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(descriptors1, descriptors2)
matches = sorted(matches, key=lambda x: x.distance)

# Step 3: Extract matched keypoints
points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

# Step 4: Estimate homography (can handle rotation and scaling)
matrix, mask = cv2.estimateAffinePartial2D(points1, points2, method=cv2.RANSAC)

# Step 5: Apply the transformation to align the images
height, width = image1.shape
aligned_image1 = cv2.warpAffine(image1, matrix, (width, height))

# Step 6: Blur images 
blurred_image1 = cv2.GaussianBlur(image1, (11, 11), 0)
blurred_aligned_image1 = cv2.GaussianBlur(aligned_image1, (11, 11), 0)
blurred_image2 = cv2.GaussianBlur(image2, (11, 11), 0)

# Step 7: Compute the original difference
original_difference = cv2.subtract(blurred_image2, blurred_image1)

# Step 8: Compute the difference between blurred images
blurred_difference = cv2.subtract(blurred_image2, blurred_aligned_image1)

# Plot points1 and points2
fig_points, axs_points = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)

axs_points[0].imshow(image1, cmap='gray')
axs_points[0].scatter(points1[:, 0, 0], points1[:, 0, 1], c='red', s=5)
axs_points[0].set_title("Keypoints from Image 1")
axs_points[0].axis('off')

axs_points[1].imshow(image2, cmap='gray')
axs_points[1].scatter(points2[:, 0, 0], points2[:, 0, 1], c='red', s=5)
axs_points[1].set_title("Keypoints from Image 2")
axs_points[1].axis('off')

plt.tight_layout()
plt.show()

# Plot Images and Differencing
fig, axs = plt.subplots(3, 3, figsize=(10, 6), sharex=True, sharey=True)

axs[0, 0].imshow(image1, cmap='gray')
axs[0, 0].set_title("Image 1")
axs[0, 0].axis("off")

axs[0, 1].imshow(aligned_image1, cmap='gray')
axs[0, 1].set_title("Aligned Image 1")
axs[0, 1].axis("off")

axs[0, 2].imshow(image2, cmap='gray')
axs[0, 2].set_title("Image 2")
axs[0, 2].axis("off")

axs[1, 0].imshow(blurred_image1, cmap='gray')
axs[1, 0].set_title("Blurred Image 1")
axs[1, 0].axis("off")

axs[1, 1].imshow(blurred_aligned_image1, cmap='gray')
axs[1, 1].set_title("Blurred Aligned Image 1")
axs[1, 1].axis("off")

axs[1, 2].imshow(blurred_image2, cmap='gray')
axs[1, 2].set_title("Blurred Image 2")
axs[1, 2].axis("off")

axs[2, 0].imshow(image2, cmap='gray')
axs[2, 0].set_title("Original Image 2")
axs[2, 0].axis("off")

axs[2, 1].imshow(original_difference, cmap='gray')
axs[2, 1].set_title("Unaligned Blurred Difference")
axs[2, 1].axis("off")

axs[2, 2].imshow(blurred_difference, cmap='gray')
axs[2, 2].set_title("Aligned Blurred Difference")
axs[2, 2].axis("off")


plt.tight_layout()
plt.show()

# Compare Differenced Images
fig_diff, axs_diff = plt.subplots(1, 3, figsize=(8, 4), sharex=True, sharey=True)

axs_diff[0].imshow(image2, cmap='gray')
axs_diff[0].set_title("Original Image 2")
axs_diff[0].axis("off")

axs_diff[1].imshow(original_difference, cmap='gray')
axs_diff[1].set_title("Unaligned Blurred Difference")
axs_diff[1].axis("off")

axs_diff[2].imshow(blurred_difference, cmap='gray')
axs_diff[2].set_title("Aligned Blurred Difference")
axs_diff[2].axis("off")

plt.tight_layout()
plt.show()