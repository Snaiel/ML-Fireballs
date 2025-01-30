import cv2
import numpy as np


def difference_images(image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
    """
    image1 - image2
    """

    if len(image1.shape) == 3:
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    if len(image2.shape) == 3:
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

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
    matrix, mask = cv2.estimateAffinePartial2D(points2, points1, method=cv2.RANSAC)
    transformation_magnitude = np.linalg.norm(matrix - np.eye(2, 3))
    
    blurred_image1 = cv2.GaussianBlur(image1, (11, 11), 0)
    blurred_image2 = cv2.GaussianBlur(image2, (11, 11), 0)
    unaligned_difference = cv2.subtract(blurred_image1, blurred_image2)

    output_image = unaligned_difference

    if transformation_magnitude > 10:
        return output_image
    
    # Apply the transformation to align the images
    height, width = image1.shape
    aligned_image2 = cv2.warpAffine(image2, matrix, (width, height))

    blurred_aligned_image2 = cv2.GaussianBlur(aligned_image2, (11, 11), 0)
    aligned_difference = cv2.subtract(blurred_image1, blurred_aligned_image2)

    # Convert images to float32 for safe computation
    unaligned_difference = unaligned_difference.astype(np.float32)
    aligned_difference = aligned_difference.astype(np.float32)

    # Calculate weighted average with more weight for the lower value
    output_image = np.where(unaligned_difference < aligned_difference,
        unaligned_difference, # use pixel from unaligned_difference since lower brightness
        aligned_difference # use pixel from aligned_difference since lower brightness
    )  

    # Convert back to uint8
    output_image = np.clip(output_image, 0, 255).astype(np.uint8)

    return output_image