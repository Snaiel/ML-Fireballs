
import os
import re
from math import sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import skimage as ski
from matplotlib.widgets import RadioButtons
from skimage.feature import blob_dog
from sklearn.linear_model import RANSACRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

file_path = Path(__file__)
dfn_highlights_folder = Path(file_path.parents[1], 'dfn_highlights')

# Define regular expression pattern to capture thumb, numbers, and tile
pattern = re.compile(r'^(.*?)\.thumb\.(\d+)\.(\d+)\.tile', re.IGNORECASE)

for event_folder in os.listdir(dfn_highlights_folder):
    print(event_folder)

    match = None
    for file in os.listdir(os.path.join(dfn_highlights_folder, event_folder)):
        match = re.search(pattern, file)
        if match:
            break
    
    if match is None:
        continue

    event_name = match.group(1)
    x = match.group(2)
    y = match.group(3)
    print(event_name, x, y)

    image = ski.io.imread(os.path.join(dfn_highlights_folder, event_folder, event_name + "-G.jpeg"))

    crop_size = 2500  # Size of the square crop

    # Calculate the top-left corner coordinates for cropping
    tl_x_coord = int(x) - crop_size // 2
    tl_x_coord = tl_x_coord if tl_x_coord >= 0 else 1
    tl_y_coord = int(y) - crop_size // 2
    tl_y_coord = tl_y_coord if tl_y_coord >= 0 else 1

    br_x_coord = tl_x_coord + crop_size
    br_x_coord = br_x_coord if br_x_coord <= image.shape[1] else image.shape[1]
    br_y_coord = tl_y_coord + crop_size
    br_y_coord = br_y_coord if br_y_coord <= image.shape[0] else image.shape[0]

    # Perform cropping
    cropped_image = image[tl_y_coord:br_y_coord, tl_x_coord:br_x_coord]

    # Return (y-coord, x-coord, radius)
    blobs_dog = blob_dog(cropped_image, max_sigma=20, threshold=.025)
    # Compute radii in the 3rd column.
    blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)

    # Extract x and y coordinates from blobs_dog
    y_coords = blobs_dog[:, 0]
    x_coords = blobs_dog[:, 1]

    # Fit a polynomial curve using RANSAC
    model = make_pipeline(PolynomialFeatures(2), RANSACRegressor(residual_threshold=10, max_trials=100))
    model.fit(x_coords.reshape(-1, 1), y_coords)

    # Predict y values using the fitted model
    x_values = np.linspace(0, cropped_image.shape[1], 1000)
    y_values = model.predict(x_values.reshape(-1, 1))

    # Plot the original image with fitted curve
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title('Difference of Gaussian, Polynimal 2 Curve Using RANSAC')
    ax.imshow(cropped_image, cmap='gray')

    plt.tight_layout()
    plt.axis('off')

    # Check if the predicted y-values fall within the image dimensions
    valid_indices = np.where((y_values >= 0) & (y_values < cropped_image.shape[0]))

    # Plot the curve only if it stays within the image bounds
    if len(valid_indices[0]) > 0:
        line, = ax.plot(x_values[valid_indices], y_values[valid_indices], color='red', linestyle='-', linewidth=2)

    # Get the inlier indices
    inlier_indices = model.named_steps['ransacregressor'].inlier_mask_

    # Plot pink circles around inlier points
    inlier_circles = []
    for idx in np.where(inlier_indices)[0]:
        blob = blobs_dog[idx]
        y, x, r = blob
        c = plt.Circle((x, y), r, color='pink', linewidth=2, fill=False)
        inlier_circles.append(c)
        ax.add_patch(c)

    # Plot blobs_dog points
    all_blobs = []
    for blob in blobs_dog:
        y, x, r = blob
        c = plt.Circle((x, y), r, color='green', linewidth=2, fill=False)
        all_blobs.append(c)
        ax.add_patch(c)

    # Define radio button click actions
    def toggle_visibility(label):
        line.set_visible(False)
        for circle in inlier_circles:
                circle.set_visible(False)
        for circle in all_blobs:
                circle.set_visible(False)

        if label == 'Blobs':
            for circle in all_blobs:
                circle.set_visible(True)
        elif label == 'Line':
            line.set_visible(True)
        elif label == 'Inliers':
            for circle in inlier_circles:
                circle.set_visible(True)
        
        plt.draw()


    toggle_visibility('None')

    # Add radio buttons
    ax_radio = plt.axes([0.01, 0.5, 0.1, 0.15])
    radio = RadioButtons(ax_radio, ('None', 'Blobs', 'Line', 'Inliers'))
    radio.on_clicked(toggle_visibility)

    # plt.title(event_folder)
    plt.show()