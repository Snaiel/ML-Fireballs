import matplotlib.pyplot as plt
import pandas as pd
from skimage import io
from pathlib import Path
from object_detection.dataset import GFO_JPEGS, GFO_PICKINGS


def show_points(fireball_name: str) -> None:
    # Load the image using skimage
    image = io.imread(Path(GFO_JPEGS, fireball_name + ".thumb.jpg"))

    # Plot the image using matplotlib
    plt.imshow(image)
    plt.axis('off')  # Hide axes for better image display

    # Read the CSV file using pandas
    data = pd.read_csv(Path(GFO_PICKINGS, fireball_name + ".csv"))

    # Assume the CSV has columns named 'x' and 'y'
    x_coords = data['x_image_thumb']
    y_coords = data['y_image_thumb']

    # Plot a dot on the image for every row in the data
    plt.scatter(x_coords, y_coords, c='red', s=16)  # c for color, s for size of the dots

    # Calculate the bounding box
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()

    # Create a rectangle patch around the points
    width = x_max - x_min
    height = y_max - y_min
    rect = plt.Rectangle((x_min - 0.05 * width, y_min - 0.1 * height), width * 1.1, height * 1.2, linewidth=3, edgecolor='red', facecolor='none')
    plt.gca().add_patch(rect)

    # Show the final plot with the image, points, and rectangle
    plt.title(fireball_name)
    plt.show()


def main():
    fireball_name = "03_2021-02-08_093629_K_DSC_0169"
    show_points(fireball_name)


if __name__ == "__main__":
    main()
