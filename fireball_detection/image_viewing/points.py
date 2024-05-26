import matplotlib.pyplot as plt
import pandas as pd
from skimage import io
from pathlib import Path
from dataset import GFO_JPEGS, GFO_PICKINGS


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
    plt.scatter(x_coords, y_coords, c='red', s=10)  # c for color, s for size of the dots

    # Show the final plot with the image and points
    plt.title(fireball_name)
    plt.show()


def main():
    fireball_name = "029_2019-10-01_124829_E_DSC_0258"
    show_points(fireball_name)


if __name__ == "__main__":
    main()