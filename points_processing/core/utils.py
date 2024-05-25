import skimage
import numpy as np

def make_image_landscape(image: np.ndarray) -> np.ndarray:
    """
        Make sure image is landscape. Ensures the regression
        curve is able to properly fit the fireball and makes
        retrieving the sequence based on x coordinate clearer

        ### Parameters
        | Name   | Type          | Description                                |
        |--------|---------------|--------------------------------------------|
        | image  | np.ndarray    | The input image to be transformed.         |

        ### Returns
        | Type          | Description                                     |
        |---------------|-------------------------------------------------|
        | np.ndarray    | The transformed image in landscape orientation. |
    """
    height, width = image.shape[:2]
    if width < height:
        image = skimage.transform.rotate(image, angle=90, resize=True)
    return image