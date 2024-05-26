import matplotlib.pyplot as plt
from skimage import io, transform
from pathlib import Path

def downscale_image(image_path, max_dim=1280):
    # Load the image
    image = io.imread(image_path)
    
    # Get the dimensions of the image
    height, width = image.shape[:2]
    
    # Determine the scaling factor
    if height > width:
        scale_factor = max_dim / height
    else:
        scale_factor = max_dim / width
    
    # Calculate the new dimensions
    new_height = int(height * scale_factor)
    new_width = int(width * scale_factor)
    
    # Resize the image
    downscaled_image = transform.resize(image, (new_height, new_width), anti_aliasing=True)
    
    print(new_width, new_height)

    return downscaled_image

def display_image(image):
    # Display the image using matplotlib
    plt.imshow(image)
    plt.axis('off')  # Hide axis
    plt.show()

image_path = Path("../data/GFO_fireball_object_detection_training_set/jpegs/62_2016-05-12_180528_S_DSC_3102.thumb.jpg")
downscaled_image = downscale_image(image_path)
display_image(downscaled_image)