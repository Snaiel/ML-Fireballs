import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path

# Directory paths for the two folders
folder1 = Path(Path(__file__).parents[1], "data", "kfold_fireball_detection", "fold0", "0border", "preds")
folder2 = Path(Path(__file__).parents[1], "data", "kfold_fireball_detection", "fold0", "8border", "preds")

# Get the list of image files in both folders
images1 = sorted([f for f in os.listdir(folder1) if f.endswith(('png', 'jpg', 'jpeg'))])
images2 = sorted([f for f in os.listdir(folder2) if f.endswith(('png', 'jpg', 'jpeg'))])

# Ensure that both folders contain the same image files
assert images1 == images2, "The two folders do not contain the same image files"

# Plot images side by side
for img_name in images1:
    img1_path = os.path.join(folder1, img_name)
    img2_path = os.path.join(folder2, img_name)
    
    img1 = mpimg.imread(img1_path)
    img2 = mpimg.imread(img2_path)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img1)
    axes[0].set_title(f'{img_name} 0 border')
    axes[0].axis('off')
    axes[1].imshow(img2)
    axes[1].set_title(f'{img_name} 8 border')
    axes[1].axis('off')
    
    plt.show()
