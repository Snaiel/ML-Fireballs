import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
import pandas as pd

# Load the image
image_path = "../data/GFO_fireball_object_detection_training_set/jpegs/31_2018-01-09_104328_S_DSC_7090.thumb.jpg"
image = mpimg.imread(image_path)
img_h, img_w, _ = image.shape

print(img_h, img_w)


square_size = 400


def start_points(size, split_size, overlap=0):
    points = [0]
    stride = int(split_size * (1-overlap))
    counter = 1
    while True:
        pt = stride * counter
        if pt + split_size >= size:
            if split_size == size:
                break
            points.append(size - split_size)
            break
        else:
            points.append(pt)
        counter += 1
    return points


X_points = start_points(img_w, square_size, 0.5)
Y_points = start_points(img_h, square_size, 0.5)

points = [(y, x) for y in Y_points for x in X_points]

all_coordinates = pd.DataFrame(points, columns=['y', 'x']).astype(float)


discarded = pd.read_csv("discard_overlap.csv")

discarded_coordinates = discarded.copy()
discarded_coordinates["y"] = discarded_coordinates["y"].map(lambda y: img_h - square_size if y == -1 else y * square_size)
discarded_coordinates["x"] = discarded_coordinates["x"].map(lambda x: img_w - square_size if x == -1 else x * square_size)



# Step 1: Perform an outer join to include all rows
merged_df = pd.merge(all_coordinates, discarded_coordinates, on=['y', 'x'], how='outer', indicator=True)

# Step 2: Identify common rows
common_rows = merged_df[merged_df['_merge'] == 'both']

# Step 3: Remove common rows from the merged DataFrame
result_df = merged_df[merged_df['_merge'] != 'both'].drop(columns=['_merge'])


print(result_df)


# Create a figure and axis
fig, ax = plt.subplots(figsize=(12, 12))

# Display the image
ax.imshow(image)

# Draw a red square at each (x, y) coordinate
for index, row in result_df.iterrows():
    y = row[0]
    x = row[1]
    square = patches.Rectangle((x, y), square_size, square_size, linewidth=1, edgecolor='lime', facecolor='none')
    ax.add_patch(square)

# Remove axes for a cleaner look
ax.axis('off')

# Adjust layout and display the image
plt.tight_layout()

if __name__ == "__main__":
    plt.show()