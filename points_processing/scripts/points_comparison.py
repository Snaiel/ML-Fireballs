import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from pipelines.full_auto import retrieve_fireball
from skimage import io, transform

## Retrieve necessary files
fireball_name = "071_CAO_RASC"

file_path = Path(__file__)
dfn_highlights_folder = Path(file_path.parents[2], "data", "dfn_highlights")
for folder in os.listdir(dfn_highlights_folder):
    if folder == fireball_name:
        for file in os.listdir(Path(dfn_highlights_folder, folder)):
            if file.endswith("-G.jpeg"):
                image_path = Path(dfn_highlights_folder, folder, file)
            if file.endswith(".csv"):
                csv_path = Path(dfn_highlights_folder, folder, file)


## Fireball Cropping
dfn_highlights_croppings = {
    "025_Elginfield": ((4900, 1150), (5600, 2800)),
    "044_Vermilion": ((1050, 3040), (2250, 3690)),
    "051_Kanandah": ((2500, 370), (3200, 500)),
    "071_CAO_RASC": ((4950, 2220), (6300, 2430))
}

current_fireball_croppings = dfn_highlights_croppings[fireball_name]

image = io.imread(image_path)
cropped_image = image[
    current_fireball_croppings[0][1]:current_fireball_croppings[1][1],
    current_fireball_croppings[0][0]:current_fireball_croppings[1][0]
]
print(cropped_image.shape)


## Upscaling image
scale_factor = 4
scaled_image = transform.resize(
    cropped_image,
    (cropped_image.shape[0] * scale_factor, cropped_image.shape[1] * scale_factor),
    preserve_range=True
)

start_time = time.time()

## Picking points
fireball = retrieve_fireball(scaled_image, min_radius=12, threshold=4)

end_time = time.time()
execution_time = end_time - start_time
print("Execution time:", execution_time)

print("Result")
points = fireball.fireball_points

# Create dataframe from auto pickings
auto_df = pd.DataFrame(columns=['auto_x', 'auto_y', 'de_bruijn_index', 'zero_or_one'])

for point in points:
    x = (point.x / scale_factor) + current_fireball_croppings[0][0]
    y = (point.y / scale_factor) + current_fireball_croppings[0][1]
    auto_df = pd.concat(
        [auto_df, pd.DataFrame([[x, y, point.de_bruijn_pos, point.de_bruijn_val]],
        columns=['auto_x', 'auto_y', 'de_bruijn_index', 'zero_or_one'])],
        ignore_index=True
    )

print(auto_df.to_string(index=False))


## Read manual pickings
pd.set_option('display.max_columns', None)
manual_df = pd.read_csv(csv_path)
manual_df = manual_df.iloc[:, [0, 1, 4, 6]]

manual_df = manual_df.rename(columns={
    "x_image": "manual_x",
    "y_image": "manual_y",
    "de_bruijn_sequence_element_index": "de_bruijn_index"
})

print(manual_df.to_string(index=False))



## Visualise Fireball
# Create a figure and axis
fig, ax = plt.subplots()


# Plot the image
ax.imshow(fireball.image, cmap='gray', aspect='equal')


# Plot Blobs
for node in fireball.fireball_blobs:
    x, y, r = node
    # blob radius
    outer_circle = patches.Circle((x, y), r, color='lime', linewidth=2, fill=False)
    # Add the patches to the axis
    ax.add_patch(outer_circle)


# Plot Auto Pickings
for index, row in auto_df.iterrows():
    x = row['auto_x']
    y = row['auto_y']

    x = (x - current_fireball_croppings[0][0]) * scale_factor
    y = (y - current_fireball_croppings[0][1]) * scale_factor

    ax.add_patch(
        patches.Circle((x, y), 0.5, color='lime', fill=True)
    )

    ax.text(x, y + 50, int(row['zero_or_one']), color="pink").set_clip_on(True)
    ax.text(x, y + 100, int(row['de_bruijn_index']), color="pink").set_clip_on(True)


# Plot Manual Pickings
for index, row in manual_df.iterrows():
    x = row['manual_x']
    y = row['manual_y']

    x = (x - current_fireball_croppings[0][0]) * scale_factor
    y = (y - current_fireball_croppings[0][1]) * scale_factor

    if fireball.rotated:
        x, y = y, x
        y = fireball.image.shape[0] - y

    ax.add_patch(
        patches.Circle((x, y), 0.5, color='red', fill=True)
    )

    ax.text(x, y - 30, int(row['zero_or_one']), color="white").set_clip_on(True)
    ax.text(x, y - 80, int(row['de_bruijn_index']), color="white").set_clip_on(True)



# Display the plot
plt.show()