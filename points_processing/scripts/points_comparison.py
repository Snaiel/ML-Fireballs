import time
from math import sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from fireball_point_picker import FireballPointPicker
from skimage import io, transform

current_fireball = "071_CAO_RASC"

dfn_highlights_croppings = {
    "025_Elginfield": ((4900, 1150), (5600, 2800)),
    "044_Vermilion": ((1050, 3040), (2250, 3690)),
    "051_Kanandah": ((2500, 370), (3200, 500)),
    "071_CAO_RASC": ((4950, 2220), (6300, 2430))
}

current_fireball_croppings = dfn_highlights_croppings[current_fireball]

file_path = Path(__file__)
image_path = Path(file_path.parents[1], 'dfn_highlights/071_CAO_RASC/071_2021-12-14_032259_E_DSC_0611-G.jpeg')

image = io.imread(image_path)
cropped_image = image[
    current_fireball_croppings[0][1]:current_fireball_croppings[1][1],
    current_fireball_croppings[0][0]:current_fireball_croppings[1][0]
]

scale_factor = 4

print(cropped_image.shape)

scaled_image = transform.resize(
    cropped_image,
    (cropped_image.shape[0] * scale_factor, cropped_image.shape[1] * scale_factor),
    preserve_range=True
)

start_time = time.time()

fireball_point_picker = FireballPointPicker(scaled_image)

end_time = time.time()
execution_time = end_time - start_time
print("Execution time:", execution_time)

print("Result")
points = fireball_point_picker.fireball_points
for point in points:
    if fireball_point_picker.rotated:
        point[0], point[1] = point[1], point[0]
        point[0] = scaled_image.shape[1] - point[0]
    point[0] = (point[0] / scale_factor) + current_fireball_croppings[0][0]
    point[1] = (point[1] / scale_factor) + current_fireball_croppings[0][1]
    print(point)

fireball_point_picker.show_plot()

pd.set_option('display.max_columns', None)


df = pd.read_csv(Path(file_path.parents[1], "dfn_highlights/071_CAO_RASC/071_2021-12-14_032259_E_DSC_0611-G_DN211214_03_2024-02-20_114318_sophie_nocomment.csv"))
df = df.iloc[:, [0, 1, 4, 6]]

df = df.rename(columns={
    "x_image": "manual_x",
    "y_image": "manual_y",
    "de_bruijn_sequence_element_index": "de_bruijn_index"
})

df.insert(2, 'auto_x', None)
df.insert(3, 'auto_y', None)
df.insert(6, 'offset_x', None)
df.insert(7, 'offset_y', None)
df.insert(8, 'distance', None)

print(df)

offsets = []

df_i = 0


for i, point in enumerate(points):

    while df_i < len(df) - 1 and point[2] > df.iloc[df_i, 4]:
        df_i += 1

    if point[3] == '0':
        df.iloc[df_i, 2] = point[0]
        df.iloc[df_i, 3] = point[1]
        df.iloc[df_i, 6] = point[0] - df.iloc[df_i, 0]
        df.iloc[df_i, 7] = point[1] - df.iloc[df_i, 1]
        df.iloc[df_i, 8] = sqrt((point[0] - df.iloc[df_i, 0])**2 + (point[1] - df.iloc[df_i, 1])**2)
    else:
        closest_i = 0
        closest_d = 1000
        f_df = df[df['de_bruijn_index'] == point[2]]
        
        for index, row in f_df.iterrows():
            d = sqrt((point[0] - row[0])**2 + (point[1] - row[1])**2)
            if d < closest_d:
                closest_i = index
                closest_d = d

        df.iloc[closest_i, 2] = point[0]
        df.iloc[closest_i, 3] = point[1]
        df.iloc[closest_i, 6] = point[0] - df.iloc[closest_i, 0]
        df.iloc[closest_i, 7] = point[1] - df.iloc[closest_i, 1]
        df.iloc[closest_i, 8] = closest_d


print(df.to_string(index=False))
print("Average Distance:", df['distance'].mean())


# Assuming df is your DataFrame
plt.figure(figsize=(8, 6))

# Plot the offset values
plt.scatter(df['offset_x'], df['offset_y'], color='black', label='Offset')

# Plot a red dot at the origin
plt.scatter(0, 0, color='red', label='Origin')

# Get the maximum absolute offset value
max_offset = max(abs(df['offset_x'].max()), abs(df['offset_x'].min()), abs(df['offset_y'].max()), abs(df['offset_y'].min()))

# Set the limits of the plot axes symmetrically around the origin
plt.xlim(-max_offset, max_offset)
plt.ylim(-max_offset, max_offset)

plt.xlabel('Offset X')
plt.ylabel('Offset Y')
plt.title('Scatter Plot of Offset Values with Origin at Center: ' + current_fireball)
plt.legend()
plt.grid(True)
plt.show()