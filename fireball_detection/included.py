from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pandas as pd

from fireball_detection import IMAGE_DIMENSIONS, SQUARE_SIZE


def retrieve_start_points(size, split_size, overlap=0):
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


def retrieve_included_coordinates():
    """
    Retrieve coordinates of tiles

    Returns:
        list[tuple]: A list of tuples where each tuple represents the coordinate (x, y) of a tile

    Note:
        The positions in the `discard_overlap.csv` file use -1 to indicate the last position in that 
        dimension.
    """
    
    X_points = retrieve_start_points(IMAGE_DIMENSIONS[0], SQUARE_SIZE, 0.5)
    Y_points = retrieve_start_points(IMAGE_DIMENSIONS[1], SQUARE_SIZE, 0.5)

    points = [(y, x) for y in Y_points for x in X_points]

    all_coordinates = pd.DataFrame(points, columns=['y', 'x']).astype(int)

    discarded = pd.read_csv(Path(Path(__file__).parent,"discard_overlap.csv"))
    discarded_coordinates = discarded.copy()
    # Turn positions into coordinates
    discarded_coordinates["y"] = discarded_coordinates["y"]\
        .map(lambda y: IMAGE_DIMENSIONS[1] - SQUARE_SIZE if y == -1 else y * SQUARE_SIZE)\
        .astype(int)
    discarded_coordinates["x"] = discarded_coordinates["x"]\
        .map(lambda x: IMAGE_DIMENSIONS[0] - SQUARE_SIZE if x == -1 else x * SQUARE_SIZE)\
        .astype(int)

    # Perform an outer join to include all rows
    merged_df = pd.merge(all_coordinates, discarded_coordinates, on=['y', 'x'], how='outer', indicator=True)
    # Remove common rows from the merged DataFrame
    included_coordinates = merged_df[merged_df['_merge'] != 'both'].drop(columns=['_merge'])

    included_coordinates['x'], included_coordinates['y'] = included_coordinates['y'], included_coordinates['x']

    return list(included_coordinates.itertuples(index=False, name=None))



def main():
    image_path = Path(Path(__file__).parents[1], "data/GFO_fireball_object_detection_training_set/jpegs/03_2020-08-20_070659_K_DSC_6995.thumb.jpg")
    image = mpimg.imread(image_path)

    included_coordinates = retrieve_included_coordinates()

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(image)

    for pos in included_coordinates:
        square = patches.Rectangle(pos, SQUARE_SIZE, SQUARE_SIZE, linewidth=1, edgecolor='lime', facecolor='none')
        ax.add_patch(square)

    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    plt.tight_layout()
    # fig.savefig("bruh.png", bbox_inches='tight', pad_inches=0, dpi=600)
    plt.show()


if __name__ == "__main__":
    main()