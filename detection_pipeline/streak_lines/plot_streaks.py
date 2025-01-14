from math import sqrt

import matplotlib.pyplot as plt
import skimage as ski

from detection_pipeline.streak_lines import StreakLine


def main():
    # original_image = "data/detections_dfn-l0-20151101/dfn-l0-20151101/DFNSMALL48/48_2015-11-01_110129_DSC_8995/48_2015-11-01_110129_DSC_8995.thumb.jpg"

    # streak_images = [
    #     "data/detections_dfn-l0-20151101/dfn-l0-20151101/DFNSMALL48/48_2015-11-01_110129_DSC_8995/48_2015-11-01_110129_DSC_8995_32_1605-1076-1741-1204.differenced.jpg",
    #     "data/detections_dfn-l0-20151101/dfn-l0-20151101/DFNSMALL48/48_2015-11-01_110159_DSC_8996/48_2015-11-01_110159_DSC_8996_67_1764-797-2071-1054.differenced.jpg",
    #     "data/detections_dfn-l0-20151101/dfn-l0-20151101/DFNSMALL48/48_2015-11-01_110229_DSC_8997/48_2015-11-01_110229_DSC_8997_51_2084-586-2368-783.differenced.jpg",
    #     "data/detections_dfn-l0-20151101/dfn-l0-20151101/DFNSMALL48/48_2015-11-01_110259_DSC_8998/48_2015-11-01_110259_DSC_8998_62_2476-329-2858-520.differenced.jpg",
    #     "data/detections_dfn-l0-20151101/dfn-l0-20151101/DFNSMALL48/48_2015-11-01_110329_DSC_8999/48_2015-11-01_110329_DSC_8999_68_2897-174-3275-317.differenced.jpg",
    #     "data/detections_dfn-l0-20151101/dfn-l0-20151101/DFNSMALL48/48_2015-11-01_110359_DSC_9000/48_2015-11-01_110359_DSC_9000_57_3319-91-3607-165.differenced.jpg",
    #     "data/detections_dfn-l0-20151101/dfn-l0-20151101/DFNSMALL48/48_2015-11-01_110429_DSC_9001/48_2015-11-01_110429_DSC_9001_46_3741-24-4039-72.differenced.jpg"
    # ]

    # original_image = "data/detections_dfn-l0-20151101/dfn-l0-20151101/DFNSMALL41/41_2015-11-01_110059_DSC_0205/41_2015-11-01_110059_DSC_0205.thumb.jpg"

    # streak_images = [
    #     "/home/snaiel/Dropbox/Curtin/Year 3/NPSC3000 Research, Leadership and Entrepreneurship in Science 2/Fireballs/ML-Fireballs/data/detections_dfn-l0-20151101/dfn-l0-20151101/DFNSMALL41/41_2015-11-01_110059_DSC_0205/41_2015-11-01_110059_DSC_0205_45_1237-2122-1364-2426.differenced.jpg",
    #     "/home/snaiel/Dropbox/Curtin/Year 3/NPSC3000 Research, Leadership and Entrepreneurship in Science 2/Fireballs/ML-Fireballs/data/detections_dfn-l0-20151101/dfn-l0-20151101/DFNSMALL41/41_2015-11-01_110129_DSC_0206/41_2015-11-01_110129_DSC_0206_37_1367-1748-1551-2088.differenced.jpg",
    #     "/home/snaiel/Dropbox/Curtin/Year 3/NPSC3000 Research, Leadership and Entrepreneurship in Science 2/Fireballs/ML-Fireballs/data/detections_dfn-l0-20151101/dfn-l0-20151101/DFNSMALL41/41_2015-11-01_110159_DSC_0207/41_2015-11-01_110159_DSC_0207_66_1586-1357-1803-1681.differenced.jpg",
    #     "/home/snaiel/Dropbox/Curtin/Year 3/NPSC3000 Research, Leadership and Entrepreneurship in Science 2/Fireballs/ML-Fireballs/data/detections_dfn-l0-20151101/dfn-l0-20151101/DFNSMALL41/41_2015-11-01_110229_DSC_0208/41_2015-11-01_110229_DSC_0208_53_1822-981-2113-1324.differenced.jpg",
    #     "/home/snaiel/Dropbox/Curtin/Year 3/NPSC3000 Research, Leadership and Entrepreneurship in Science 2/Fireballs/ML-Fireballs/data/detections_dfn-l0-20151101/dfn-l0-20151101/DFNSMALL41/41_2015-11-01_110259_DSC_0209/41_2015-11-01_110259_DSC_0209_79_2144-651-2474-964.differenced.jpg",
    #     "/home/snaiel/Dropbox/Curtin/Year 3/NPSC3000 Research, Leadership and Entrepreneurship in Science 2/Fireballs/ML-Fireballs/data/detections_dfn-l0-20151101/dfn-l0-20151101/DFNSMALL41/41_2015-11-01_110329_DSC_0210/41_2015-11-01_110329_DSC_0210_75_2503-386-2842-624.differenced.jpg"
    # ]

    # original_image = "data/detections_dfn-l0-20151101/dfn-l0-20151101/DFNSMALL25/25_2015-11-01_101629_DSC_0035/25_2015-11-01_101629_DSC_0035.thumb.jpg"

    # streak_images = [
    #     "data/detections_dfn-l0-20151101/dfn-l0-20151101/DFNSMALL25/25_2015-11-01_101629_DSC_0035/25_2015-11-01_101629_DSC_0035_47_4557-3224-4639-3403.differenced.jpg",
    #     "data/detections_dfn-l0-20151101/dfn-l0-20151101/DFNSMALL25/25_2015-11-01_101758_DSC_0038/25_2015-11-01_101758_DSC_0038_45_3993-4231-4114-4407.differenced.jpg"
    # ]

    # original_image = "data/detections_dfn-l0-20151101/dfn-l0-20151101/DFNSMALL25/25_2015-11-01_103259_DSC_0068/25_2015-11-01_103259_DSC_0068.thumb.jpg"

    # streak_images = [
    #     "data/detections_dfn-l0-20151101/dfn-l0-20151101/DFNSMALL25/25_2015-11-01_103259_DSC_0068/25_2015-11-01_103259_DSC_0068_55_3274-3204-3439-3646.differenced.jpg",
    #     "data/detections_dfn-l0-20151101/dfn-l0-20151101/DFNSMALL25/25_2015-11-01_103329_DSC_0069/25_2015-11-01_103329_DSC_0069_67_3083-3735-3244-4181.differenced.jpg"
    # ]

    # original_image = "data/detections_dfn-l0-20151101/dfn-l0-20151101/DFNSMALL10/10_2015-11-01_114358_DSC_0030/10_2015-11-01_114358_DSC_0030.thumb.jpg"

    # streak_images = [
    #     "data/detections_dfn-l0-20151101/dfn-l0-20151101/DFNSMALL10/10_2015-11-01_114358_DSC_0030/10_2015-11-01_114358_DSC_0030_33_4871-4320-4991-4604.differenced.jpg",
    #     "data/detections_dfn-l0-20151101/dfn-l0-20151101/DFNSMALL10/10_2015-11-01_114458_DSC_0032/10_2015-11-01_114458_DSC_0032_71_4828-1363-4982-1691.differenced.jpg",
    #     "data/detections_dfn-l0-20151101/dfn-l0-20151101/DFNSMALL10/10_2015-11-01_114528_DSC_0033/10_2015-11-01_114528_DSC_0033_52_4338-602-4735-1182.differenced.jpg",
    #     "data/detections_dfn-l0-20151101/dfn-l0-20151101/DFNSMALL10/10_2015-11-01_114628_DSC_0035/10_2015-11-01_114628_DSC_0035_50_3787-6-3949-166.differenced.jpg"
    # ]

    original_image = "data/detections_dfn-l0-20151101/dfn-l0-20151101/DFNSMALL16/16_2015-11-01_122829_DSC_0125/16_2015-11-01_122829_DSC_0125.thumb.jpg"

    streak_images = [
        "data/detections_dfn-l0-20151101/dfn-l0-20151101/DFNSMALL16/16_2015-11-01_122829_DSC_0125/16_2015-11-01_122829_DSC_0125_53_5436-2812-5487-3186.differenced.jpg",
        "data/detections_dfn-l0-20151101/dfn-l0-20151101/DFNSMALL16/16_2015-11-01_122859_DSC_0126/16_2015-11-01_122859_DSC_0126_33_5388-1801-5457-2200.differenced.jpg"
    ]

    original = ski.io.imread(original_image)

    # Initialize the plot
    _, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(original, cmap="gray", aspect="equal")
    ax.set_title("Start and End Points of Streaks on Original Image")

    colors = ["blue", "yellow", "orange", "green", "red", "purple", "cyan"]  # Colors for different streaks

    streak_lines = [StreakLine(s) for s in streak_images]

    for idx, streak in enumerate(streak_lines):

        if not streak.is_valid:
            print(f"Streak image {idx + 1} has less than 3 blobs. Skipping.")
            continue

        print(streak.midpoint, streak.angle)

        if idx > 0:
            prev_streak = streak_lines[idx - 1]
            print(streak.midpoint_to_midpoint(prev_streak), abs(streak.angle - prev_streak.angle))
            print(streak.same_trajectory(prev_streak))

        # Get start and end points
        start_point = streak.startpoint
        end_point = streak.endpoint

        # Draw a line connecting the start and end points
        ax.plot(
            [start_point[0], end_point[0]],
            [start_point[1], end_point[1]],
            linestyle="-",
            color=colors[idx % len(colors)],
            label=f"Streak {idx + 1} Line",
        )

    # Configure plot
    ax.legend(loc="upper right")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
    


if __name__ == "__main__":
    main()
