from pathlib import Path

from detection_pipeline.streak_lines.streak_line import StreakLine


def find_similar_lines(streak_lines: dict[str, StreakLine]) -> list[list[str]]:
    """
        Compares streak lines (assumed to be of detections from a camera's night of images)
        to check if lines are similar in position, length, angle between each other.

        Returns a list of lists of detections where each sublist contains detections
        whose lines are similar to each other.
    """

    groups: list[list] = []

    detections = list(streak_lines.keys())

    if not streak_lines:
        return groups

    for i in range(0, len(streak_lines) - 1):
        
        current_image: Path = detections[i]
        current_streak = streak_lines[current_image]

        if not current_streak.is_valid:
            continue
        
        for j in range(i + 1, len(streak_lines)):
            
            other_image = detections[j]
            other_streak = streak_lines[other_image]

            if not other_streak.is_valid:
                continue
            
            # print(current_image.name)
            # print(other_image.name)
            # print(
            #     current_streak.midpoint_to_midpoint(other_streak),
            #     current_streak.angle_between(other_streak),
            #     current_streak.length,
            #     other_streak.length
            # )
            # print()

            if current_streak.similar_line(other_streak):
                for group in groups:
                    if current_image in group:
                        if other_image not in group:
                            group.append(other_image)
                        break
                    elif other_image in group:
                        if current_image not in group:
                            group.append(current_image)
                            group.sort()
                        break
                else:
                    groups.append([current_image, other_image])
    
    return groups
