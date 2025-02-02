from detection_pipeline.streak_lines import (StreakLine, find_similar_lines,
                                             find_slow_objects,
                                             get_streak_lines)

from .saved_detections import get_detections, remove_saved_detection


class FilteredDetections:

    _all_detections: set
    _erroneous_detections: set
    _streak_lines: dict[str, StreakLine]
    _invalid_lines: set
    _similar_lines: list[list[str]]
    _slow_objects: list[list[str]]
    _final_detections: set

    def __init__(self, output_folder: str):
        """
        output_folder is the folder containing subfolders for each image

        ```txt
        DFNSMALL41 <- this one
            41_2015-01-14_110659_DSC_0108
                41_2015-01-14_110659_DSC_0108.json
                41_2015-01-14_110659_DSC_0108.thumb.differenced.jpg 
                41_2015-01-14_110659_DSC_0108.thumb.jpg 
                41_2015-01-14_110659_DSC_0108_20_5148-2666-5216-2805.differenced.jpg 
                41_2015-01-14_110659_DSC_0108_20_5148-2666-5216-2805.jpg
            41_2015-01-14_110729_DSC_0109
            41_2015-01-14_110959_DSC_0114
        ```
        """

        all_detections = set(get_detections(output_folder))
        erroneous_detections = set()

        streak_lines = get_streak_lines(output_folder)

        invalid_lines = {
            name
            for name, streak_line in streak_lines.items()
            if not streak_line.is_valid and
            int(name.split("_")[-2]) < 60 # if confidence is high, allow it
        }

        erroneous_detections = erroneous_detections.union(invalid_lines)

        similar_lines = find_similar_lines(streak_lines)
        for group in similar_lines:
            for i in group:
                erroneous_detections.add(i)

        slow_objects = find_slow_objects(output_folder, streak_lines)
        for group in slow_objects:
            for i in group:
                erroneous_detections.add(i)
        
        final_detections = all_detections.difference(erroneous_detections)

        self._all_detections = all_detections
        self._erroneous_detections = erroneous_detections
        self._invalid_lines = invalid_lines
        self._streak_lines = streak_lines
        self._similar_lines = similar_lines
        self._slow_objects = slow_objects
        self._final_detections = final_detections
    
    @property
    def all_detections(self) -> set:
        return self._all_detections

    @property
    def erroneous_detections(self) -> set:
        return self._erroneous_detections

    @property
    def streak_lines(self) -> dict[str, StreakLine]:
        return self._streak_lines

    @property
    def invalid_lines(self) -> set:
        return self._invalid_lines

    @property
    def similar_lines(self) -> list[list[str]]:
        return self._similar_lines

    @property
    def slow_objects(self) -> list[list[str]]:
        return self._slow_objects

    @property
    def final_detections(self) -> set:
        return self._final_detections

    @property
    def total_similar_lines(self) -> int:
        return sum(len(group) for group in self._similar_lines)

    @property
    def total_slow_objects(self) -> int:
        return sum(len(group) for group in self._slow_objects)


def main():
    import argparse
    import json
    from dataclasses import dataclass
    
    @dataclass
    class Args:
        output_path: str
        delete_erroneous: bool
    
    parser = argparse.ArgumentParser(
        description="Print filtered detections from output folder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("output_path", type=str, help="Detection outputs folder containing subfolders for each image")
    parser.add_argument("--delete_erroneous", action="store_true", help="Delete erroneous detections if flag is set")

    args = Args(**vars(parser.parse_args()))
    print("\nargs:", json.dumps(vars(args), indent=4), "\n")


    filtered_detections = FilteredDetections(args.output_path)


    print("\n\n\nDetections:\n")
    for detection in sorted(list(filtered_detections.all_detections)):
        print(detection)
    print("\nTotal detections:", len(filtered_detections.all_detections))


    print("\n\nInvalid lines:\n")
    for streak in sorted(list(filtered_detections.invalid_lines)):
        print(streak)
    print()
    print("Total invalid lines:", len(filtered_detections.invalid_lines))


    print("\n\nLikely static lines and slow moving objects (not mutually exclusive)")

    print("\nSimilar lines:\n")
    for group in filtered_detections.similar_lines:
        for i in group:
            print(i)
        print()
    print("Total similar lines:", filtered_detections.total_similar_lines)

    print("\n\nSlow moving objects:\n")
    for group in filtered_detections.slow_objects:
        for i in group:
            print(i)
        print()
    print("Total slow objects:", filtered_detections.total_slow_objects)


    print("\n\nCombined erroneous detections:\n")
    for i in sorted(list(filtered_detections.erroneous_detections)):
        print(i)
    print("\nTotal erroneous detections:", len(filtered_detections.erroneous_detections))

            
    print("\n\nFinal detections:\n")
    for i in sorted(list(filtered_detections.final_detections)):
        print(i)
    print("\nTotal final detections:", len(filtered_detections.final_detections))


    if args.delete_erroneous:
        for erroneous in filtered_detections.erroneous_detections:
            remove_saved_detection(args.output_path, erroneous)


if __name__ == "__main__":
    main()