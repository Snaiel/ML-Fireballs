from detection_pipeline.streak_lines import (StreakLine, find_similar_lines,
                                             find_slow_objects,
                                             get_streak_lines)

from ..saved_detections import get_detections, remove_saved_detection


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
