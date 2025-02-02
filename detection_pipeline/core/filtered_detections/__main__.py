import argparse
import json
from dataclasses import dataclass

from ..saved_detections import remove_saved_detection
from .filtered_detections import FilteredDetections


def main():
    
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