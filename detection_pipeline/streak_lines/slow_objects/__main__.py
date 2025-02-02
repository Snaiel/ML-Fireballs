import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from .slow_objects import find_slow_objects


def main():
    @dataclass
    class Args:
        folder_path: str
    
    parser = argparse.ArgumentParser(
        description="Plot a .differenced.jpg detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("folder_path", type=str, help="Detection outputs folder containing subfolders for each image")

    args = Args(**vars(parser.parse_args()))
    print("\nargs:", json.dumps(vars(args), indent=4), "\n")

    total = 0

    groups = find_slow_objects(Path(args.folder_path))
    
    for group in groups:
        print()
        for i in group:
            print(i)
        total += len(group)
    print()

    print(total)


if __name__ == "__main__":
    main()