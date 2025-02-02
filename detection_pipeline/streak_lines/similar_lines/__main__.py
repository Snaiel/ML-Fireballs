import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from ..utils import get_streak_lines
from .similar_lines import find_similar_lines


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

    streak_lines = get_streak_lines(Path(args.folder_path))
    groups = find_similar_lines(streak_lines)
    
    for group in groups:
        print()
        for i in group:
            print(i)
        total += len(group)
    print()

    print(total)


if __name__ == "__main__":
    main()