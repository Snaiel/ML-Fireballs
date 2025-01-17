import argparse
import json
from dataclasses import dataclass

import jsonlines


def parse_logs(path: str) -> list[dict]:
    parsed_objects = []

    with jsonlines.open(path) as reader:
        parsed_objects = list(reader.iter())
    
    return parsed_objects


def main() -> None:

    @dataclass
    class Args:
        logs_path: str
    
    parser = argparse.ArgumentParser(
        description="Print logs of detections.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--logs_path",
        type=str,
        required=True,
        help="Path to the logs."
    )
    
    args = Args(**vars(parser.parse_args()))
    print("\nargs:", json.dumps(vars(args), indent=4), "\n")

    parsed_logs = parse_logs(args.logs_path)
    for log in parsed_logs:
        print(json.dumps(log, indent=4))


if __name__ == "__main__":
    main()