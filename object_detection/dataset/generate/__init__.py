from dataclasses import dataclass
import argparse


@dataclass
class GenerateDatasetArgs:
    negative_ratio: int = 1
    overwrite: bool = False
    num_processes: int = 8
    source: str = "all"


def get_args() -> GenerateDatasetArgs:
    parser = argparse.ArgumentParser(
        description="Generate dataset for object detection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--negative_ratio', type=int, default=1, 
                        help='Ratio of negative examples to positive examples.')
    parser.add_argument('--overwrite', action='store_true', 
                        help='Overwrite the output directory if it exists.')
    parser.add_argument('--num_processes', type=int, default=8,
                        help='Number of processes to use for multiprocessing.')
    parser.add_argument('--source', type=str, choices=['all', '2015_removed'], default='all',
                        help='Source of the dataset, either "all" or "2015_removed".')

    args = GenerateDatasetArgs(**vars(parser.parse_args()))

    return args