import os, shutil
from pathlib import Path
from dataset import GFO_PICKINGS, GFO_FIXES_FOLDER


def main():
    if not GFO_FIXES_FOLDER.exists():
        os.mkdir(GFO_FIXES_FOLDER)

    with open(Path(Path(__file__).parents[3], "data", "dodgy_fireballs.txt"), "r") as file:
        dodgy_fireballs = [line.strip() for line in file.readlines()]

    for i in dodgy_fireballs:
        print(i)
        csv_file = Path(GFO_PICKINGS, i + ".csv")
        if not csv_file.exists():
            print("does not exist")
            continue
        shutil.copy(
            Path(GFO_PICKINGS, i + ".csv"),
            GFO_FIXES_FOLDER
        )

    print(f"Copied dodgy fireball pickings to {GFO_FIXES_FOLDER}, now go and fix them lol.")


if __name__ == "__main__":
    main()