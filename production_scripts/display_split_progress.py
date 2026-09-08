#!/usr/bin/env python3
"""
activate_PV_detection;
export PYTHONPATH=~/sgis:~/sgis/src:~/sgis/production_scripts:$PYTHONPATH
dept_code_list="22,44,49,50,63";
python ~/sgis/production_scripts/display_split_progress.py --dept_code_list $dept_code_list
"""

import argparse
from pathlib import Path

def count_lines(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            return sum(1 for _ in handle)
    except FileNotFoundError:
        raise

def return_associated_jp2_directory(file: Path) -> Path:
    with file.open("r", encoding="utf-8") as f:
        first_line = f.readline().strip()
    Path_jp2 = Path(first_line).parent
    # Exemple : ~/temporary_LaCie/rasters/only_tiles/44/2025/44-2025-0M20-RGB/*.jp2
    return Path_jp2

def print_first_line(file: Path):
    with file.open("r", encoding="utf-8") as f:
        first_line = f.readline().strip()
    print(f"{file}: {first_line}")


def count_jp2_files(directory: Path) -> int:
    if not directory.is_dir():
        return 0
    return sum(1 for path in directory.iterdir() if path.is_file() and path.suffix.lower() == ".jp2")


def main(dept="all"):
    
    print("Searching for progress.txt files in ~/split/...")
    from pathlib import Path
    if dept == "all":
        root = Path("/home/nerotb/split/") # TODO make this more generic
    else:
        root = Path(f"/home/nerotb/split/{dept}")
    
    print(f"Searching in {root}...")
    for progress_txt_file in root.rglob("progress.txt"):
        nb_lines = count_lines(progress_txt_file)
        nb_jp2_files = count_jp2_files(return_associated_jp2_directory(progress_txt_file))
        print(f"{progress_txt_file}: {nb_lines} lines, {nb_jp2_files} JP2 files")   
        
def print_split_progress(dept_code_list : list[str]):
    for dept in dept_code_list:
        main(dept)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Display the progress of split processing')
    parser.add_argument("--dept_code_list"  , type=str, required=True       , help="Comma-separated list of department codes: 2 digits from 01 to 99 else 3 digits")
    args = parser.parse_args()    
    dept_code_list = args.dept_code_list.split(",")

    print_split_progress(dept_code_list)
