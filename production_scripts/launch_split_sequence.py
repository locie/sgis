#!/usr/bin/env python3
"""
activate_PV_detection;
export PYTHONPATH=~/sgis:~/sgis/src:~/sgis/production_scripts:$PYTHONPATH
dept_code_list="22,44,49,50,63"; year="2025"; resolution=20; threads_num=6; 
python ~/sgis/production_scripts/launch_split_sequence.py --dept_code_list $dept_code_list --year $year --resolution $resolution --threads_num $threads_num
"""


import argparse
from split import main as run_split

RESOLUTION = 20
THREADS_NUM = 6

def launch_split_sequence(dept_code_list: list[str], year : str, resolution : int = RESOLUTION, threads_num : int = THREADS_NUM) -> None:
    """
    Launch the split sequence for the given department code and year.
    
    Args:
        dept_code_list (list[str]): List of department codes.
        year (str): The year for which to launch the split sequence.
        resolution (int): The resolution for the split sequence.
        threads_num (int): The number of threads to use for the split sequence.
    """

    # Launch the split sequence for each department code
    for dept in dept_code_list:
        run_split(dept, year, resolution, threads_num)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--year"            , type=str, required=True       , help='BDOrtho version [YYYY]')
    parser.add_argument("--dept_code_list"  , type=str, required=True       , help="Comma-separated list of department codes: 2 digits from 01 to 99 else 3 digits")
    parser.add_argument("--threads_num"     , type=int, default=THREADS_NUM , choices=range(1, 7), help='Number of concurrent threads for splitting. Each thread splits one raster tile.')
    parser.add_argument("--resolution"      , type=int, default=RESOLUTION  , help='Raster resolution, in cm.')
    args = parser.parse_args()
    
    dept_code_list = args.dept_code_list.split(",")
    year = args.year
    threads_num = args.threads_num
    resolution = args.resolution
    
    launch_split_sequence(dept_code_list, year, resolution, threads_num)
    