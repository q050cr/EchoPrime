#!/usr/bin/env python3
"""
Analyze processing times from view classification results.

Usage:
    python scripts/investigate_timing.py data/path_gp_50
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np


def analyze_timing(data_dir):
    """Analyze processing times from all view_classification.json files."""
    
    # Find all view_classification.json files
    json_files = glob.glob(f'{data_dir}/**/view_classification.json', recursive=True)
    
    if not json_files:
        print(f"No view_classification.json files found in {data_dir}")
        return
    
    print(f"Found {len(json_files)} processed series\n")
    
    # Extract processing times
    times = []
    for json_path in json_files:
        with open(json_path) as f:
            data = json.load(f)
            time_sec = data['metadata']['processing_time_seconds']
            times.append(time_sec)
    
    times = np.array(times)
    
    # Calculate statistics
    print("Processing Time Statistics:")
    print(f"{'='*50}")
    print(f"Total series processed: {len(times)}")
    print(f"Total time:            {times.sum()/60:.2f} minutes ({times.sum():.1f} seconds)")
    print(f"\nPer-series statistics:")
    print(f"  Average:             {times.mean():.2f} seconds")
    print(f"  Std deviation:       {times.std():.2f} seconds")
    print(f"  Min:                 {times.min():.2f} seconds")
    print(f"  Max:                 {times.max():.2f} seconds")
    print(f"  Median:              {np.median(times):.2f} seconds")
    print(f"\nPercentiles:")
    print(f"  25th percentile:     {np.percentile(times, 25):.2f} seconds")
    print(f"  75th percentile:     {np.percentile(times, 75):.2f} seconds")
    print(f"  95th percentile:     {np.percentile(times, 95):.2f} seconds")


def main():
    parser = argparse.ArgumentParser(description='Analyze view classification processing times')
    parser.add_argument('data_dir', help='Root data directory (e.g., data/path_gp_50)')
    args = parser.parse_args()
    
    if not Path(args.data_dir).exists():
        print(f"Error: {args.data_dir} does not exist")
        sys.exit(1)
    
    analyze_timing(args.data_dir)


if __name__ == '__main__':
    main()
