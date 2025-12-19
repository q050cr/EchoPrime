#!/usr/bin/env python3
"""
Batch process all studies to classify echocardiogram views.

Groups DICOM files by series directory and saves view classification results
as 'view_classification.json' in each series directory. Automatically skips
already-processed series.

Usage:
    # Process all series (skips already processed)
    uv run scripts/batch_classify.py data/path_gp_50

    # Test with a few series first
    uv run scripts/batch_classify.py data/path_gp_50 --limit 5

    # Use CPU instead of GPU
    uv run scripts/batch_classify.py data/path_gp_50 --device cpu

    # Force recalculation and overwrite existing JSON files
    uv run scripts/batch_classify.py data/path_gp_50 --force

Note: Activate .venv first with: source .venv/bin/activate
"""

import argparse
import glob
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
import utils
from scripts.classify_views import load_view_classifier, process_single_dicom


def batch_process(data_dir, device='cuda', limit=None, force=False):
    """Process all studies in data directory, grouping by series."""
    device = torch.device(device)
    
    # Load model once
    print("Loading view classifier...")
    model = load_view_classifier(device)
    
    # Preprocessing constants
    mean = torch.tensor([29.110628, 28.076836, 29.096405]).reshape(3, 1, 1)
    std = torch.tensor([47.989223, 46.456997, 47.20083]).reshape(3, 1, 1)
    
    # Find all DICOM files and group by series directory
    print(f"Scanning {data_dir} for DICOM files...")
    all_dicoms = glob.glob(f'{data_dir}/**/*.dcm', recursive=True)
    
    # Group by series directory (parent of DICOM file)
    series_map = defaultdict(list)
    for dcm_path in all_dicoms:
        series_dir = str(Path(dcm_path).parent)
        series_map[series_dir].append(dcm_path)
    
    print(f"Found {len(all_dicoms)} DICOM files in {len(series_map)} series")
    
    # Process each series
    total_start = time.time()
    processed_series = 0
    skipped_series = 0
    
    for series_dir, dicom_paths in series_map.items():
        # Skip if already processed (unless force flag is set)
        output_path = Path(series_dir) / 'view_classification.json'
        if output_path.exists() and not force:
            skipped_series += 1
            print(f"[SKIP {skipped_series}] {Path(series_dir).name} - already processed")
            continue
        
        # Check limit for testing
        if limit and processed_series >= limit:
            print(f"\nReached limit of {limit} series, stopping...")
            break
        
        series_start = time.time()
        
        # Process DICOMs in this series
        frames = []
        valid_paths = []
        
        for dicom_path in sorted(dicom_paths):
            try:
                frame = process_single_dicom(dicom_path, mean, std)
                if frame is not None:
                    frames.append(frame)
                    valid_paths.append(dicom_path)
            except Exception as e:
                print(f"Error processing {Path(dicom_path).name}: {e}")
        
        if len(frames) == 0:
            continue
        
        # Batch inference
        frames_tensor = torch.stack(frames).to(device)
        with torch.no_grad():
            logits = model(frames_tensor)
            probabilities = F.softmax(logits, dim=1)
            predicted_classes = torch.argmax(logits, dim=1)
        
        # Build results
        results = {}
        for dicom_path, pred_class, probs in zip(valid_paths, predicted_classes, probabilities):
            view_name = utils.COARSE_VIEWS[pred_class.item()]
            prob_dict = {utils.COARSE_VIEWS[i]: float(probs[i]) for i in range(11)}
            
            results[dicom_path] = {
                'predicted_view': view_name,
                'confidence': float(probs[pred_class]),
                'all_probabilities': prob_dict
            }
        
        # Add timing metadata
        series_time = time.time() - series_start
        metadata = {
            'series_dir': series_dir,
            'num_files': len(valid_paths),
            'processing_time_seconds': round(series_time, 3),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save to series directory  
        output_data = {
            'metadata': metadata,
            'results': results
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        processed_series += 1
        print(f"[{processed_series}/{len(series_map)}] {Path(series_dir).name}: {len(valid_paths)} videos in {series_time:.2f}s -> {output_path}")
    
    total_time = time.time() - total_start
    print(f"\nCompleted: {processed_series} series processed, {skipped_series} skipped, in {total_time:.2f}s")
    if processed_series > 0:
        print(f"Average: {total_time/processed_series:.2f}s per series")


def main():
    parser = argparse.ArgumentParser(description='Batch classify views for all series in data directory')
    parser.add_argument('data_dir', help='Root data directory (e.g., data/path_gp_50)')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--limit', type=int, help='Limit number of series to process (for testing)')
    parser.add_argument('--force', action='store_true', help='Force recalculation and overwrite existing JSON files')
    args = parser.parse_args()

    if not Path(args.data_dir).exists():
        print(f"Error: {args.data_dir} does not exist")
        sys.exit(1)

    batch_process(args.data_dir, device=args.device, limit=args.limit, force=args.force)


if __name__ == '__main__':
    main()
