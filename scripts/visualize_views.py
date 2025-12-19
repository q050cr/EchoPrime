#!/usr/bin/env python3
"""
Visualize view classification results from JSON file.

Usage:
    python scripts/visualize_views.py results.json
    python scripts/visualize_views.py results.json --output custom_name.png
    python scripts/visualize_views.py patient_dir/  # Finds all view_classification.json files recursively
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pydicom

sys.path.insert(0, str(Path(__file__).parent.parent))
import utils


def visualize_results(json_path, output_path=None):
    """Visualize predicted views on DICOM first frames."""
    with open(json_path) as f:
        data = json.load(f)

    # Handle nested structure (with 'results' key) or flat structure
    if 'results' in data:
        results = data['results']
    else:
        results = data

    if not results:
        print("No results to visualize")
        return

    # Calculate grid dimensions
    n = len(results)
    cols = 12
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols, rows))
    axes = axes.flatten() if n > 1 else [axes]

    for i, (dicom_path, result) in enumerate(results.items()):
        try:
            # Read and process DICOM first frame
            dcm = pydicom.dcmread(dicom_path)
            pixels = dcm.pixel_array

            if pixels.ndim == 3:
                pixels = np.repeat(pixels[..., None], 3, axis=3)

            pixels = utils.mask_outside_ultrasound(pixels)
            frame = utils.crop_and_scale(pixels[0])

            # Add view annotation
            view_text = result['predicted_view'].replace("_", " ")
            conf = result['confidence']
            cv2.putText(frame, f"{view_text}", (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
            cv2.putText(frame, f"{conf:.2f}", (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1)

            axes[i].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            axes[i].axis('off')
        except Exception as e:
            print(f"Error processing {Path(dicom_path).name}: {e}")
            axes[i].axis('off')

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.tight_layout()

    # Save figure
    if output_path is None:
        output_dir = Path("figures/vc")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{Path(json_path).stem}.png"

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to {output_path}")


def find_all_view_jsons(directory):
    """Recursively find all view_classification.json files in directory."""
    return sorted(Path(directory).rglob('view_classification.json'))


def generate_output_path(json_path, base_output, total=None):
    """Generate output path for a given JSON file."""
    if base_output:
        base_output = Path(base_output)
        if total == 1:
            # Only one JSON found, use the exact output name
            return base_output
        else:
            # Multiple JSONs, append series directory name
            series_dir = json_path.parent.name
            stem = base_output.stem
            suffix = base_output.suffix
            return base_output.parent / f"{stem}_{series_dir}{suffix}"
    else:
        # Default path with series directory name
        output_dir = Path("figures/vc")
        output_dir.mkdir(parents=True, exist_ok=True)
        series_dir = json_path.parent.name
        return output_dir / f"{series_dir}.png"


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize view classification results')
    parser.add_argument('json_file', help='JSON file with classification results OR directory (searches recursively for view_classification.json)')
    parser.add_argument('--output', '-o', help='Output image path (default: figures/vc/<series_name>.png)')
    args = parser.parse_args()

    input_path = Path(args.json_file)

    if input_path.is_dir():
        # Check if view_classification.json exists in the exact directory first
        direct_json = input_path / 'view_classification.json'
        if direct_json.exists():
            json_paths = [direct_json]
        else:
            # Search recursively for all view_classification.json files
            json_paths = find_all_view_jsons(input_path)

        if not json_paths:
            print(f"Error: No view_classification.json files found in {input_path}")
            sys.exit(1)

        print(f"Found {len(json_paths)} view_classification.json file(s)")

        # Process each JSON file
        for idx, json_path in enumerate(json_paths, 1):
            print(f"\n[{idx}/{len(json_paths)}] Processing {json_path.parent.name}...")
            output_path = generate_output_path(json_path, args.output, len(json_paths))
            visualize_results(json_path, output_path)

    elif input_path.exists():
        # Direct JSON file path
        visualize_results(input_path, args.output)
    else:
        print(f"Error: {args.json_file} not found")
        sys.exit(1)
