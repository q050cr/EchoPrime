#!/usr/bin/env python3
"""
Visualize view classification results from JSON file.

Usage:
    python scripts/visualize_views.py results.json
    python scripts/visualize_views.py results.json --output custom_name.png
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize view classification results')
    parser.add_argument('json_file', help='JSON file with classification results OR series directory containing view_classification.json')
    parser.add_argument('--output', '-o', help='Output image path (default: figures/vc/<json_name>.png)')
    args = parser.parse_args()

    # If the path is a directory, look for view_classification.json inside it
    input_path = Path(args.json_file)
    if input_path.is_dir():
        json_path = input_path / 'view_classification.json'
        if not json_path.exists():
            print(f"Error: view_classification.json not found in {input_path}")
            sys.exit(1)
    elif input_path.exists():
        json_path = input_path
    else:
        print(f"Error: {args.json_file} not found")
        sys.exit(1)

    visualize_results(json_path, args.output)
