#!/usr/bin/env python3
"""
View Classification Script
Classifies echocardiogram views for all DICOM files in a study directory.

Usage:
    python scripts/classify_views.py model_data/PSEUDO_PATH_GP000001
    python scripts/classify_views.py model_data/PSEUDO_PATH_GP000001 --output results.json
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pydicom
import torch
import torch.nn.functional as F
import torchvision

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent))
import utils


def load_view_classifier(device):
    """Load the pretrained view classifier model."""
    state_dict = torch.load("model_data/weights/view_classifier.pt", map_location=device)

    model = torchvision.models.convnext_base()
    model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, 11)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    return model


def process_single_dicom(dicom_path, mean, std):
    """Process a single DICOM file and return the first frame tensor."""
    dcm = pydicom.dcmread(dicom_path)
    pixels = dcm.pixel_array

    # Skip non-video files
    if pixels.ndim < 3 or pixels.shape[2] == 3:
        return None

    # Convert single channel to 3 channels
    if pixels.ndim == 3:
        pixels = np.repeat(pixels[..., None], 3, axis=3)

    # Mask outside ultrasound region
    pixels = utils.mask_outside_ultrasound(pixels)

    # Preprocess first frame
    first_frame = utils.crop_and_scale(pixels[0])
    first_frame = torch.as_tensor(first_frame, dtype=torch.float).permute([2, 0, 1])

    # Normalize - mean/std shape: (3, 1, 1)
    first_frame.sub_(mean).div_(std)

    return first_frame


def classify_study(study_path, device='cuda', output_file=None):
    """
    Classify all DICOM files in a study directory.

    Args:
        study_path: Path to study directory containing DICOM files
        device: Device to run inference on ('cuda' or 'cpu')
        output_file: Optional path to save results as JSON

    Returns:
        Dictionary mapping DICOM paths to classification results
    """
    device = torch.device(device)

    # Load model
    print(f"Loading view classifier...")
    model = load_view_classifier(device)

    # Preprocessing constants
    mean = torch.tensor([29.110628, 28.076836, 29.096405]).reshape(3, 1, 1)
    std = torch.tensor([47.989223, 46.456997, 47.20083]).reshape(3, 1, 1)

    # Find all DICOM files
    dicom_paths = sorted(glob.glob(f'{study_path}/**/*.dcm', recursive=True))
    print(f"Found {len(dicom_paths)} DICOM files in {study_path}")

    if len(dicom_paths) == 0:
        print("No DICOM files found!")
        return {}

    results = {}
    frames = []
    valid_paths = []

    # Process all DICOMs
    print("Processing DICOM files...")
    for dicom_path in dicom_paths:
        try:
            frame = process_single_dicom(dicom_path, mean, std)
            if frame is not None:
                frames.append(frame)
                valid_paths.append(dicom_path)
        except Exception as e:
            print(f"Error processing {dicom_path}: {e}")

    if len(frames) == 0:
        print("No valid video files found!")
        return {}

    # Batch inference
    print(f"Classifying {len(frames)} videos...")
    frames_tensor = torch.stack(frames).to(device)

    with torch.no_grad():
        logits = model(frames_tensor)
        probabilities = F.softmax(logits, dim=1)
        predicted_classes = torch.argmax(logits, dim=1)

    # Build results dictionary
    for dicom_path, pred_class, probs in zip(valid_paths, predicted_classes, probabilities):
        view_name = utils.COARSE_VIEWS[pred_class.item()]
        prob_dict = {utils.COARSE_VIEWS[i]: float(probs[i]) for i in range(11)}

        results[dicom_path] = {
            'predicted_view': view_name,
            'confidence': float(probs[pred_class]),
            'all_probabilities': prob_dict
        }

    # Print summary
    print(f"\nClassification Summary:")
    print(f"{'View':<25} {'Count':>6}")
    print("-" * 32)
    view_counts = {}
    for result in results.values():
        view = result['predicted_view']
        view_counts[view] = view_counts.get(view, 0) + 1

    for view, count in sorted(view_counts.items()):
        print(f"{view:<25} {count:>6}")

    # Save to file if requested
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Classify echocardiogram views from DICOM files')
    parser.add_argument('study_path', help='Path to study directory containing DICOM files')
    parser.add_argument('--output', '-o', help='Output JSON file path')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'],
                        help='Device to run inference on')

    args = parser.parse_args()

    if not Path(args.study_path).exists():
        print(f"Error: Study path '{args.study_path}' does not exist")
        sys.exit(1)

    results = classify_study(args.study_path, device=args.device, output_file=args.output)

    if not args.output:
        # Print sample results
        print("\nSample Results (first 3):")
        for i, (path, result) in enumerate(list(results.items())[:3]):
            print(f"\n{Path(path).name}:")
            print(f"  View: {result['predicted_view']} (confidence: {result['confidence']:.3f})")
            top_3 = sorted(result['all_probabilities'].items(), key=lambda x: x[1], reverse=True)[:3]
            print(f"  Top 3: {', '.join([f'{v}={p:.3f}' for v, p in top_3])}")


if __name__ == '__main__':
    main()
