# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Guidelines

- **Package Manager**: Use `uv` for package management
- **Code Style**: Write concise code - avoid unnecessary verbosity
- **File Creation**: Only create new files when absolutely necessary. Do not create README files unless explicitly requested
- **Virtual Environment**: Activate `.venv` before running Python scripts: `source .venv/bin/activate`

## Project Overview

EchoPrime is a Multi-Video View-Informed Vision-Language Model for Comprehensive Echocardiography Interpretation. The system processes DICOM echocardiogram videos, generates latent embeddings, and produces clinical reports and cardiac measurements.

Paper: https://arxiv.org/abs/2410.09704

## Setup and Installation

### Initial Setup
```bash
# Clone and navigate to repository
git clone <repo-url>
cd EchoPrime

# Download model data
wget https://github.com/echonet/EchoPrime/releases/download/v1.0.0/model_data.zip
wget https://github.com/echonet/EchoPrime/releases/download/v1.0.0/candidate_embeddings_p1.pt
wget https://github.com/echonet/EchoPrime/releases/download/v1.0.0/candidate_embeddings_p2.pt
unzip model_data.zip
mv candidate_embeddings_p1.pt model_data/candidates_data/
mv candidate_embeddings_p2.pt model_data/candidates_data/

# Install dependencies (using uv)
uv sync
# or with pip
pip install -r requirements.txt

# Activate virtual environment
source .venv/bin/activate
```

### Docker Setup
```bash
# Build the container
docker build -t echo-prime .

# Run the container with GPU support
docker run -d --name echoprime-container --gpus all echo-prime tail -f /dev/null

# Attach and access notebook at /workspace/EchoPrime/EchoPrimeDemo.ipynb
```

## Core Architecture

### Model Components

The system consists of three neural network components:

1. **Echo Encoder** (`echo_prime_encoder.pt`): MViT-v2 based video encoder that transforms preprocessed echo videos into 512-dimensional embeddings
2. **View Classifier** (`view_classifier.ckpt`): ConvNeXt-based classifier that identifies echocardiogram views from video frames (11 view types)
3. **Candidate Database**: Pre-computed embeddings, reports, and labels from 1.2M+ candidate studies used for nearest-neighbor retrieval

### Data Flow

```
DICOM files → Video Processing → View Classification → Embedding → Report Generation
                                                    ↓
                                              Metric Prediction
```

1. **Input Processing**: DICOM files are loaded from study directories (e.g., `model_data/example_study/**/*.dcm`)
2. **Preprocessing** ([video_utils.py](video_utils.py)):
   - Mask outside ultrasound region using `mask_outside_ultrasound()`
   - Crop and scale frames to 224x224 using `crop_and_scale()`
   - Normalize with mean=[29.11, 28.08, 29.10], std=[47.99, 46.46, 47.20]
   - Sample 32 frames with stride 2 → 16 frames per video
3. **Embedding Generation**: Each video → 512-dim feature vector via echo_encoder
4. **View Classification**: First frame of each video → one-hot encoding (11 classes)
5. **Study Encoding**: Concatenate embeddings + view encodings → (N videos, 523) tensor

### Report Generation Pipeline

Located in [EchoPrimeDemo.ipynb](EchoPrimeDemo.ipynb):

- **Section-Based Generation**: Reports are structured into 16 anatomical sections (defined in `utils.ALL_SECTIONS`)
- **MIL Weighting**: Multiple Instance Learning weights ([MIL_weights.csv](MIL_weights.csv)) determine which views contribute to each section
- **Retrieval-Based**: For each section:
  1. Compute weighted study embedding using view-specific MIL weights
  2. Find most similar candidate embedding via cosine similarity
  3. Extract corresponding section from candidate report
- **Phrase Structure**: Reports use templated phrases with regex patterns ([per_section.json](per_section.json), [all_phr.json](all_phr.json))

### Metric Prediction

The `predict_metrics()` function predicts 21 clinical features:
- Binary features (e.g., pacemaker presence, valve stenosis/regurgitation)
- Continuous features (e.g., ejection fraction, pulmonary artery pressure)
- Uses k-NN (k=50) on per-section embeddings weighted by MIL weights
- ROC thresholds in [roc_thresholds.csv](roc_thresholds.csv) maximize TPR while minimizing FPR

## Key Files

- [utils.py](utils.py): Report structuring, feature extraction, phrase encoding/decoding
- [video_utils.py](video_utils.py): DICOM processing, ultrasound masking, video I/O operations
- [EchoPrimeDemo.ipynb](EchoPrimeDemo.ipynb): Main inference pipeline demonstration
- [ViewClassificationDemo.ipynb](ViewClassificationDemo.ipynb): View classification workflow
- `model_data/weights/`: Pre-trained model checkpoints
- `model_data/candidates_data/`: Retrieval database (embeddings, reports, labels, studies)

## Important Constants

### View Types (11 classes)
```python
COARSE_VIEWS = ['A2C', 'A3C', 'A4C', 'A5C', 'Apical_Doppler',
                'Doppler_Parasternal_Long', 'Doppler_Parasternal_Short',
                'Parasternal_Long', 'Parasternal_Short', 'SSN', 'Subcostal']
```

### Anatomical Sections (16 sections)
```python
ALL_SECTIONS = ["Left Ventricle", "Resting Segmental Wall Motion Analysis",
                "Right Ventricle", "Left Atrium", "Right Atrium",
                "Atrial Septum", "Mitral Valve", "Aortic Valve",
                "Tricuspid Valve", "Pulmonic Valve", "Pericardium",
                "Aorta", "IVC", "Pulmonary Artery", "Pulmonary Veins",
                "Postoperative Findings"]
```

### Clinical Features (21 predictions)
Defined in [per_section.json](per_section.json), includes:
- `ejection_fraction`, `pulmonary_artery_pressure_continuous` (regression)
- `mitral_regurgitation`, `aortic_stenosis`, `pericardial_effusion`, etc. (binary)

## Processing Requirements

- **GPU Required**: Models run on CUDA (specified as `device = torch.device("cuda")`)
- **Memory**: Batch size of 50 videos used in embedding generation
- **Input Format**: DICOM files with 4D pixel arrays (F×H×W×C where F=frames)
- **Color Space**: Handles YBR_FULL → RGB conversion for DICOM files

## Common Gotchas

1. **Green-Tinted Images**: Ensure correct libraries from requirements.txt are installed (opencv-python-headless==4.5.5.64)
2. **View Classification**: Only the first frame of each video is used for view classification
3. **Padding**: Videos shorter than 32 frames are zero-padded before encoding
4. **Normalization**: Preprocessing statistics are specific to the training data distribution
5. **Report Structure**: Generated reports use `[SEP]` tokens between sections
6. **Study Structure**: Input must be organized as `<study_dir>/**/*.dcm` with DICOM files nested in subdirectories
