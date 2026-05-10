# StreetViewSentinel - Opuntia Detection Pipeline

This project implements a complete computer vision pipeline for object detection using **YOLOv8 + SAHI sliced inference**, including dataset preprocessing, smart tiling, model inference, and structured export of predictions with metadata alignment.

---

## Project Overview

The pipeline is designed to:

1. Download and preprocess dataset from Roboflow
2. Apply smart tiling to large images
3. Train an object detection model (YOLOv8)
4. Run sliced inference using SAHI
5. Associate predictions with metadata
6. Export structured results to CSV for analysis

---

## Key Features

- Smart image tiling for high-resolution images  
- Sliced inference using SAHI for better detection on large images  
- YOLOv8-based detection model  
- Automatic metadata association per image  
- Reconstruction of original image identity from cropped halves (_L, _R)  
- Structured CSV export for downstream analysis  

---

## Project Structure

StreetViewSentinel_DataAnalyst/
│
├── data/
│   ├── raw/                  # Original dataset (Roboflow)
│   ├── processed/            # Tiled / transformed dataset
│
├── models/
│   └── yolo8_56.pt           # Trained detection model
│
├── scripts/
│   ├── inference.py          # Runs SAHI inference (outputs JSON)
│   ├── export_csv.py         # Converts predictions to CSV
│   ├── preprocessing.py      # Smart tiling & cropping
│   └── training.py           # Model training
│
├── outputs/
│   ├── predictions_raw.json  # Raw inference results
│   └── predictions.csv       # Final structured dataset
│
├── utils/
│   ├── metadata.py
│   ├── image_utils.py
│
├── requirements.txt
└── README.md

---

## Pipeline Workflow

### 1. Dataset Preparation

Images are downloaded from Roboflow and stored in: data/raw/

Each image may include:
- Original image
- Metadata file (same name as image)
- Additional auxiliary files (ignored)

---

### 2. Smart Tiling (Preprocessing)

Large images are split into smaller tiles to improve detection performance:
- Vertical split into left/right halves  
- Cropping using predefined margins  
- Outputs saved in data/processed/

---

### 3. Model Training

A YOLOv8 model is trained using the processed dataset.
Model is stored in: models/yolo8_56.pt

---

### 4. Inference (SAHI)

Run sliced inference:
python scripts/inference.py

This step:
- Uses SAHI sliding window inference
- Processes each image in tiles
- Outputs raw predictions in JSON format
- Output: outputs/predictions_raw.json

---

### 5. Export to CSV

Convert predictions to structured dataset:
python scripts/export_csv.py

This step:
- Loads raw predictions
- Matches metadata files
- Reconstructs original image identity (_L, _R)
- Extracts bounding boxes and confidence scores
- Generates final CSV
- Output: outputs/predictions.csv

---

## CSV Format

| Column | Description |
| :--- | :--- |
| image_file | Cropped image used for inference |
| original_image | Original image name |
| crop_side | Left / Right / Full |
| metadata_file | Associated metadata file |
| metadata_content | Raw metadata |
| class | Predicted class |
| confidence | Detection confidence |
| bbox_xmin | Bounding box coordinate |
| bbox_ymin | Bounding box coordinate |
| bbox_xmax | Bounding box coordinate |
| bbox_ymax | Bounding box coordinate |

---

## Requirements

Install dependencies:
pip install -r requirements.txt

Main libraries:
- sahi
- ultralytics
- opencv-python
- numpy
- pandas

---

## Execution Pipeline

1. Run inference: python scripts/inference.py
2. Export results: python scripts/export_csv.py

---

## Notes

- outputs/ is ignored in version control (.gitignore)
- Metadata must share the same base filename as images
- Cropped images are identified using _L and _R suffixes
- Designed for high-resolution aerial/street imagery

---

## Future Improvements

- Parquet export instead of CSV
- Parallel inference
- Experiment tracking (MLflow integration)
- Automatic dataset validation
- Spatial reconstruction of detections across tiles

---

## Author

Developed for a Master’s Thesis (TFM) in Bioinformatics and Biostatistics.