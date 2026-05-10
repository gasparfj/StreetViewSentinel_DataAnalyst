import os
import json
import time
from pathlib import Path

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction


# =========================
# CONFIGURATION
# =========================

MODEL_PATH = r"models/yolo8_56.pt"
INPUT_DIR = r"D:/test_clas_opuntia/test/Opuntia"
OUTPUT_DIR = Path("outputs")
RAW_PRED_PATH = OUTPUT_DIR / "predictions_raw.json"

VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


# =========================
# MODEL INITIALIZATION
# =========================

detection_model = AutoDetectionModel.from_pretrained(
    model_type="yolov8",
    model_path=MODEL_PATH,
    confidence_threshold=0.3,
    device="cpu",
)


# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# INFERENCE PIPELINE
# =========================

def run_inference():
    """
    Runs inference over all images in the input directory using SAHI sliced prediction.

    The function:
    - Loads a pretrained YOLOv8 model via SAHI
    - Iterates over all valid images in INPUT_DIR
    - Applies sliced inference (tiling-based detection)
    - Stores results in a structured JSON format
    - Saves raw predictions for later post-processing

    Output format:
        [
            {
                "image": "image_name.jpg",
                "predictions": [
                    {
                        "class": str,
                        "confidence": float,
                        "bbox": {
                            "xmin": float,
                            "ymin": float,
                            "xmax": float,
                            "ymax": float
                        }
                    }
                ]
            }
        ]

    Returns
    -------
    None
    """

    all_results = []

    start_total = time.time()

    for file in os.listdir(INPUT_DIR):

        if not file.lower().endswith(VALID_EXT):
            continue

        image_path = str(Path(INPUT_DIR) / file)

        print(f"Processing: {file}")

        start_img = time.time()

        # Run SAHI sliced prediction
        result = get_sliced_prediction(
            image_path,
            detection_model,
            slice_height=256,
            slice_width=256,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
        )

        # Serialize predictions
        image_results = {
            "image": file,
            "predictions": []
        }

        for pred in result.object_prediction_list:

            bbox = pred.bbox

            image_results["predictions"].append({
                "class": pred.category.name,
                "confidence": float(pred.score.value),
                "bbox": {
                    "xmin": bbox.minx,
                    "ymin": bbox.miny,
                    "xmax": bbox.maxx,
                    "ymax": bbox.maxy,
                }
            })

        all_results.append(image_results)

        end_img = time.time()
        print(f"Image time: {end_img - start_img:.2f}s")

    # Save JSON output
    with open(RAW_PRED_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4)

    end_total = time.time()

    print("\n==============================")
    print(f"Total time: {end_total - start_total:.2f}s")
    print(f"Saved raw predictions: {RAW_PRED_PATH}")
    print("==============================")


# =========================
# ENTRY POINT
# =========================

if __name__ == "__main__":
    run_inference()