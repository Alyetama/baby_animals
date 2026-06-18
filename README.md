# Baby Animals Labeling and ML Pipeline

**Project Goal:** The primary objective of this project is to train machine learning models capable of classifying the life stage of animals from camera trap images.

To support this goal, this repository contains a set of scripts and utilities for managing an image dataset of animals, interfacing with Label Studio for annotation, creating image crops for training models, and deploying a FastAPI prediction server to generate pre-annotations.

## Scripts Overview

The following scripts are tracked and actively used in this repository:

### 1. `get_images.py`
A Python script used to parse and download images from an HTML dump.
- **Description:** Reads an HTML file, extracts image URLs via regex, and downloads them locally.
- **Arguments:**
  - `-f`, `--html-file`: HTML file to parse (default `html_body_2.txt`).
  - `-o`, `--output-dir`: Directory to save downloaded images.

### 2. `export_annotations.sh`
A bash script to quickly export annotations from Label Studio.
- **Description:** Uses `curl` to export project annotations in `JSON_MIN` format.
- **Requirements:** Requires a `.env` file defining `LABEL_STUDIO_TOKEN` and `LABEL_STUDIO_BASE_URL`.

### 3. `prepare_dataset.py`
Prepares a training and validation dataset for ML models from Label Studio annotations.
- **Description:** Parses exported `JSON_MIN` data, crops original images based on bounding box coordinates, and organizes the crops into classification folders before splitting them into `train` and `val` sets.
- **Arguments:**
  - `-d`, `--data-file-path`: Path to the MIN JSON data file.
  - `-c`, `--classes`: Comma-separated list of expected classes (e.g., `adult,juvenile`).
  - `-b`, `--classify-by`: Dict key used for classification.
  - `-o`, `--output-dir`: Output directory for the datasets.
  - `-s`, `--split_ratio`: Train-val split ratio (default: `0.8`).

### 4. `prediction_api.py`
A FastAPI-based inference server providing ML predictions for Label Studio.
- **Description:** When sent an image task, the API optionally queries a MegaDetector API to find animal bounding boxes, crops the image, and then runs a custom YOLO classification model on the crop. The results are returned as pre-annotations compatible with Label Studio. Features lazy model loading and GPU VRAM garbage collection for idle timeouts.
- **Arguments:**
  - `-w`, `--weights`: Path to the YOLO classification weights.
  - `-m`, `--model-version`: Model version name.
  - `-d`, `--image-dir`: Local image directories to prevent unnecessary downloads.
  - `--md-api-url`: MegaDetector API base URL.
  - `-H`, `--host`: API host (default `0.0.0.0`).
  - `-s`, `--port`: API port (default `8000`).

### 5. `send_prediction_tasks.py`
Automates the generation of predictions for unannotated tasks.
- **Description:** Fetches tasks from Label Studio, filters out cancelled or already-annotated tasks, calls the `prediction_api.py` endpoint, and uploads the predictions back to Label Studio to assist human annotators.
- **Arguments:**
  - `--ls-url`: Label Studio instance URL.
  - `--project-id`: Label Studio project ID.
  - `--ml-api-url`: The prediction API endpoint URL.
  - `--model-version`: The model version string to manage stale predictions.
  - `--after-date`: Only process tasks created after a specific date.

## Environment Variables
Some scripts (e.g. `export_annotations.sh`, `send_prediction_tasks.py`) expect a `.env` file in the project root. You can copy the provided `.env.example` file to create yours:
```bash
cp .env.example .env
```
Then, populate the .env file with the appropriate values.
