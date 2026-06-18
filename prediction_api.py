#!/usr/bin/env python
# coding: utf-8

import argparse
import gc
import io
import os
import threading
import time
import uuid
from typing import Optional
from urllib.parse import urlparse

import requests
import torch
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel
from ultralytics import YOLO

# ----------------------------------------------------------------------------

app = FastAPI()

# --- Global Variables for Memory Management ---
MODEL_OBJ = None
WEIGHTS_PATH = None
MODEL_VERSION = None
IMAGE_DIR = None
MD_API_URL = None

LAST_ACTIVE_TIME = time.time()
IDLE_TIMEOUT_SECONDS = 300
# ----------------------------------------------------------------------------


def opts() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('-w',
                        '--weights',
                        help='Path to the classification weights file',
                        type=str)
    parser.add_argument('-m',
                        '--model-version',
                        help='Name and model version',
                        type=str)
    parser.add_argument(
        '-d',
        '--image-dir',
        help='Optional: Local directory (or directories) containing images to avoid downloading',
        type=str,
        nargs='+',
        default=None)
    parser.add_argument(
        '--md-api-url',
        help='MegaDetector API base URL (e.g. http://localhost:62835)',
        type=str,
        default=None)
    parser.add_argument('-H',
                        '--host',
                        help='API host (default: 0.0.0.0)',
                        type=str,
                        default='0.0.0.0')
    parser.add_argument('-s',
                        '--port',
                        help='API port (default: 8000)',
                        type=int,
                        default=8000)
    return parser.parse_args()


# ----------------------------------------------------------------------------


class Task(BaseModel):
    task: dict
    project: Optional[int] = None


def load_model_lazy():
    """Loads the model into the GPU if it isn't already loaded."""
    global MODEL_OBJ
    if MODEL_OBJ is None:
        print(f"Cold start: Loading YOLO model from {WEIGHTS_PATH} to GPU...")
        MODEL_OBJ = YOLO(WEIGHTS_PATH)


def unload_model():
    """Deletes the model and forces the GPU to clear the VRAM cache."""
    global MODEL_OBJ
    if MODEL_OBJ is not None:
        print(
            f"Idle timeout ({IDLE_TIMEOUT_SECONDS}s) reached. Unloading YOLO model from GPU VRAM..."
        )
        del MODEL_OBJ
        MODEL_OBJ = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def memory_manager():
    """Background thread that checks for inactivity."""
    global LAST_ACTIVE_TIME
    while True:
        time.sleep(60)
        if MODEL_OBJ is not None and (time.time() -
                                      LAST_ACTIVE_TIME) > IDLE_TIMEOUT_SECONDS:
            unload_model()


# ----------------------------------------------------------------------------
# Result builders


def _rectangle_dict(region_id: str, x: float, y: float, w: float,
                    h: float) -> dict:
    return {
        'id': region_id,
        'type': 'rectangle',
        'from_name': 'rect-1',
        'to_name': 'image',
        'value': {
            'x': x,
            'y': y,
            'width': w,
            'height': h,
            'rotation': 0,
        },
    }


def _choices_pred_dict(model_version: str, label: str, score: float,
                       region_id: str) -> dict:
    return {
        'id': region_id,  # same id as the rectangle — this is how LS links perRegion choices
        'type': 'choices',
        'score': score,
        'value': {'choices': [label]},
        'to_name': 'image',
        'from_name': 'life_stage',
        'model_version': model_version,
    }


# ----------------------------------------------------------------------------
# MegaDetector helpers


def _call_md_api(task: dict) -> list:
    """Call MegaDetector API; return list of LS value dicts for animal detections."""
    if not MD_API_URL:
        print("MD_API_URL not set — skipping MegaDetector step")
        return []
    try:
        r = requests.post(f"{MD_API_URL}/predict",
                          json={'task': task},
                          timeout=60)
        if r.status_code != 200:
            print(f"MD API error {r.status_code}: {r.text}")
            return []
        detections = []
        for item in r.json().get('result', []):
            val = item.get('value', {})
            labels = val.get('rectanglelabels', [])
            if labels and labels[0].lower() == 'animal':
                detections.append(val)
        return detections
    except Exception as e:
        print(f"MD API call failed: {e}")
        return []


def _crop_image(img: Image.Image, x_pct: float, y_pct: float, w_pct: float,
                h_pct: float) -> Image.Image:
    """Crop a PIL image using Label Studio percentage coordinates (top-left origin)."""
    img_w, img_h = img.size
    x0 = max(0, int(x_pct / 100 * img_w))
    y0 = max(0, int(y_pct / 100 * img_h))
    x1 = min(img_w, int((x_pct + w_pct) / 100 * img_w))
    y1 = min(img_h, int((y_pct + h_pct) / 100 * img_h))
    return img.crop((x0, y0, x1, y1))


# ----------------------------------------------------------------------------


@app.post('/predict')
def predict_endpoint(task: Task):
    global LAST_ACTIVE_TIME
    LAST_ACTIVE_TIME = time.time()

    _task = task.task
    if not _task.get('project'):
        if task.project:
            _task['project'] = task.project
        else:
            raise HTTPException(
                404, 'Parameter `project` is required when the task does not '
                'contain a project id number!')
    task = _task

    load_model_lazy()
    model = MODEL_OBJ
    model_version = MODEL_VERSION

    image_url = task['data']['image']
    filename = os.path.basename(urlparse(image_url).path)

    # Resolve image — prefer local copy to avoid double download
    local_image_path = None
    if IMAGE_DIR:
        for d in IMAGE_DIR:
            candidate = os.path.join(d, filename)
            if os.path.exists(candidate):
                local_image_path = candidate
                break

    if local_image_path:
        pil_img = Image.open(local_image_path).convert('RGB')
    else:
        r = requests.get(image_url)
        if r.status_code != 200:
            return JSONResponse(content=r.text, status_code=404)
        pil_img = Image.open(io.BytesIO(r.content)).convert('RGB')

    # Step 1: MegaDetector — get animal bounding boxes
    detections = _call_md_api(task)

    if not detections:
        return JSONResponse(status_code=200, content={})

    # Step 2: Classify each detected animal crop
    results_list = []
    scores = []

    for det in detections:
        x, y, w, h = det['x'], det['y'], det['width'], det['height']

        crop = _crop_image(pil_img, x, y, w, h)
        model_preds = model(crop)  # ultralytics accepts PIL.Image directly
        result = model_preds[0]

        if result.probs is None:
            continue

        top_idx = int(result.probs.top1)
        score = float(result.probs.top1conf.item())
        label = model.names[top_idx]

        region_id = uuid.uuid4().hex[:8]
        scores.append(score)
        results_list.append(_rectangle_dict(region_id, x, y, w, h))
        results_list.append(_choices_pred_dict(model_version, label, score, region_id))

    if not results_list:
        return JSONResponse(status_code=200, content={})

    pred = {
        'result': results_list,
        'score': sum(scores) / len(scores),
        'model_version': model_version,
    }

    return JSONResponse(status_code=200, content=pred)


# ----------------------------------------------------------------------------

if __name__ == '__main__':
    load_dotenv()
    args = opts()

    WEIGHTS_PATH = args.weights
    MODEL_VERSION = args.model_version
    IMAGE_DIR = args.image_dir
    MD_API_URL = (args.md_api_url
                  or os.environ.get('MD_API_URL', '')).rstrip('/')

    threading.Thread(target=memory_manager, daemon=True).start()

    uvicorn.run(app, host=args.host, port=args.port)
