#!/usr/bin/env python

import argparse
import os
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv
from tqdm import tqdm


def get_tasks(ls_url, project_id, headers):
    tqdm.write(f"Fetching tasks for project {project_id}...")
    export_url = f"{ls_url}/api/projects/{project_id}/export?exportType=JSON&download_all_tasks=true"
    response = requests.get(export_url, headers=headers)
    response.raise_for_status()
    return response.json()


def get_predictions_by_task(ls_url, project_id, headers):
    """Fetch all predictions for a project; return dict keyed by task id."""
    tqdm.write("Fetching existing predictions...")
    predictions: dict[int, list] = {}
    page = 1
    while True:
        resp = requests.get(
            f"{ls_url}/api/predictions",
            params={"project": project_id, "page": page, "page_size": 1000},
            headers=headers,
        )
        resp.raise_for_status()
        data = resp.json()
        results = data if isinstance(data, list) else data.get("results", [])
        for pred in results:
            tid = pred["task"]
            predictions.setdefault(tid, []).append(pred)
        if isinstance(data, list) or not data.get("next"):
            break
        page += 1
    return predictions


def is_cancelled(task):
    if task.get('cancelled_annotations', 0) > 0:
        return True
    return any(a.get('was_cancelled', False) for a in task.get('annotations', []))


def real_annotations(task):
    return [a for a in task.get('annotations', []) if not a.get('was_cancelled', False)]


def delete_prediction(ls_url, pred_id, headers):
    resp = requests.delete(f"{ls_url}/api/predictions/{pred_id}", headers=headers)
    if resp.status_code not in (200, 204):
        tqdm.write(f"  Warning: failed to delete prediction {pred_id}: {resp.text}")


def main(args, api_key):
    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": "application/json"
    }

    tasks = get_tasks(args.ls_url, args.project_id, headers)
    predictions_by_task = get_predictions_by_task(args.ls_url, args.project_id, headers)

    after_date = None
    if args.after_date:
        after_date = datetime.strptime(args.after_date, "%d-%m-%Y").replace(tzinfo=timezone.utc)

    # Build work list: (task, stale_prediction_ids_to_delete)
    target_tasks = []
    skipped_cancelled = 0
    skipped_annotated = 0
    skipped_current_model = 0
    skipped_date = 0

    for task in tasks:
        if after_date:
            created_at = task.get('created_at', '')
            if not created_at or datetime.fromisoformat(created_at.replace('Z', '+00:00')) < after_date:
                skipped_date += 1
                continue
        if is_cancelled(task):
            skipped_cancelled += 1
            continue
        if real_annotations(task):
            skipped_annotated += 1
            continue

        predictions = predictions_by_task.get(task['id'], [])
        stale = [p for p in predictions if p.get('model_version') != args.model_version]
        current = [p for p in predictions if p.get('model_version') == args.model_version]

        if current:
            skipped_current_model += 1
            continue

        target_tasks.append((task, stale))

    tqdm.write(
        f"Tasks: {len(tasks)} total | "
        + (f"{skipped_date} before {args.after_date} | " if after_date else "")
        + f"{skipped_annotated} annotated | "
        f"{skipped_cancelled} cancelled | "
        f"{skipped_current_model} already have '{args.model_version}' predictions | "
        f"{len(target_tasks)} to process"
    )

    if not target_tasks:
        tqdm.write("Nothing to do. Exiting.")
        return

    for task, stale_preds in tqdm(target_tasks, desc="Generating predictions", unit="task"):
        task_id = task["id"]

        # Remove stale predictions from a different model
        for pred in stale_preds:
            tqdm.write(f"  Deleting stale prediction {pred['id']} (model: {pred.get('model_version')}) on task {task_id}")
            delete_prediction(args.ls_url, pred['id'], headers)

        # Request new prediction
        payload = {"task": task, "project": args.project_id}
        try:
            api_resp = requests.post(args.ml_api_url, json=payload)

            if api_resp.status_code != 200:
                tqdm.write(f"  API error on task {task_id}: {api_resp.status_code} - {api_resp.text}")
                continue

            pred_data = api_resp.json()
            if not pred_data or not pred_data.get("result"):
                continue

            pred_data["task"] = task_id

            ls_resp = requests.post(f"{args.ls_url}/api/predictions",
                                    headers=headers,
                                    json=pred_data)
            if ls_resp.status_code not in (200, 201):
                tqdm.write(f"  Label Studio error on task {task_id}: {ls_resp.text}")

        except Exception as e:
            tqdm.write(f"  Exception on task {task_id}: {e}")


if __name__ == "__main__":
    load_dotenv()

    api_key = os.environ.get("LABEL_STUDIO_TOKEN")
    if not api_key:
        raise ValueError("LABEL_STUDIO_TOKEN must be set in the .env file")

    parser = argparse.ArgumentParser(
        description="Send unannotated tasks to a YOLO ML endpoint for predictions.")
    parser.add_argument("--ls-url", type=str, required=True,
                        help="Label Studio instance URL")
    parser.add_argument("--project-id", type=int, required=True,
                        help="Label Studio project ID")
    parser.add_argument("--ml-api-url", type=str, required=True,
                        help="ML API endpoint URL")
    parser.add_argument("--model-version", type=str, required=True,
                        help="Model version string to match against existing predictions")
    parser.add_argument("--after-date", type=str, default=None,
                        help="Only process tasks created on or after this date (format: DD-MM-YYYY)")

    args = parser.parse_args()
    main(args, api_key)
