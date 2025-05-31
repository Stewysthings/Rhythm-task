# task_logger.py
import json
import os
from datetime import datetime

DATA_PATH = "data/tasks.json"

def log_task(task_name: str):
    now = datetime.now()
    task_entry = {
        "task": task_name,
        "day_of_week": now.weekday(),  # Monday=0, Sunday=6
        "hour": now.hour
    }

    # Load existing tasks
    if os.path.exists(DATA_PATH):
        with open(DATA_PATH, "r") as f:
            tasks = json.load(f)
    else:
        tasks = []

    tasks.append(task_entry)

    # Save back
    with open(DATA_PATH, "w") as f:
        json.dump(tasks, f, indent=2)
