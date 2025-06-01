# task_logger.py
import json
import os
from datetime import datetime

DATA_PATH = "data/tasks.json"

# Helper function to safely load tasks
def load_tasks_file(filename):
    try:
        with open(filename, "r") as f:
            data = json.load(f)
            if isinstance(data, list):
                # Only keep strings or dictionaries (avoid corruption)
                data = [t for t in data if isinstance(t, (str, dict))]
            else:
                data = []
            return data
    except (json.JSONDecodeError, FileNotFoundError):
        # If file is corrupted or not found, reset to an empty list
        return []

def log_task(task_name: str):
    now = datetime.now()
    task_entry = {
        "task": task_name,
        "day_of_week": now.weekday(),  # Monday=0, Sunday=6
        "hour": now.hour
    }

    # Load existing tasks safely
    tasks = load_tasks_file(DATA_PATH)
    tasks.append(task_entry)

    # Save back
    with open(DATA_PATH, "w") as f:
        json.dump(tasks, f, indent=2)

def add_task(new_task):
    # Load existing tasks safely
    tasks = load_tasks_file(DATA_PATH)
    if new_task not in tasks:
        tasks.append(new_task)
        with open(DATA_PATH, "w") as f:
            json.dump(tasks, f, indent=2)
    else:
        print("Task already exists!")



