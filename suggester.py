import json
import os

def suggest_task(task_list=None):
    if task_list is None:
        # Load tasks from JSON file
        file_path = os.path.join("data", "tasks.json")
        with open(file_path, "r") as f:
            task_list = json.load(f)
    # For now, just return the first task
    return task_list[0] if task_list else None
