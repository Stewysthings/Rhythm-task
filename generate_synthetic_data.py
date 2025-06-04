import json
import random

def generate_synthetic_task_logs(num_entries=200):
    task_categories = [
        "exercise",
        "cleaning",
        "work",
        "shopping",
        "relaxation",
        "cooking",
        "communication",
    ]

    synthetic_logs = []

    for _ in range(num_entries):
        # Randomly pick a task category
        task = random.choice(task_categories)
        
        # Random hour (0-23)
        hour = random.randint(0, 23)
        
        # Random day of week (0=Monday, 6=Sunday)
        day_of_week = random.randint(0, 6)

        # Simulate if task was done or not (1 or 0)
        # Let's say tasks are more likely done during specific hours for realism
        if task == "exercise":
            done = 1 if 6 <= hour <= 9 else 0
        elif task == "work":
            done = 1 if 9 <= hour <= 17 else 0
        elif task == "cleaning":
            done = 1 if 10 <= hour <= 14 else 0
        elif task == "relaxation":
            done = 1 if 18 <= hour <= 22 else 0
        else:
            # random for other tasks
            done = random.choice([0, 1])

        log_entry = {
            "task_category": task,
            "hour": hour,
            "day_of_week": day_of_week,
            "done": done
        }

        synthetic_logs.append(log_entry)

    # Save to JSON file for your app to use
    with open("synthetic_task_logs.json", "w") as f:
        json.dump(synthetic_logs, f, indent=2)

    print(f"Generated {num_entries} synthetic task log entries.")

if __name__ == "__main__":
    generate_synthetic_task_logs()
