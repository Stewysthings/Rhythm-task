# generate_structured_synthetic_data.py

import json
import random

task_categories = [
    "cleaning", "communication", "cooking",
    "exercise", "relaxation", "shopping", "work"
]

structured_patterns = {
    "cleaning": {"days": [5, 6], "hours": range(9, 18)},
    "communication": {"days": range(7), "hours": range(8, 22)},
    "cooking": {"days": range(7), "hours": range(17, 21)},
    "exercise": {"days": range(7), "hours": range(6, 10)},
    "relaxation": {"days": [5, 6], "hours": range(18, 23)},
    "shopping": {"days": [5, 6], "hours": range(11, 16)},
    "work": {"days": range(0, 5), "hours": range(9, 17)},
}

synthetic_data = []

for _ in range(2000):
    task_category = random.choice(task_categories)
    # Choose hour & day according to patterns
    pattern = structured_patterns[task_category]
    day_of_week = random.choice(pattern["days"])
    hour = random.choice(pattern["hours"])
    # Randomly decide if it was done
    done = 1 if random.random() > 0.2 else 0  # 80% chance of done

    log_entry = {
        "task_category": task_category,
        "hour": hour,
        "day_of_week": day_of_week,
        "done": done
    }
    synthetic_data.append(log_entry)

# Save to JSON
with open("data/structured_synthetic_task_logs.json", "w") as f:
    json.dump(synthetic_data, f, indent=2)

print("Generated 2000 structured synthetic task log entries.")
