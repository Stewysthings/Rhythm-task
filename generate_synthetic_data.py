import json
import random

def generate_synthetic_data(num_samples=2000, filename="data/synthetic_task_logs.json"):
    task_categories = ["cooking", "cleaning", "exercise", "work", "shopping", "communication", "relaxation"]
    synthetic_data = []

    for _ in range(num_samples):
        task_category = random.choice(task_categories)
        hour = random.randint(0, 23)
        day_of_week = random.randint(0, 6)
        done = random.choices([0, 1], weights=[0.4, 0.6])[0]  # 60% chance done, 40% not done

        synthetic_data.append({
            "task_category": task_category,
            "hour": hour,
            "day_of_week": day_of_week,
            "done": done
        })

    with open(filename, "w") as f:
        json.dump(synthetic_data, f, indent=2)

    print(f"Generated {num_samples} synthetic task log entries in '{filename}'.")

if __name__ == "__main__":
    generate_synthetic_data()
