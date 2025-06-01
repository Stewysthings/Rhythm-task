from datetime import datetime
import torch
import json
from model import RhythmNet

def suggest_task():
    with open("data/task_names.json", "r") as f:
        tasks = json.load(f)

    with open("data/task_log.json", "r") as f:
        task_log = json.load(f)

    # Compute minutes since last task
    if task_log:
        last_entry = task_log[-1]
        last_time = datetime.fromisoformat(last_entry["timestamp"])
        minutes_since_last = (datetime.now() - last_time).total_seconds() / 60.0
        # Normalize to a reasonable scale (e.g., divide by 1440, minutes in a day)
        minutes_since_last /= 1440.0
    else:
        # If no tasks have been done, use a default value (like 1)
        minutes_since_last = 1.0

    # Prepare input: hour, weekday, minutes_since_last
    now = datetime.now()
    hour = now.hour / 23.0
    day = now.weekday() / 6.0
    input_tensor = torch.tensor([[hour, day, minutes_since_last]], dtype=torch.float32)

    # Load model
    model = RhythmNet(input_size=3, hidden_size=16, output_size=len(tasks))
    model.load_state_dict(torch.load("data/task_model.pth"))
    model.eval()

    with torch.no_grad():
        output = model(input_tensor)
        predicted = torch.argmax(output, dim=1).item()

    return tasks[predicted]
