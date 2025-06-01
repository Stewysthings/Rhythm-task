import torch
import torch.nn as nn
import torch.optim as optim
import json
from model import RhythmNet
from datetime import datetime

# Load task logs
with open("data/task_logs.json", "r") as f:
    logs = json.load(f)

# Extract unique tasks and create mapping
tasks = list(set([log["task"] for log in logs]))
task_to_idx = {task: i for i, task in enumerate(tasks)}

# Prepare input (X) and target (y)
X = []
y = []

prev_time = None  # Track previous task time in minutes since week start

for log in logs:
    hour = log["hour"] / 23.0  # Normalize hour 0-23 to 0-1
    day = log["day_of_week"] / 6.0  # Normalize day_of_week 0-6 to 0-1

    # Calculate minutes since last task
    current_time = log["day_of_week"] * 1440 + log["hour"] * 60  # total minutes since week start

    if prev_time is None:
        minutes_since_last = 0
    else:
        diff = current_time - prev_time
        # If negative (week wrap-around), add full week minutes
        if diff < 0:
            diff += 7 * 1440
        minutes_since_last = diff

    prev_time = current_time

    minutes_since_last_norm = minutes_since_last / 1440  # Normalize by minutes in a day

    X.append([hour, day, minutes_since_last_norm])
    y.append(task_to_idx[log["task"]])

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# Initialize model
model = RhythmNet(input_size=3, hidden_size=16, output_size=len(tasks))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(200):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        print(f"Epoch {epoch}: loss={loss.item():.4f}")

# Save model and task names
torch.save(model.state_dict(), "data/task_model.pth")
with open("data/task_names.json", "w") as f:
    json.dump(tasks, f)
