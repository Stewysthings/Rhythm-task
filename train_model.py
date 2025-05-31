# train_model.py
import torch
from model import TaskPredictor
import json

# Load data
with open("data/tasks.json") as f:
    tasks = json.load(f)

# Encode days, hours, and task labels
day_to_idx = {d: i for i, d in enumerate(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])}
task_names = list(set(t["task"] for t in tasks))
task_to_idx = {name: i for i, name in enumerate(task_names)}

X, y = [], []
for t in tasks:
    day_idx = day_to_idx[t["day"]]
    hour = t["hour"]
    input_vec = [0]*7
    input_vec[day_idx] = 1
    input_vec.append(hour)
    X.append(input_vec)
    y.append(task_to_idx[t["task"]])

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# Initialize model
model = TaskPredictor(input_dim=8, num_tasks=len(task_names))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(200):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        print(f\"Epoch {epoch}, Loss: {loss.item():.4f}\")

# Save model and mappings
torch.save(model.state_dict(), \"data/task_model.pth\")
with open(\"data/task_names.json\", \"w\") as f:
    json.dump(task_names, f)
