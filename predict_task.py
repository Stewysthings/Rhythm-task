import torch
import datetime
import numpy as np
from train_model import TaskPredictor, tasks

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TaskPredictor(input_size=5, hidden_size=128, num_classes=7).to(device)
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

now = datetime.datetime.now()
hour = now.hour
day_of_week = now.weekday()
input_tensor = torch.tensor([[
    np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
    np.sin(2 * np.pi * day_of_week / 7), np.cos(2 * np.pi * day_of_week / 7),
    0
]], dtype=torch.float32).to(device)

with torch.no_grad():
    output = model(input_tensor)
    predicted_class = torch.argmax(output, dim=1).item()

print(f"Suggested task for now ({int(now.hour)}:00 on day {now.weekday()}): {tasks[predicted_class]}")