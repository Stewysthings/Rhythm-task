from fastapi import FastAPI
from pydantic import BaseModel
import torch
from train_model import TaskPredictor, tasks

app = FastAPI()

class TaskInput(BaseModel):
    hour: float
    day_of_week: float
    done: int

model = TaskPredictor(input_size=3, hidden_size=128, num_classes=7)
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

@app.post("/predict")
async def predict_task(input: TaskInput):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    features = torch.tensor([[input.hour/23.0, input.day_of_week/6.0, input.done]], dtype=torch.float32).to(device)
    with torch.no_grad():
        output = model(features)
        _, predicted = torch.max(output, 1)
    return {"predicted_task": tasks[predicted.item()], "hour": input.hour, "day_of_week": input.day_of_week}