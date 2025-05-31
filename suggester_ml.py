# suggester_ml.py
import torch
import json
from datetime import datetime
from model import TaskPredictor

MODEL_PATH = "data/task_model.pth"
TASK_NAMES_PATH = "data/task_names.json"

def load_model():
    model = TaskPredictor(input_size=2, hidden_size=16, output_size=0)  # output_size will be fixed later
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    return model

def suggest_task():
    # Load model and task names
    model = load_model()
    with open(TASK_NAMES_PATH, "r") as f:
        task_names = json.load(f)

    now = datetime.now()
    input_tensor = torch.tensor([[now.weekday(), now.hour]], dtype=torch.float32)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        top3_indices = torch.topk(probabilities, k=3).indices.squeeze().tolist()

    suggestions = [task_names[str(i)] for i in top3_indices]
    return suggestions
