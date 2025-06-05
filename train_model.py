import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import json
import numpy as np

# Load data
def load_and_prepare_data(filename="data/synthetic_task_logs.json"):
    with open(filename, "r") as f:
        logs = json.load(f)

    # Extract features and labels
    hours = [log["hour"] for log in logs]
    days = [log["day_of_week"] for log in logs]
    categories = [log["task_category"] for log in logs]
    done_flags = [log["done"] for log in logs]

    # One-hot encode categories
    unique_tasks = sorted(set(categories))
    task_to_idx = {task: idx for idx, task in enumerate(unique_tasks)}
    y = [task_to_idx[cat] for cat in categories]

    X = np.stack([hours, days, done_flags], axis=1)

    # Train/val/test split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.2, stratify=y_temp, random_state=42)

    print("Number of classes after filtering:", len(unique_tasks))
    print("Total samples after filtering:", len(X))
    print(f"Train size: {len(X_train)}, Validation size: {len(X_val)}, Test size: {len(X_test)}")
    print("Unique tasks in y_train:", torch.unique(torch.tensor(y_train)))
    print("Classes:", unique_tasks)
    return (torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
            torch.tensor(y_val, dtype=torch.long),
            torch.tensor(y_test, dtype=torch.long),
            unique_tasks, task_to_idx)

# Model
class TaskPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(TaskPredictor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# Training function
def train_model(X_train, y_train, X_val, y_val, num_classes, num_epochs=100, batch_size=64, lr=0.0005):
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TaskPredictor(input_size=3, hidden_size=128, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch}: Train Loss={train_loss/len(train_loader):.4f}, "
                  f"Val Loss={val_loss/len(val_loader):.4f}, Val Acc={val_acc:.4f}")

    return model

# Evaluation
def evaluate_model(model, X_test, y_test):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    X_test, y_test = X_test.to(device), y_test.to(device)
    with torch.no_grad():
        outputs = model(X_test)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, y_test)
        _, predicted = torch.max(outputs, 1)
        acc = (predicted == y_test).float().mean().item()
    print(f"Final Test Loss: {loss.item():.4f}, Test Accuracy: {acc:.4f}")


# Main
if __name__ == "__main__":
   X_train, X_val, X_test, y_train, y_val, y_test, tasks, task_to_idx = load_and_prepare_data(filename="data/structured_synthetic_task_logs.json")

 # Call train_model and get the trained model
   model = train_model(X_train, y_train, X_val, y_val, num_classes=len(tasks), num_epochs=100, batch_size=64, lr=0.0005)


torch.save(model.state_dict(), "task_model_deep.pth")
print("Training complete and model saved as 'task_model_deep.pth'.")
evaluate_model(model, X_test, y_test)
