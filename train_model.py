import json
import torch
import torch.nn as nn
import torch.optim as optim
from data_preparation import load_and_prepare_data

# Load data
X_train, X_val, X_test, y_train, y_val, y_test, tasks, task_to_idx = load_and_prepare_data(filename="data/synthetic_task_logs.json")

# Define a deeper MLP model
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(MLP, self).__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))  # Prevent overfitting
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# Hyperparameters
input_dim = X_train.shape[1]
hidden_dims = [64, 32]  # Two hidden layers
output_dim = len(tasks)
num_epochs = 50
batch_size = 16
learning_rate = 0.001

# Model, loss, optimizer
model = MLP(input_dim, hidden_dims, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
def train():
    model.train()
    for i in range(0, len(X_train), batch_size):
        X_batch = X_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

def evaluate(X, y):
    model.eval()
    with torch.no_grad():
        outputs = model(X)
        loss = criterion(outputs, y)
        _, predicted = torch.max(outputs, 1)
        acc = (predicted == y).float().mean().item()
    return loss.item(), acc

# Main training loop
for epoch in range(num_epochs):
    train()
    val_loss, val_acc = evaluate(X_val, y_val)
    if epoch % 10 == 0 or epoch == num_epochs - 1:
        train_loss, _ = evaluate(X_train, y_train)
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

# Final test evaluation
test_loss, test_acc = evaluate(X_test, y_test)
print(f"Final Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

# Save model
torch.save(model.state_dict(), "task_model_deep.pth")
print("Training complete and model saved as 'task_model_deep.pth'.")
