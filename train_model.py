import torch
import torch.nn as nn
import torch.optim as optim
from model import RhythmNet
from data_preparation import load_and_prepare_data
import json

# Load data (including validation split)
X_train, X_val, X_test, y_train, y_val, y_test, tasks = load_and_prepare_data()

# Initialize model
model = RhythmNet(input_size=3, hidden_size=16, output_size=len(tasks))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)


# Training loop
for epoch in range(200):
    # Training step
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f"Epoch {epoch}: Train loss={loss.item():.4f}")

        # Validation step
        with torch.no_grad():
            model.eval()
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)
            print(f"Epoch {epoch}: Validation loss={val_loss.item():.4f}")
            model.train()

# Final test evaluation
with torch.no_grad():
    model.eval()
    test_outputs = model(X_test)
    test_loss = criterion(test_outputs, y_test)
    print(f"Final Test loss={test_loss.item():.4f}")

# Save model and task names
torch.save(model.state_dict(), "data/task_model.pth")
with open("data/task_names.json", "w") as f:
    json.dump(tasks, f)

