import json
from sklearn.model_selection import train_test_split
import torch

def load_and_prepare_data(filename="data/task_logs.json"):
    with open(filename, "r") as f:
        logs = json.load(f)

    # Extract unique tasks and create mapping
    tasks = list(set([log["task"] for log in logs]))
    task_to_idx = {task: i for i, task in enumerate(tasks)}

    # Prepare input (X) and target (y)
    X = []
    y = []
    prev_time = None

    for log in logs:
        hour = log["hour"] / 23.0
        day = log["day_of_week"] / 6.0
        current_time = log["day_of_week"] * 1440 + log["hour"] * 60
        if prev_time is None:
            minutes_since_last = 0
        else:
            diff = current_time - prev_time
            if diff < 0:
                diff += 7 * 1440
            minutes_since_last = diff
        prev_time = current_time
        minutes_since_last_norm = minutes_since_last / 1440

        X.append([hour, day, minutes_since_last_norm])
        y.append(task_to_idx[log["task"]])

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Split train into train + validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42
    )

    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # Print sizes for sanity check
    print("Train size:", len(X_train))
    print("Validation size:", len(X_val))
    print("Test size:", len(X_test))

    return X_train, X_val, X_test, y_train, y_val, y_test, tasks
