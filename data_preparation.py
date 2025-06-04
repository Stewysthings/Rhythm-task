import json
from sklearn.model_selection import train_test_split
import torch
from collections import Counter

def load_and_prepare_data(filename="data/task_logs.json"):
    with open(filename, "r") as f:
        logs = json.load(f)

    # Extract unique tasks and create mapping
    tasks = sorted(set(log["task_category"] for log in logs))  # Sorted for consistency
    task_to_idx = {task: i for i, task in enumerate(tasks)}

    X = []
    y = []
    prev_time = None

    for log in logs:
        hour = log["hour"] / 23.0  # Normalize hour to [0,1]
        day = log["day_of_week"] / 6.0  # Normalize day to [0,1]

        # Calculate minutes since last task (considering wrap-around a week)
        current_time = log["day_of_week"] * 1440 + log["hour"] * 60
        if prev_time is None:
            minutes_since_last = 0
        else:
            diff = current_time - prev_time
            if diff < 0:
                diff += 7 * 1440  # Wrap-around for weekly cycle
            minutes_since_last = diff
        prev_time = current_time

        minutes_since_last_norm = minutes_since_last / 1440  # Normalize to days

        X.append([hour, day, minutes_since_last_norm])
        y.append(task_to_idx[log["task_category"]])

    # Filter out classes with fewer than 2 samples
    label_counts = Counter(y)
    valid_labels = {label for label, count in label_counts.items() if count >= 2}

    X_filtered = []
    y_filtered = []
    for features, label in zip(X, y):
        if label in valid_labels:
            X_filtered.append(features)
            y_filtered.append(label)

    print(f"Number of classes after filtering: {len(valid_labels)}")
    print(f"Total samples after filtering: {len(X_filtered)}")

    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_filtered, y_filtered, test_size=0.2, random_state=42, stratify=y_filtered
    )

    # Further split train into train/validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
    )

    # Convert lists to torch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # Final counts for verification
    print(f"Train size: {len(X_train)}, Validation size: {len(X_val)}, Test size: {len(X_test)}")
    print(f"Unique tasks in y_train: {torch.unique(y_train)}")

    # Return all datasets + tasks list + task_to_idx dictionary
    return X_train, X_val, X_test, y_train, y_val, y_test, tasks, task_to_idx
