import json
import matplotlib.pyplot as plt

def plot_training_logs(filename="data/training_logs.json"):
    # Load training logs
    with open(filename, "r") as f:
        logs = json.load(f)

    # Validate logs
    if not isinstance(logs, dict):
        raise ValueError("Logs file must contain a JSON object (dictionary) with 'train_losses', 'val_losses', and 'val_accuracies'.")

    required_keys = ["train_losses", "val_losses", "val_accuracies"]
    for key in required_keys:
        if key not in logs:
            raise KeyError(f"Missing key in logs: {key}")

    train_losses = logs["train_losses"]
    val_losses = logs["val_losses"]
    val_accuracies = logs["val_accuracies"]

    if not train_losses or not val_losses or not val_accuracies:
        print("Warning: One or more log lists are empty. Plots may not show meaningful data.")

    # Plotting
    epochs = range(len(train_losses))
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss", color="blue")
    plt.plot(epochs, val_losses, label="Validation Loss", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, label="Validation Accuracy", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_training_logs()
