import pandas as pd
import json  # Add this line
with open("data/structured_synthetic_task_logs.json", "r") as f:
    logs = json.load(f)
df = pd.DataFrame(logs)
print(df.groupby("task_category")[["hour", "day_of_week", "done"]].mean())
print(df["task_category"].value_counts())