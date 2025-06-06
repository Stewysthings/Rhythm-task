import json
import random
tasks = ['cleaning', 'communication', 'cooking', 'exercise', 'relaxation', 'shopping', 'work']
data = []
for _ in range(10000):
    task = random.choice(tasks)
    if task == 'cooking':
        hour = random.choice([17, 18, 19, 20])
        day = random.randint(0, 6)
    elif task == 'work':
        hour = random.choice([9, 10, 11, 12, 13, 14, 15])
        day = random.randint(0, 4)
    elif task == 'exercise':
        hour = random.choice([6, 7, 8, 17, 18])
        day = random.randint(0, 6)
    elif task == 'relaxation':
        hour = random.choice([20, 21, 22])
        day = random.randint(0, 6)
    elif task == 'shopping':
        hour = random.choice([10, 11, 14, 15, 16])
        day = random.randint(4, 6)
    elif task == 'cleaning':
        hour = random.choice([8, 9, 10, 11])
        day = random.randint(4, 6)
    elif task == 'communication':
        hour = random.choice([8, 9, 16, 17])
        day = random.randint(0, 6)
    done = random.randint(0, 1)
    data.append({"hour": hour, "day_of_week": day, "task_category": task, "done": done})
with open("data/improved_synthetic_task_logs.json", "w") as f:
    json.dump(data, f)