# model.py
import torch
import torch.nn as nn

class TaskPredictor(nn.Module):
    def __init__(self, input_dim, num_tasks):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, num_tasks)
        )
        
    def forward(self, x):
        return self.net(x)