import torch.nn as nn

class RhythmNet(nn.Module):
    def __init__(self, input_size=3, hidden_size=16, output_size=10):
        super(RhythmNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

