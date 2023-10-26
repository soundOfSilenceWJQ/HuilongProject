import torch
import torch.nn as nn


class RankNetModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super(RankNetModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x1, x2):
        o1 = self.fc(x1).squeeze()
        o2 = self.fc(x2).squeeze()
        diff = o1 - o2
        prob = torch.sigmoid(diff)
        return prob
