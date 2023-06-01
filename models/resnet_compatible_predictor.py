import torch
import torch.nn as nn
import torch.nn.functional as F


class Predictor(nn.Module):
    def __init__(self, classes, prob=0.5):
        super(Predictor, self).__init__()
        self.fc1 = nn.Linear(1000, 500)
        self.bn1_fc = nn.BatchNorm1d(500)
        self.fc2 = nn.Linear(500, classes)
        self.bn2_fc = nn.BatchNorm1d(classes)
        self.softmax = torch.nn.Softmax(dim=1)
        self.prob = prob

    def forward(self, x):
        x = F.dropout(x, training=self.training, p=self.prob)
        x = F.relu(self.bn1_fc(self.fc1(x)))
        x = F.relu(self.bn2_fc(self.fc2(x)))
        x = self.softmax(x)
        return x