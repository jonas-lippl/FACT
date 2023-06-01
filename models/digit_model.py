import torch
import torch.nn as nn
import torch.nn.functional as F


class Feature(nn.Module):
    def __init__(self):
        super(Feature, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(128)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), stride=2, kernel_size=3, padding=1)
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), stride=2, kernel_size=3, padding=1)
        x = x.view(x.size(0), 128 * 8 * 8)
        return x


class Predictor(nn.Module):
    def __init__(self, prob=0.5):
        super(Predictor, self).__init__()
        self.fc1 = nn.Linear(128 * 8 * 8, 3072)
        self.bn1_fc = nn.BatchNorm1d(3072)
        self.fc2 = nn.Linear(3072, 100)
        self.bn2_fc = nn.BatchNorm1d(100)
        self.fc3 = nn.Linear(100, 10)
        self.bn_fc3 = nn.BatchNorm1d(10)
        self.prob = prob
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = F.dropout(x, training=self.training, p=self.prob)
        x = F.relu(self.bn1_fc(self.fc1(x)))
        x = F.dropout(x, training=self.training, p=self.prob)
        x = F.relu(self.bn2_fc(self.fc2(x)))
        x = F.dropout(x, training=self.training, p=self.prob)
        x = self.fc3(x)
        x = self.softmax(x)
        return x
