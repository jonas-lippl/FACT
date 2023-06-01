import torch
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"


class CombinedModel(nn.Module):
    def __init__(self, global_model, local_model):
        super(CombinedModel, self).__init__()
        self.global_model = global_model
        self.local_model = local_model

    def train(self, mode: bool = True):
        self.global_model.train(mode)
        self.local_model.train(mode)

    def eval(self):
        self.global_model.eval()
        self.local_model.eval()

    def forward(self, x):
        return self.local_model(self.global_model(x))


class CombinedOutcomeHeadsModel(nn.Module):
    def __init__(self, G, C1, C2):
        super(CombinedOutcomeHeadsModel, self).__init__()
        self.G = G
        self.C1 = C1
        self.C2 = C2

    def train(self, mode: bool = True):
        self.G.train(mode)
        self.C1.train(mode)
        self.C2.train(mode)

    def eval(self):
        self.G.eval()
        self.C1.eval()
        self.C2.eval()

    def forward(self, x):
        feature = self.G(x)
        pred1 = self.C1(feature)
        pred2 = self.C2(feature)
        result = torch.tensor([]).to(device)
        for i in range(len(pred1)):
            if pred1[i].max() >= pred2[i].max():
                result = torch.cat((result, pred1[i]), dim=0)
            else:
                result = torch.cat((result, pred2[i]), dim=0)
        result = result.reshape((len(pred1), len(pred1[0])))
        return result