import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F

from models.combined_models import CombinedOutcomeHeadsModel, CombinedModel
from utils.custom_scheduler import CustomLRScheduler
from utils.test import test


class Client:
    """Client class to simulate the federated learning process using the FACT algorithm.

    Attributes:
        dataset_name: Name of the dataset
        name: A string to describe the client (usually the domain name).
        train_data: A dataset to use for training.
        G: Feature extractor
        C1: Classification head (only for target clients)
        C2: Classification head
        epochs: Number of epochs to train the model in each round.
        rounds: Number of rounds to repeate the training process.
        lr: Learning rate for the optimizer.
        device: GPU or CPU
        batch_size: Batch size for training.
        lr_decay_rate: Decay rate for the learning rate scheduler.
        test_data: Dataset to test the model of the client.
        source: boolean that indicates if client is source client or not
    """

    def __init__(
            self,
            dataset_name,
            name,
            train_data,
            G,
            C2,
            epochs,
            rounds,
            lr,
            device,
            batch_size,
            lr_decay_rate=0.75,
            test_data=None,
            C1=None,
            source=False,
    ):
        self.dataset_name = dataset_name
        self.name = name
        self.train_data = train_data
        self.test_data = test_data
        self.G = G
        self.C1 = C1
        self.C2 = C2
        self.epochs = epochs
        self.rounds = rounds
        self.device = device
        self.batch_size = batch_size
        self.opt_g = None
        self.opt_c1 = None
        self.opt_c2 = None
        self.scheduler_g = None
        self.scheduler_c2 = None
        self.source = source
        self.set_optimizer(lr=lr)
        self.set_scheduler(which_scheduler="custom", gamma=0.9, decay_rate=lr_decay_rate)
        self.accuracy = []
        self.loss = []
        self.loss_dis = []

    def train_at_client(self, g_state_dict, c2_state_dict, c1_state_dict=None):
        print(f"Training at client: {self.name}")

        self.G.load_state_dict(g_state_dict)
        self.C2.load_state_dict(c2_state_dict)
        if c1_state_dict:
            self.C1.load_state_dict(c1_state_dict)

        if self.source:
            self._train_at_source_data()
        else:
            self._train_at_target_data()

        if self.source:
            return self.G.state_dict(), self.C2.state_dict()
        else:
            return self.G.state_dict(), self.C1.state_dict(), self.C2.state_dict()

    def _train_at_source_data(self):
        loss_fn = nn.CrossEntropyLoss().to(self.device)

        self.G.train()
        self.C2.train()

        for epoch in range(self.epochs):
            for batch, data in enumerate(self.train_data):
                if type(data) == dict:
                    X = data['img']
                    y = data['label']
                else:
                    X, y = data
                X = Variable(X.to(self.device))
                y = Variable(y.long().to(self.device))
                # minimize cross entropy loss on source data
                feat_s = self.G(X)
                output_s2 = self.C2(feat_s)
                loss_s = loss_fn(output_s2, y)
                loss_s.backward()
                self.opt_g.step()
                self.opt_c2.step()
                self.reset_grad()

            self.scheduler_step()
            if self.test_data is not None:
                self.test_classifiers(loss_fn)

    def _train_at_target_data(self):
        loss_fn = nn.CrossEntropyLoss().to(self.device)

        for epoch in range(self.epochs):
            self.G.train()
            self.C1.train()
            self.C2.train()

            loss_dis_sum = 0.0
            for batch, data in enumerate(self.train_data):
                if type(data) == dict:
                    X = data['img']
                else:
                    X, _ = data
                X = Variable(X.to(self.device))

                # update feature generator G to minimize discrepancy on target data
                feat_t = self.G(X)
                output_t1 = self.C1(feat_t)
                output_t2 = self.C2(feat_t)
                loss_dis = self.discrepancy(output_t1, output_t2)
                loss_dis.backward()
                self.opt_g.step()
                self.reset_grad()
                loss_dis_sum += loss_dis.detach().cpu().numpy()

            self.loss_dis.append(loss_dis_sum)

            print(f"learning rate: {round(self.scheduler_g.get_last_lr()[0], 6)}")
            self.scheduler_step()

            if self.test_data is not None:
                self.test_classifiers(loss_fn)

    def finetune_classifiers(self, g_state_dict, epochs):
        loss_fn = nn.CrossEntropyLoss()
        self.G.load_state_dict(g_state_dict)
        self.C2.train()

        for epoch in range(epochs):
            for batch, data in enumerate(self.train_data):
                if type(data) == dict:
                    X = data['img']
                    y = data['label']
                else:
                    X, y = data
                X = Variable(X.to(self.device))
                y = Variable(y.long().to(self.device))

                feat_s = self.G(X)
                output_s2 = self.C2(feat_s)
                loss_s = loss_fn(output_s2, y)
                loss_s.backward()
                self.opt_c2.step()
                self.reset_grad()
        return self.C2.state_dict()

    def test_classifiers(self, loss_fn):
        if self.C1 is not None:
            current_accuracy, current_loss = test(
                self.test_data, CombinedOutcomeHeadsModel(self.G, self.C1, self.C2), loss_fn
            )
        else:
            current_accuracy, current_loss = test(
                self.test_data, CombinedModel(self.G, self.C2), loss_fn
            )
        self.accuracy.append(current_accuracy)
        self.loss.append(current_loss)

    @staticmethod
    def discrepancy(out1, out2):
        discrepancy = torch.mean(torch.abs(F.softmax(out1, dim=1) - F.softmax(out2, dim=1)))
        return discrepancy

    def set_optimizer(self, lr=0.001, momentum=0.9):
        self.opt_g = optim.SGD(
            self.G.parameters(), lr=lr, weight_decay=0.0005, momentum=momentum
        )
        self.opt_c2 = optim.SGD(
            self.C2.parameters(), lr=lr, weight_decay=0.0005, momentum=momentum
        )
        if self.C1 is not None:
            self.opt_c1 = optim.SGD(
                self.C1.parameters(), lr=lr, weight_decay=0.0005, momentum=momentum
            )

    def set_scheduler(self, which_scheduler="custom", gamma=0.9, decay_rate=0.75):
        if which_scheduler == "custom":
            self.scheduler_g = CustomLRScheduler(
                self.opt_g, total_epochs=self.epochs*self.rounds, decay_rate=decay_rate)
            if self.source:
                self.scheduler_c2 = CustomLRScheduler(
                    self.opt_c2, total_epochs=self.epochs*self.rounds, decay_rate=decay_rate)
        if which_scheduler == "exponential":
            self.scheduler_g = optim.lr_scheduler.ExponentialLR(self.opt_g, gamma=gamma)
            if self.source:
                self.scheduler_c2 = optim.lr_scheduler.ExponentialLR(self.opt_c2, gamma=gamma)

    def reset_grad(self):
        self.opt_g.zero_grad()
        self.opt_c2.zero_grad()
        if self.C1 is not None:
            self.opt_c1.zero_grad()

    def scheduler_step(self):
        self.scheduler_g.step()
        if self.source:
            self.scheduler_c2.step()
