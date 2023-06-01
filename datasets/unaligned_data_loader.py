import torch.utils.data
from builtins import object
import torchvision.transforms as transforms

from datasets.datasets_ import Dataset


class PairedData(object):
    def __init__(self, data_loader_A, data_loader_B, max_dataset_size):
        self.data_loader_A = data_loader_A
        self.data_loader_B = data_loader_B
        self.stop_A = False
        self.stop_B = False
        self.max_dataset_size = max_dataset_size

    def __iter__(self):
        self.stop_A = False
        self.stop_B = False
        self.data_loader_A_iter = iter(self.data_loader_A)
        self.data_loader_B_iter = iter(self.data_loader_B)
        self.iter = 0
        return self

    def __next__(self):
        A, A_paths = None, None
        B, B_paths = None, None
        try:
            A, A_paths = next(self.data_loader_A_iter)
        except StopIteration:
            if A is None or A_paths is None:
                self.stop_A = True
                self.data_loader_A_iter = iter(self.data_loader_A)
                A, A_paths = next(self.data_loader_A_iter)

        try:
            B, B_paths = next(self.data_loader_B_iter)
        except StopIteration:
            if B is None or B_paths is None:
                self.stop_B = True
                self.data_loader_B_iter = iter(self.data_loader_B)
                B, B_paths = next(self.data_loader_B_iter)

        if (self.stop_A and self.stop_B) or self.iter > self.max_dataset_size:
            self.stop_A = False
            self.stop_B = False
            raise StopIteration()
        else:
            self.iter += 1
            return {'S': A, 'S_label': A_paths,
                    'T': B, 'T_label': B_paths}


class UnalignedDataLoader():
    def initialize(self, source, target, batch_size1, batch_size2, scale=32):
        transform = transforms.Compose([
            transforms.Resize(scale),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset_source = Dataset(source['imgs'], source['labels'], transform=transform)
        dataset_target = Dataset(target['imgs'], target['labels'], transform=transform)
        data_loader_s = torch.utils.data.DataLoader(
            dataset_source,
            batch_size=batch_size1,
            shuffle=True,
            num_workers=4)

        data_loader_t = torch.utils.data.DataLoader(
            dataset_target,
            batch_size=batch_size2,
            shuffle=True,
            num_workers=4)
        self.dataset_s = dataset_source
        self.dataset_t = dataset_target
        self.paired_data = PairedData(data_loader_s, data_loader_t,
                                      float("inf"))

    def name(self):
        return 'UnalignedDataLoader'

    def load_data(self):
        return self.paired_data

    def __len__(self):
        return min(max(len(self.dataset_s), len(self.dataset_t)), float("inf"))


class fdaData(object):
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.stop_A = False

    def __iter__(self):
        self.stop_A = False
        self.data_loader_iter = iter(self.data_loader)
        self.iter = 0
        return self

    def __next__(self):
        A, A_paths = None, None
        try:
            A, A_paths = next(self.data_loader_iter)
        except StopIteration:
            if A is None or A_paths is None:
                self.stop_A = True
                self.data_loader_iter = iter(self.data_loader)
                A, A_paths = next(self.data_loader_iter)
        if self.stop_A:
            self.stop_A = False
            raise StopIteration()
        else:
            self.iter += 1
            return {'img': A, 'label': A_paths}

    def __len__(self):
        return len(self.data_loader)


class fda_DataLoader():
    def initialize(self, S, batch_size, scale=32):
        transform = transforms.Compose([
            transforms.Resize(scale),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset = Dataset(S['imgs'], S['labels'], transform=transform)
        self.dataset = dataset
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.fda_data = fdaData(data_loader)

    def name(self):
        return 'fda_DataLoader'

    def load_data(self):
        return self.fda_data

    def __len__(self):
        return (len(self.dataset))
