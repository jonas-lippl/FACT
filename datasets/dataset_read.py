from torch.utils.data import DataLoader, TensorDataset, Subset
from torchvision import datasets
from torchvision.transforms import transforms

from datasets.unaligned_data_loader import fda_DataLoader
from datasets.svhn import load_svhn
from datasets.mnist import load_mnist
from datasets.mnist_m import load_mnistm
from datasets.usps_ import load_usps
from datasets.syn import load_syn


def return_dataset(data, scale=False):
    if data == 'svhn':
        train_image, train_label, test_image, test_label = load_svhn()
    if data == 'mnist':
        train_image, train_label, test_image, test_label = load_mnist(scale=scale)
    if data == 'usps':
        train_image, train_label, test_image, test_label = load_usps()
    if data == 'mnistm':
        train_image, train_label, test_image, test_label = load_mnistm()
    if data == 'syn':
        train_image, train_label, test_image, test_label = load_syn()

    return train_image, train_label, test_image, test_label


def digit_five_dataset_read(domain, batch_size, device, scale=False, index_range=None):
    print(f"Load {domain}...")
    S = {}
    S_test = {}
    train_data, train_label, test_data, test_label = return_dataset(domain, scale=scale)
    S['imgs'] = train_data
    S['labels'] = train_label

    S_test['imgs'] = test_data
    S_test['labels'] = test_label
    scale = 32

    size_train = len(train_label)
    size_test = len(test_label)
    train_loader = fda_DataLoader()
    train_loader.initialize(S, size_train, scale)
    dataset = train_loader.load_data()
    test_loader = fda_DataLoader()
    test_loader.initialize(S_test, size_test, scale)
    dataset_test = test_loader.load_data()

    for x_train, y_train in dataset.data_loader:
        X_train = x_train.to(device)
        Y_train = y_train.to(device)
    dataset_train = TensorDataset(X_train, Y_train)
    if index_range is not None:
        print(f"Picking indices {index_range} from dataset")
        dataset_train = Subset(dataset_train, index_range)

    for x_test, y_test in dataset_test.data_loader:
        X_test = x_test.to(device)
        Y_test = y_test.to(device)
    dataset_test = TensorDataset(X_test, Y_test)

    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)

    return dataloader_train, dataloader_test


def office_datasets_read(domain: str, dataset_name: str, batch_size: int, index_range=None, device: str | None = None):
    if dataset_name == 'office':
        data_dir = f"data/{dataset_name}/{domain}/images"
    else:
        data_dir = f"data/{dataset_name}/{domain}"
    print(f"Load {data_dir}...")
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize([300, 300]),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset_lazy = datasets.ImageFolder(data_dir, transform)
    size = len(dataset_lazy)

    for x, y in DataLoader(dataset_lazy, batch_size=size):
        if device is not None:
            X = x.to(device)
            Y = y.to(device)
        else:
            X = x
            Y = y
    dataset = TensorDataset(X, Y)
    if index_range is not None:
        print(f"Picking indices {index_range} from dataset")
        dataset = Subset(dataset, index_range)
    drop_last = False
    if dataset_name == 'office' and domain == 'amazon':
        drop_last = True
    dataloader_train = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last)
    dataloader_test = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last)
    return dataloader_train, dataloader_test


def dataset_read(dataset: str, domain: str, batch_size: int, device: str, index_range: range | None = None):
    if dataset == 'digit-five':
        return digit_five_dataset_read(domain, batch_size, device, index_range=index_range)
    if dataset == 'office_caltech_10':
        return office_datasets_read(domain, dataset, batch_size, index_range, device)  # Remove device from here if GPU has not enough memory to store the whole dataset
    if dataset == 'office':
        return office_datasets_read(domain, dataset, batch_size, index_range, device)  # Remove device from here if GPU has not enough memory to store the whole dataset
