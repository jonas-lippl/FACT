import numpy as np


def dense_to_one_hot(labels_dense):
    """Convert class labels from scalars to one-hot vectors."""
    labels_one_hot = np.zeros((len(labels_dense),))
    labels_dense = list(labels_dense)
    for i, t in enumerate(labels_dense):
        if t == 10:
            t = 0
            labels_one_hot[i] = t
        else:
            labels_one_hot[i] = t
    return labels_one_hot


def get_source_domains_for_dataset(dataset: str, target_domains=None, exclude_domains=None):
    domains = []
    if dataset == 'digit-five':
        domains = ["mnistm", "mnist", "svhn", "syn", "usps"]
    if dataset == 'office_caltech_10':
        domains = ["Caltech", "amazon", "dslr", "webcam"]
    if dataset == 'office':
        domains = ['amazon', 'dslr', 'webcam']

    if target_domains:
        for target_domain in target_domains:
            domains.remove(target_domain)
    if exclude_domains:
        for domain in exclude_domains:
            domains.remove(domain)
    return domains
