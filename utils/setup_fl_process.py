from datasets.utils import get_source_domains_for_dataset

dataset_domain_sample_count_mapping = {
    'digit-five': {
        'mnistm': 25000,
        'svhn': 25000,
        'usps': 7438,
        'syn': 25000,
        'mnist': 25000,
    },
    'office': {
        'amazon': 2817,
        'dslr': 498,
        'webcam': 795,
    },
    'office_caltech_10': {
        'amazon': 958,
        'Caltech': 1123,
        'dslr': 157,
        'webcam': 295,
    }
}


def arg_str_to_bool(arg_str) -> bool:
    return arg_str == 'yes'


def get_source_and_target_domains(args) -> (list[str], list[str]):
    target_domains = []
    if args.target != '':
        target_domains = [domain.strip() for domain in args.target.split(",")]
    exclude_domains = []
    if args.exclude_domains != '':
        exclude_domains = [domain.strip() for domain in args.exclude_domains.split(",")]
    source_domains = get_source_domains_for_dataset(args.dataset, target_domains, exclude_domains)
    print(f"source: {source_domains}, target: {target_domains}")
    return source_domains, target_domains
