from torchvision.models import resnet101, ResNet101_Weights

from models import digit_model, resnet_compatible_predictor


def get_dataset_specific_generator(dataset: str):
    if dataset == 'digit-five':
        return digit_model.Feature()
    if dataset in ['office_caltech_10', 'office']:
        return resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)


def get_dataset_specific_classifier(dataset: str):
    if dataset == 'digit-five':
        return digit_model.Predictor()
    if dataset == 'office_caltech_10':
        return resnet_compatible_predictor.Predictor(classes=10)
    if dataset == 'office':
        return resnet_compatible_predictor.Predictor(classes=31)
