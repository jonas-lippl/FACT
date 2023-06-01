from torch.optim.lr_scheduler import LambdaLR


class CustomLRScheduler(LambdaLR):
    """
    Custom learning rate scheduler for the one shot FACT scenario. Learning rate is decayed according
    to the following formula: lr = initial_lr * (1 + 10 * epoch / total_epochs) ** (-decay_rate)
    """
    def __init__(self, optimizer, total_epochs, decay_rate, last_epoch=-1):
        self.total_epochs = total_epochs
        self.decay_rate = decay_rate
        super().__init__(optimizer, lr_lambda=self.custom_lr_lambda, last_epoch=last_epoch)

    def custom_lr_lambda(self, epoch):
        return (1 + 10 * epoch / self.total_epochs) ** (-self.decay_rate)
