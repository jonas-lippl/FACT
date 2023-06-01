import torch
from torch.autograd import Variable


def test(dataloader, model, loss_fn):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    size = 0
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for data in dataloader:
            if type(data) == dict:
                X = data['img']
                y = data['label']
            else:
                X, y = data
            X = Variable(X.to(device))
            y = Variable(y.long().to(device))
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            size += y.data.size()[0]
    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f}\n"
    )
    return correct, test_loss
