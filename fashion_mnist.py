import functools
import os
import operator
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import (
        datasets,
        transforms)

HIDDEN_UNITS = 10

class FashionMNISTClassifier(nn.Module):
    def __init__(self, input_dims, num_classes):
        super().__init__()
        n = functools.reduce(operator.mul, input_dims)
        self.mlp = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_features=n, out_features=HIDDEN_UNITS),
                nn.ReLU(),
                nn.Linear(in_features=HIDDEN_UNITS, out_features=HIDDEN_UNITS),
                nn.ReLU(),
                nn.Linear(in_features=HIDDEN_UNITS, out_features=num_classes),
                )

    def forward(self, x):
        return self.mlp(x)

def count_true_positives(y_true, y_pred):
    return torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal

def evaluate_model(train_dataloader, model, loss_function, device):
    model.eval()

    train_loss = 0
    true_positives = 0.
    with torch.inference_mode():
        for (x, y) in train_dataloader:
            x.to(device)
            y.to(device)
            y_pred = model(x).squeeze()
            y_softmax = torch.softmax(y_pred, dim=1)
            loss = loss_function(y_softmax, y)
            y_pred_labels = y_softmax.argmax(dim=1).squeeze()

            train_loss += loss.detach().numpy()
            true_positives = count_true_positives(y, y_pred_labels)

    accuracy = 100. * true_positives / len(train_dataloader)
    train_loss /= len(train_dataloader)

    return train_loss, accuracy

def main():
    train_data = datasets.FashionMNIST(
            root=os.path.expanduser('~/datasets/fashion_mnist'),
            train=True,
            download=True,
            transform=transforms.ToTensor(),
            target_transform=None)

    print(train_data)
    print(len(train_data))
    image, _ = train_data[0]
    image_shape, num_classes = image.shape, len(train_data.classes)

    batch_size = 32
    train_dataloader = DataLoader(train_data, batch_size, shuffle=True)
    print('Data loader', train_dataloader)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device = ', device)
    model = FashionMNISTClassifier(image_shape, num_classes).to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    epochs = 10

    for epoch in range(epochs):
        model.train()
        for batch_id, (x, y) in enumerate(train_dataloader):
            x.to(device)
            y.to(device)

            y_pred = model(x).squeeze()
            y_softmax = torch.softmax(y_pred, dim=1)
            loss = loss_function(y_softmax, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss, accuracy = evaluate_model(train_dataloader, model, loss_function, device)
        print(f'Epoch = {epoch} | Loss {loss} | Accuracy {accuracy}')


if __name__ == '__main__':
    main()
