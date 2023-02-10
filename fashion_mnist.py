import argparse
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

class FashionMNISTClassifierLinear(nn.Module):
    def __init__(self, input_dims, num_classes):
        super().__init__()
        n = functools.reduce(operator.mul, input_dims)
        self.mlp = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_features=n, out_features=HIDDEN_UNITS),
                nn.Linear(in_features=HIDDEN_UNITS, out_features=HIDDEN_UNITS),
                )

    def forward(self, x):
        return self.mlp(x)

class FashionMNISTClassifierReLU(nn.Module):
    def __init__(self, input_dims, num_classes):
        super().__init__()
        n = functools.reduce(operator.mul, input_dims)
        self.mlp = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_features=n, out_features=HIDDEN_UNITS),
                nn.ReLU(),
                nn.Linear(in_features=HIDDEN_UNITS, out_features=HIDDEN_UNITS),
                nn.ReLU(),
                )

    def forward(self, x):
        return self.mlp(x)

def count_true_positives(y_true, y_pred):
    return torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal

def evaluate_model(train_dataloader, model, loss_function, device):
    model.eval()

    train_loss = 0
    true_positives = 0.
    total = 0
    with torch.inference_mode():
        for (x, y) in train_dataloader:
            x.to(device)
            y.to(device)
            
            y_pred = model(x).squeeze()
            y_softmax = torch.softmax(y_pred, dim=1)

            loss = loss_function(y_softmax, y)
            train_loss += loss.detach().numpy()

            y_pred_labels = y_softmax.argmax(dim=1).squeeze()

            true_positives += count_true_positives(y, y_pred_labels)
            total += len(y_pred_labels)

    accuracy = 100. * true_positives / total
    train_loss /= len(train_dataloader)

    return train_loss, accuracy

def main():
    parser = argparse.ArgumentParser(description='Fashion MNIST model')                       
    parser.add_argument('--model', type=str, help='Model type', required=True, choices=['linear', 'relu'])    

    args = parser.parse_args()

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
    print('Num classes = ', num_classes)

    batch_size = 32
    train_dataloader = DataLoader(train_data, batch_size, shuffle=True)
    print('Data loader', train_dataloader)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device = ', device)
    
    if args.model == 'linear':
        model = FashionMNISTClassifierLinear(image_shape, num_classes).to(device)
    elif args.model == 'relu':
        model = FashionMNISTClassifierReLU(image_shape, num_classes).to(device)

    print(model)

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    torch.manual_seed(100)

    epochs = 100

    for epoch in range(epochs):
        model.train()
        train_loss = 0
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
