import functools
import os
import operator
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import (
        datasets,
        transforms)

class FashionMNISTClassifier(nn.Module):
    def __init__(self, input_dims, num_classes):
        super().__init__()
        n = functools.reduce(operator.mul, input_dims)
        self.mlp = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_features=n, out_features=1024),
                nn.ReLU(),
                nn.Linear(in_features=1024, out_features=1024),
                nn.ReLU(),
                nn.Linear(in_features=1024, out_features=num_classes)
                )

    def forward(self, x):
        return self.mlp(x)

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

    loss_function = nn.CrossEntropy()
    optimizer = nn.optim.SGD(model.parameters(), lr=0.1)

    for batch, (X, y) in enumerate(train_dataloader):
        print('Batch', batch)

if __name__ == '__main__':
    main()
