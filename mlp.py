import torch
from torch import nn
import numpy as np
from sklearn import datasets
from matplotlib import pyplot as plt

class CircleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features=2, out_features=10),
            nn.ReLU(),
            nn.Linear(in_features=10, out_features=10),
            nn.ReLU(),
            nn.Linear(in_features=10, out_features=10),
            nn.ReLU(),
            nn.Linear(in_features=10, out_features=1),
            nn.ReLU()
        )
    def forward(self, x) -> torch.Tensor:
        return self.mlp(x)

def setup_training_data():
    nsamples = 1000
    x_np, y_np = datasets.make_circles(nsamples, noise=0.03, random_state=100) 
    x = torch.from_numpy(x_np).type(torch.float)
    y = torch.from_numpy(y_np).type(torch.float)
    return x, y

def estimate_accuracy(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100 
    return acc

def main():
    torch.manual_seed(100)
    x, y = setup_training_data()

    model = CircleModel()
    epochs = 5000
    check_point = epochs / 10

    loss_function = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1)

    for epoch in range(epochs):
        # Put model in training mode
        model.train()

        # Compute loss
        y_pred = model(x)
        loss = loss_function(y_pred.squeeze(), y)

        y_prob = torch.sigmoid(y_pred)
        y_pred_labels = torch.round(y_pred).squeeze()

        accuracy = estimate_accuracy(y, y_pred_labels)

        # Invoke the optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Put the model in evaluation mode
        model.eval()
        with torch.inference_mode():
            y_pred = model(x)
            training_loss = loss_function(y_pred.squeeze(), y)

            y_prob = torch.sigmoid(y_pred)
            y_pred_labels = torch.round(y_prob).squeeze()        
            accuracy = estimate_accuracy(y, y_pred_labels)

            if epoch % check_point == 0:
                training_loss_value = training_loss.detach().numpy()
                print(f"Epoch: {epoch} | Accuracy: {accuracy} | Train Loss: {training_loss_value}")

if __name__ == '__main__':
    main()
