import torch
from torch import nn
import numpy as np
from sklearn import datasets
from matplotlib import pyplot as plt

class CircleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features=2, out_features=5),
            nn.ReLU(),
            nn.Linear(in_features=5, out_features=5),
            nn.ReLU(),
            nn.Linear(in_features=5, out_features=5),
            nn.ReLU(),
            nn.Linear(in_features=5, out_features=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.mlp(x)

def setup_training_data():
    nsamples = 1000
    x_np, y_np = datasets.make_circles(nsamples, noise=0.03, random_state=100) 
    x = torch.from_numpy(x_np).type(torch.float)
    y = torch.from_numpy(y_np).type(torch.float)
    return x, y

def estimate_accuracy(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    accuracy = (correct / len(y_pred)) * 100 
    return accuracy

def main():
    x, y = setup_training_data()

    model = CircleModel()
    print(model)
    epochs = 100000
    check_point = 1000

    loss_function = nn.BCELoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1)
    print(optimizer)

    torch.manual_seed(100)

    for epoch in range(epochs):
        # Put model in training mode
        model.train()
    
        # Compute loss
        y_pred = model(x).squeeze()
        y_logits = torch.sigmoid(y_pred)
        loss = loss_function(y_logits, y)
        
        y_pred_labels = torch.round(y_logits).squeeze()
        accuracy = estimate_accuracy(y, y_pred_labels)
    
        # Invoke the optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        # Put the model in evaluation mode
        if epoch % check_point:
            continue

        model.eval()
        with torch.inference_mode():
            y_pred = model(x).squeeze()
            y_logits = torch.sigmoid(y_pred)
            loss = loss_function(y_logits, y)
            y_pred_labels = torch.round(y_logits).squeeze()        

            accuracy = estimate_accuracy(y, y_pred_labels)
            loss = loss.detach().numpy()
            print(f"Epoch: {epoch} | Accuracy: {accuracy} | Train Loss: {loss}")

            if accuracy > 99.0:
                break


if __name__ == '__main__':
    main()
