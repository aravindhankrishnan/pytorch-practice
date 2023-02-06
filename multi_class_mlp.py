import torch
from torch import nn
import numpy as np
from sklearn import datasets
from matplotlib import pyplot as plt

RANDOM_SEED = 42

class BlobModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
                nn.Linear(in_features=2, out_features=8),
                nn.ReLU(),
                nn.Linear(in_features=8, out_features=8),
                nn.ReLU(),
                nn.Linear(in_features=8, out_features=4),
                #nn.ReLU()
                )

    def forward(self, x):
        return self.mlp(x)

def setup_training_data():
    nsamples = 1000
    x_np, y_np = datasets.make_blobs(n_samples=1000,
        n_features=2,
        centers=4,
        cluster_std=1.5,
        random_state=RANDOM_SEED
    )

    x = torch.from_numpy(x_np).type(torch.float)
    y = torch.from_numpy(y_np).type(torch.LongTensor)
    return x, y

def estimate_accuracy(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    accuracy = (correct / len(y_pred)) * 100 
    return accuracy

def main():
    x, y = setup_training_data()

    model = BlobModel()
    epochs = 100000
    check_point = 1000

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1)

    torch.manual_seed(RANDOM_SEED)

    for epoch in range(epochs):
        # Put model in training mode
        model.train()
    
        # Compute loss
        y_pred = model(x).squeeze()
        y_softmax = torch.softmax(y_pred, dim=1)
        loss = loss_function(y_softmax, y)
        
        y_pred_labels = y_softmax.argmax(dim=1).squeeze()
    
        # Invoke the optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        # Put the model in evaluation mode
        if epoch % check_point:
            continue

        model.eval()
        with torch.inference_mode():
            # Compute loss
            y_pred = model(x).squeeze()
            y_softmax = torch.softmax(y_pred, dim=1)
            loss = loss_function(y_softmax, y)
            
            y_pred_labels = y_softmax.argmax(dim=1).squeeze()

            accuracy = estimate_accuracy(y, y_pred_labels)
            loss = loss.detach().numpy()
            print(f"Epoch: {epoch} | Accuracy: {accuracy} | Train Loss: {loss}")

            if accuracy > 99.0:
                break


if __name__ == '__main__':
    main()
