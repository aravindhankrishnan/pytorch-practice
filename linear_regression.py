import torch
from torch import nn
import numpy as np

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1, dtype=torch.float), requires_grad=True)
        self.bias = nn.Parameter(torch.randn(1, dtype=torch.float), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias

def setup_training_data():
    x = torch.arange(0, 1.0, 0.01)
    w = 2.5
    b = 0.1
    y = w * x + b
    return x, y

def main():
    x, y = setup_training_data()

    ## Setting up the model definition
    torch.manual_seed(42)
    linear_regression_model = LinearRegressionModel()
    loss_fn = nn.L1Loss()
    optimizer = torch.optim.SGD(params=linear_regression_model.parameters(), lr=0.01)

    print('Model parameters', list(linear_regression_model.parameters())) # Also informs about required_grad parameter
    print('Model state dict', linear_regression_model.state_dict())

    torch.manual_seed(42)
    epochs = 1000
    
    for epoch in range(epochs):
        # Put model in training mode
        linear_regression_model.train()
    
        # Compute loss
        y_pred = linear_regression_model(x)
        loss = loss_fn(y_pred, y)
    
        # Invoke the optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        # Put the model in evaluation mode
        linear_regression_model.eval()
        with torch.inference_mode():
          y_pred_new = linear_regression_model(x)
          train_loss = loss_fn(y_pred_new, y.type(torch.float)) # predictions come in torch.float datatype, so comparisons need to be done with tensors of the same type
    
          if epoch % 10 == 0:
                loss = train_loss.detach().numpy()
                print(f"Epoch: {epoch} | Train Loss: {loss}")

    print('w = ', linear_regression_model.weights.detach().numpy())
    print('b = ', linear_regression_model.bias.detach().numpy())

if __name__ == '__main__':
    main()
