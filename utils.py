import torch
import functools

def estimate_number_of_parameters_in_a_model(model):
  # model is the loaded torchvision model
  # e.g. model = torchvision.models.resnet18()
  
  params_layer_wise = [tensor_info.shape for _, tensor_info in model.state_dict().items() if tensor_info.shape]
  return sum([functools.reduce(lambda x, y : x*y, shape) for shape in params_layer_wise])
