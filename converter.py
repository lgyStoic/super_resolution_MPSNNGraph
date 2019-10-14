import torch
import numpy
import os

dest_path = './espcnn/parameter'

model = torch.load('model_epoch_493.pth')

for param_tensor in model.state_dict():
    t = model.state_dict()[param_tensor].numpy()
    t.tofile(os.path.join(dest_path, param_tensor + '.bin'))
