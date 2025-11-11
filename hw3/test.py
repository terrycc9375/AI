import torch

print("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.get_device_name(0))
