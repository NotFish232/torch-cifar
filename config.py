import torch

device_name: str = "cuda" if torch.cuda.is_available() else "cpu"
device: torch.device = torch.device(device_name)


conv_kernel_size: int = 3
pool_kernel_size: int = 2
stride_size: int = 2
dropout_prob: float = 0.0025

batch_size: int = 8
num_epochs: int = 10