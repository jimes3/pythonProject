import torch
print(torch.__version__)           # PyTorch 版本
print(torch.version.cuda)          # CUDA 版本
print(torch.cuda.is_available())   # True 表示 GPU 可用
print(torch.cuda.get_device_name(0))  # GPU 名称
