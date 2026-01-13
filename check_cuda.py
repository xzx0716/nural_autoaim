#!/usr/bin/env python3
import torch
print('PyTorch版本:', torch.__version__)
print('CUDA可用:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU设备:', torch.cuda.get_device_name(0))
    print('CUDA版本:', torch.version.cuda)
    print('设备数量:', torch.cuda.device_count())
    print('当前设备:', torch.cuda.current_device())
else:
    print('CUDA不可用')

# 运行训练脚本
import subprocess
print('\n正在启动训练...')
subprocess.run(['python', 'train.py'])