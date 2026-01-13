#!/usr/bin/env python3
import os
import sys

# 检查Python和pip版本
print('Python版本:', sys.version)
print('pip版本:', os.popen('pip --version').read().strip())

# 检查PyTorch安装
print('\n检查PyTorch安装:')
os.system('pip list | grep torch')

# 检查CUDA可用性
try:
    import torch
    print('\nPyTorch版本:', torch.__version__)
    print('CUDA可用:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('GPU设备:', torch.cuda.get_device_name(0))
        print('CUDA版本:', torch.version.cuda)
        # 设置CUDA可见设备
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        print('CUDA_VISIBLE_DEVICES设置为:', os.environ.get('CUDA_VISIBLE_DEVICES'))
    else:
        print('CUDA不可用')
except Exception as e:
    print('导入PyTorch失败:', e)

# 尝试启动训练
print('\n正在启动训练...')
os.system('python train.py')