#!/usr/bin/env python3
import torch
import os

# 将CUDA检查结果写入文件
with open('cuda_check.txt', 'w') as f:
    f.write(f'PyTorch版本: {torch.__version__}\n')
    f.write(f'CUDA可用: {torch.cuda.is_available()}\n')
    if torch.cuda.is_available():
        f.write(f'GPU设备: {torch.cuda.get_device_name(0)}\n')
        f.write(f'CUDA版本: {torch.version.cuda}\n')
        f.write(f'设备数量: {torch.cuda.device_count()}\n')
        f.write(f'当前设备: {torch.cuda.current_device()}\n')
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        f.write(f'CUDA_VISIBLE_DEVICES设置为: {os.environ.get("CUDA_VISIBLE_DEVICES")}\n')
    else:
        f.write('CUDA不可用\n')

# 运行训练脚本
import subprocess
print('正在启动训练...')
process = subprocess.Popen(['python', 'train.py'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

# 实时输出训练日志
while True:
    line = process.stdout.readline()
    if not line and process.poll() is not None:
        break
    if line:
        print(line.strip())
        # 将训练日志写入文件
        with open('train_log.txt', 'a') as f:
            f.write(line)

# 获取进程退出码
retcode = process.poll()
print(f'训练结束，退出码: {retcode}')