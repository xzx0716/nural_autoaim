#!/usr/bin/env python3
import subprocess
import sys

# 检查当前PyTorch版本和CUDA状态
def check_environment():
    try:
        import torch
        print(f"PyTorch版本: {torch.__version__}")
        print(f"CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA版本: {torch.version.cuda}")
            print(f"GPU设备: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"无法导入PyTorch: {e}")

# 安装GPU版本的PyTorch
def install_gpu_pytorch():
    print("正在安装GPU版本的PyTorch...")
    cmd = [
        sys.executable, 
        "-m", "pip", 
        "install", 
        "torch", "torchvision", "torchaudio",
        "--index-url", "https://download.pytorch.org/whl/cu121"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(f"安装输出: {result.stdout}")
    print(f"安装错误: {result.stderr}")
    print(f"返回码: {result.returncode}")

if __name__ == "__main__":
    print("=== 检查当前环境 ===")
    check_environment()
    
    print("\n=== 安装GPU版本的PyTorch ===")
    install_gpu_pytorch()
    
    print("\n=== 安装后检查环境 ===")
    check_environment()