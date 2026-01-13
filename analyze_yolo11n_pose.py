#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO11n-pose模型结构分析脚本
"""

from ultralytics import YOLO
import torch

# 加载YOLO11n-pose模型
model = YOLO('yolo11n-pose.pt')

print("=" * 80)
print("YOLO11n-pose 模型结构分析")
print("=" * 80)
print(f"模型名称: {model.model.__class__.__name__}")

# 直接打印模型摘要信息
print("\n模型摘要:")
model.model.info()

print(f"\n模型设备: {model.device}")
print("=" * 80)

print("\n模型基本结构:")
print("=" * 80)
print(model.model)

print("\n" + "=" * 80)
print("模型层详细信息:")
print("=" * 80)

# 打印模型的主要组件
for i, module in enumerate(model.model.children()):
    print(f"\n层 {i}: {module.__class__.__name__}")
    print(f"  输入形状: {module.input_shape if hasattr(module, 'input_shape') else '未知'}")
    print(f"  输出形状: {module.output_shape if hasattr(module, 'output_shape') else '未知'}")
    print(f"  参数数量: {sum(p.numel() for p in module.parameters()) if hasattr(module, 'parameters') else 'N/A'}")

# 检查模型的检测头结构
print("\n" + "=" * 80)
print("检测头结构:")
print("=" * 80)

# YOLO11的检测头通常在模型的最后几个模块中
model_children = list(model.model.children())
detect_heads = model_children[-3:]
for i, head in enumerate(detect_heads):
    print(f"\n检测头 {i}: {head.__class__.__name__}")
    print(f"  类型: {type(head)}")
    print(f"  结构: {head}")

# 分析检测头的输出格式
print("\n" + "=" * 80)
print("检测头输出格式分析:")
print("=" * 80)

# 创建一个测试输入
test_input = torch.randn(1, 3, 640, 640)

# 前向传播，查看输出
with torch.no_grad():
    outputs = model.model(test_input)

print(f"\n模型输出数量: {len(outputs)}")
for i, output in enumerate(outputs):
    print(f"输出 {i} 类型: {type(output)}")
    if hasattr(output, 'shape'):
        print(f"输出 {i} 形状: {output.shape}")
    else:
        print(f"输出 {i} 结构: {output}")

# 特别关注姿态估计的关键点输出
print("\n" + "=" * 80)
print("关键点输出分析:")
print("=" * 80)

# 获取模型配置信息
model_info = model.model.yaml
print(f"模型配置: {model_info}")

# 检查是否包含姿态估计相关配置
if 'pose' in model_info:
    print(f"姿态估计配置: {model_info['pose']}")
    print(f"关键点数量: {model_info['pose'].get('kpt_shape', [0])[0]}")

print("\n" + "=" * 80)
print("YOLO11n-pose 核心组件总结")
print("=" * 80)
print("1. 骨干网络 (Backbone): 提取图像特征")
print("2. 颈部网络 (Neck): 特征融合，增强语义信息")
print("3. 检测头 (Head): 同时输出目标检测框和关键点")
print("   - 边界框输出: 中心坐标、宽度、高度")
print("   - 关键点输出: 多个关键点的坐标和可见性")
print("4. 损失函数: 同时优化检测框和关键点")
print("   - 边界框损失")
print("   - 类别损失")
print("   - 关键点坐标损失")
print("   - 关键点可见性损失")
