#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO11n-pose Training Script for Rune Detection (CPU版本)
赛事专属配置：
- 适配灯环一体化PCB设计的WS2812灯珠检测
- 旋转速度>0.4转/秒的能量机关识别
- 5米距离命中率考核
- 嵌入式部署优化
"""

import os
import yaml
from ultralytics import YOLO
from ultralytics.utils import callbacks
import torch
import datetime
import subprocess
import time
import webbrowser

# 加载数据集配置
with open('rune_pose.yaml', 'r', encoding='utf-8') as f:
    data_config = yaml.safe_load(f)

# 训练参数配置
train_params = {
    'data': 'rune_pose.yaml',
    'imgsz': 640,                # 图像大小
    'batch': 16,                 # 批处理大小
    'lr0': 0.01,                 # 初始学习率
    'lrf': 0.01,                 # 最终学习率
    'momentum': 0.937,           # 动量
    'weight_decay': 0.0005,      # 权重衰减
    'warmup_epochs': 3,          # 预热轮数
    'warmup_momentum': 0.8,      # 预热动量
    'warmup_bias_lr': 0.1,       # 预热偏置学习率
    'box': 7.5,                  # 边界框损失权重
    'cls': 0.5,                  # 类别损失权重
    'dfl': 1.5,                  # 分布焦点损失权重
    'pose': 12.0,                # 姿态关键点损失权重
    'kobj': 1.0,                 # 关键点可见性损失权重

    'nbs': 64,                   # 标称批大小
    'overlap_mask': True,        # 重叠掩码
    'mask_ratio': 4,             # 掩码比例
    'dropout': 0.0,              # Dropout概率
    'val': True,                 # 训练期间验证
    'save_period': 10,           # 每10轮保存一次检查点
    'exist_ok': True,            # 覆盖现有目录
    'pretrained': True,          # 使用预训练权重
    'optimizer': 'SGD',          # 优化器
    'verbose': True,             # 详细输出
    'deterministic': True,       # 确定性训练
    'seed': 42,                  # 随机种子
    'workers': 4,                # 数据加载线程数
    'amp': True,                 # 混合精度训练
    'fraction': 1.0,             # 训练集使用比例
    'profile': False,            # 性能分析
    'cos_lr': False,             # 余弦学习率调度
    'close_mosaic': 10,          # 最后10轮关闭马赛克增强
    'resume': False,             # 从断点恢复训练
    'single_cls': False,         # 单类别训练
    'device': 'cpu',             # 强制使用CPU训练
    'visualize': False,          # 可视化训练
    'plots': True,               # 生成训练图表
}

# 导入自定义数据增强配置
from data_augmentation import high_level_augmentation

# 数据增强配置
data_augmentation = {
    'augment': True,
    'hsv_h': 0.2,                # 色相调整 - 适应灯珠颜色变化
    'hsv_s': 0.5,                # 饱和度调整 - 增强灯珠发光效果
    'hsv_v': 0.5,                # 亮度调整 - 模拟不同光照条件
    'degrees': 15.0,             # 旋转角度 - 适应旋转速度>0.4转/秒的能量机关
    'translate': 0.15,           # 平移比例 - 适应5米距离视角变化
    'scale': 0.7,                # 缩放比例 - 扩大尺度变化范围
    'shear': 0.0,                # 剪切角度
    'perspective': 0.0,          # 透视变换
    'flipud': 0.0,               # 上下翻转概率（保持物理合理性）
    'fliplr':