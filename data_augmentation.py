#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据增强配置文件
针对能量机关硬件特性优化：
- WS2812灯珠动态颜色切换
- 旋转速度>0.4转/秒的能量机关
- 5米距离视角变化
- 环境光照差异
"""

from ultralytics.data.augment import Compose, LetterBox, RandomFlip, RandomPerspective
from ultralytics.data.augment import RandomHSV
from ultralytics.data.augment import CopyPaste, Mosaic, MixUp, Albumentations

class CustomRuneAugmentation:
    """自定义能量机关数据增强策略"""
    
    @staticmethod
    def get_train_augmentations():
        """获取训练数据增强配置"""
        return Compose([
            # 基础预处理
            LetterBox(new_shape=640, auto=True, scale_fill=False, scaleup=True),
            
            # 几何变换 - 适应旋转速度>0.4转/秒的能量机关
            RandomPerspective(
                degrees=15.0,               # 旋转角度扩大到15度，适应快速旋转
                translate=0.15,              # 平移比例
                scale=0.7,                   # 缩放比例扩大
                shear=0.0,                   # 剪切角度
                perspective=0.0,             # 透视变换
                border=(0, 0),
            ),
            
            # 翻转增强
            RandomFlip(p=0.5, direction='horizontal'),  # 左右翻转
            
            # 颜色变换 - 针对WS2812灯珠动态颜色切换
            RandomHSV(
                hgain=0.2,                   # 色相调整范围扩大
                sgain=0.5,                   # 饱和度调整范围扩大
                vgain=0.5,                   # 亮度调整范围扩大
            ),
            
            # 马赛克增强 - 提高模型对复杂场景的适应性
            Mosaic(p=1.0, n=4),
            
            # 混合增强 - 提高模型泛化能力
            MixUp(p=0.2, beta=1.0),
        ])
    
    @staticmethod
    def get_val_augmentations():
        """获取验证数据增强配置"""
        return Compose([
            LetterBox(new_shape=640, auto=True, scale_fill=False, scaleup=False),
        ])

# 针对灯珠亮度和光照变化的高级增强配置
high_level_augmentation = {
    # 颜色增强参数
    'hsv_h': 0.2,      # 色相调整范围，适应灯珠颜色变化
    'hsv_s': 0.5,      # 饱和度调整范围，增强灯珠发光效果
    'hsv_v': 0.5,      # 亮度调整范围，模拟不同光照条件
    
    # 旋转增强参数 - 适应快速旋转的能量机关
    'degrees': 15.0,   # 旋转角度扩大，覆盖0.4-1转/秒的旋转速度
    
    # 尺度和视角增强 - 适应5米距离的视角变化
    'translate': 0.15, # 平移比例
    'scale': 0.7,      # 缩放比例
    'shear': 0.0,      # 剪切角度
    'perspective': 0.0,# 透视变换
    
    # 翻转增强
    'flipud': 0.0,     # 上下翻转概率（不使用，保持物理合理性）
    'fliplr': 0.5,     # 左右翻转概率
    
    # 马赛克和混合增强
    'mosaic': 1.0,     # 马赛克增强概率
    'mixup': 0.2,      # 混合增强概率
    'copy_paste': 0.0, # 复制粘贴增强概率
    
    # 自动增强
    'auto_augment': 'randaugment',  # 自动增强策略
    'erasing': 0.2,    # 随机擦除概率
}

# 针对旋转速度优化的时间序列增强
temporal_augmentation = {
    'motion_blur': 0.3,  # 运动模糊概率，模拟快速旋转
    'blur_kernel_size': (3, 7),  # 模糊核大小范围
    'rotation_velocity': (0.4, 1.0),  # 旋转速度范围（转/秒）
    'frame_interval': 1,  # 帧间隔
}

# 针对5米距离视角变化的增强
view_augmentation = {
    'perspective_transform': 0.1,  # 透视变换概率
    'view_angle_variation': (-10, 10),  # 视角变化范围（度）
    'distance_variation': (4.5, 5.5),  # 距离变化范围（米）
}

# 针对WS2812灯珠的颜色增强
led_color_augmentation = {
    'color_jitter_brightness': 0.3,  # 亮度抖动
    'color_jitter_contrast': 0.3,    # 对比度抖动
    'color_jitter_saturation': 0.5,  # 饱和度抖动
    'color_jitter_hue': 0.2,         # 色相抖动
    'led_intensity_range': (0.3, 1.2),  # LED亮度范围
    'led_color_temperature': (3000, 6500),  # 色温范围
}

def apply_custom_augmentations(dataset):
    """为数据集应用自定义增强策略"""
    dataset.transforms = CustomRuneAugmentation.get_train_augmentations()
    dataset.val_transforms = CustomRuneAugmentation.get_val_augmentations()
    return dataset

# if __name__ == '__main__':
#     # 测试增强配置
#     print("数据增强配置信息：")
#     print("=" * 60)
#     print("训练增强策略：")
#     for t in CustomRuneAugmentation.get_train_augmentations().transforms:
#         print(f"  - {type(t).__name__}")
#     
#     print("\n验证增强策略：")
#     for t in CustomRuneAugmentation.get_val_augmentations().transforms:
#         print(f"  - {type(t).__name__}")
#     
#     print("\n高级增强参数：")
#     for k, v in high_level_augmentation.items():
#         print(f"  {k}: {v}")
