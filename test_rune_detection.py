#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试改进后的模型在视频验证中的表现，特别是rune_center的检测稳定性
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time
import os

def calculate_size_score(box_size, optimal_range):
    """
    计算边界框大小得分
    
    Args:
        box_size: 边界框大小
        optimal_range: 最佳大小范围 (min, max)
        
    Returns:
        大小得分 (0-1)
    """
    min_size, max_size = optimal_range
    
    if min_size <= box_size <= max_size:
        return 1.0
    elif box_size < min_size:
        return max(0.1, box_size / min_size)
    else:
        return max(0.1, max_size / box_size)

def calculate_keypoint_score(keypoints):
    """
    计算关键点质量得分
    
    Args:
        keypoints: 关键点列表
        
    Returns:
        关键点得分 (0-1)
    """
    if not keypoints:
        return 0.0
    
    valid_count = sum(1 for k in keypoints if k[0] != 0 and k[1] != 0)
    total_count = len(keypoints)
    
    return valid_count / total_count if total_count > 0 else 0.0

def calculate_scene_complexity(img):
    """
    计算场景复杂度
    
    Args:
        img: 输入图像
        
    Returns:
        场景复杂度得分 (0-1)，值越高表示场景越复杂
    """
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 计算边缘密度
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])
    
    # 计算纹理复杂度（使用方差）
    texture_complexity = np.var(gray) / 255.0
    
    # 计算亮度变化
    brightness_std = np.std(gray) / 255.0
    
    # 综合得分
    complexity_score = (edge_density + texture_complexity + brightness_std) / 3.0
    
    return min(1.0, complexity_score)

def calculate_adaptive_threshold(base_threshold, size_score, keypoint_score, scene_score, 
                               size_weight, keypoint_weight, scene_weight, 
                               min_threshold, max_threshold):
    """
    计算自适应阈值
    
    Args:
        base_threshold: 基础阈值
        size_score: 大小得分
        keypoint_score: 关键点得分
        scene_score: 场景复杂度得分
        size_weight: 大小权重
        keypoint_weight: 关键点权重
        scene_weight: 场景权重
        min_threshold: 最小阈值
        max_threshold: 最大阈值
        
    Returns:
        自适应阈值
    """
    # 计算综合得分
    total_score = (size_score * size_weight + 
                  keypoint_score * keypoint_weight + 
                  (1 - scene_score) * scene_weight)  # 场景越简单，得分越高
    
    # 根据综合得分调整阈值
    # 得分越高，阈值越低（更容易检测）
    adaptive_threshold = base_threshold * (1 - total_score * 0.5)
    
    # 确保阈值在合理范围内
    adaptive_threshold = max(min_threshold, min(max_threshold, adaptive_threshold))
    
    return adaptive_threshold

def test_rune_detection():
    """测试rune_center检测稳定性"""
    # 加载最新的模型
    model_path = "runs/pose/rune_pose_model_stage2/weights/best.pt"
    if not os.path.exists(model_path):
        model_path = "runs/pose/rune_pose_model_stage2/weights/last.pt"
    
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return
    
    print(f"✅ 加载模型: {model_path}")
    model = YOLO(model_path)
    
    # 设置测试参数 - 短期改进：使用动态阈值策略
    base_conf_threshold = 0.5  # 基础置信度阈值
    iou_threshold = 0.4   # 降低IOU阈值以减少重叠检测的过滤
    min_rune_center_conf = 0.15  # 最小rune_center检测阈值
    max_rune_center_conf = 0.3  # 最大rune_center检测阈值
    armor_module_conf = 0.5  # 保持armor_module的高检测阈值
    
    # 边界框特征参数
    optimal_rune_size_range = (40, 250)  # 调整最佳rune_center大小范围，适应更多场景
    size_weight = 0.25  # 调整大小特征权重
    keypoint_weight = 0.5  # 增加关键点质量权重，因为关键点对rune_center识别更重要
    scene_weight = 0.25  # 调整场景复杂度权重
    
    # 准备测试图像
    test_images = []
    # 遍历验证集图像
    val_images_dir = "images/val"
    if os.path.exists(val_images_dir):
        test_images = [os.path.join(val_images_dir, f) for f in os.listdir(val_images_dir) 
                      if f.endswith('.jpg') or f.endswith('.png')]
    
    if not test_images:
        print(f"❌ 没有找到测试图像: {val_images_dir}")
        return
    
    print(f"✅ 找到测试图像数量: {len(test_images)}")
    
    # 测试结果统计
    rune_center_detections = 0
    armor_module_detections = 0
    unstable_rune_detections = 0
    
    print(f"\n=== 开始测试rune_center检测稳定性 ===")
    
    for i, img_path in enumerate(test_images[:20]):  # 测试前20张图像
        print(f"\n测试图像 {i+1}/{20}: {img_path}")
        
        try:
            # 读取图像
            img = cv2.imread(img_path)
            if img is None:
                print(f"  ❌ 无法读取图像: {img_path}")
                continue
            
            # 模型推理 - 使用更低的基础置信度，后续根据类别过滤
            results = model(img, conf=0.1, iou=iou_threshold, device='cuda' if model.device.type == 'cuda' else 'cpu')
            
            # 解析结果
            for result in results:
                boxes = result.boxes
                keypoints = result.keypoints
                
                for j, box in enumerate(boxes):
                    conf = box.conf[0].item()
                    cls = int(box.cls[0].item())
                    cls_name = model.names[cls]
                    
                    # 根据类别应用不同的检测阈值 - 动态阈值策略
                    if cls_name == "rune_center":
                        # 计算边界框大小
                        box_size = (box[2] - box[0]) * (box[3] - box[1])
                        
                        # 检查关键点质量
                        kpt = keypoints[j].xy[0].tolist() if keypoints is not None else []
                        
                        # 计算场景复杂度
                        scene_complexity = calculate_scene_complexity(img)
                        
                        # 计算各项得分
                        size_score = calculate_size_score(box_size, optimal_rune_size_range)
                        keypoint_score = calculate_keypoint_score(kpt)
                        scene_score = scene_complexity
                        
                        # 计算自适应阈值
                        dynamic_threshold = calculate_adaptive_threshold(
                            base_conf_threshold,
                            size_score,
                            keypoint_score,
                            scene_score,
                            size_weight,
                            keypoint_weight,
                            scene_weight,
                            min_rune_center_conf,
                            max_rune_center_conf
                        )
                        
                        print(f"    📊 动态阈值计算: 大小得分={size_score:.2f}, 关键点得分={keypoint_score:.2f}, 场景复杂度={scene_score:.2f}, 阈值={dynamic_threshold:.3f}")
                        
                        if conf < dynamic_threshold:
                            continue  # 跳过置信度不足的rune_center
                    elif cls_name == "armor_module" and conf < armor_module_conf:
                        continue  # 跳过置信度不足的armor_module
                    
                    # 获取边界框
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    
                    # 计算边界框大小
                    box_size = (x2 - x1) * (y2 - y1)
                    
                    if cls_name == "rune_center":
                        rune_center_detections += 1
                        print(f"  ✅ 检测到 rune_center - 置信度: {conf:.3f}, 大小: {box_size}")
                        
                        # 检查置信度是否稳定
                        if conf < 0.7:
                            unstable_rune_detections += 1
                            print(f"    ⚠️  置信度较低: {conf:.3f}")
                        
                        # 检查关键点
                        kpt = keypoints[j].xy[0] if keypoints is not None else None
                        if kpt is not None:
                            # 计算有效关键点数量
                            valid_kpts = sum(1 for k in kpt if k[0] != 0 and k[1] != 0)
                            print(f"    ✅ 有效关键点数量: {valid_kpts}/9")
                    
                    elif cls_name == "armor_module":
                        armor_module_detections += 1
                        print(f"  ✅ 检测到 armor_module - 置信度: {conf:.3f}")
            
        except Exception as e:
            print(f"  ❌ 测试失败: {e}")
            continue
    
    # 输出测试结果
    print(f"\n=== 测试结果总结 ===")
    print(f"总测试图像数: {len(test_images[:20])}")
    print(f"检测到 rune_center: {rune_center_detections}")
    print(f"检测到 armor_module: {armor_module_detections}")
    print(f"不稳定的 rune_center 检测: {unstable_rune_detections}")
    
    # 计算稳定性指标
    if rune_center_detections > 0:
        stability_rate = (rune_center_detections - unstable_rune_detections) / rune_center_detections * 100
        print(f"rune_center 检测稳定率: {stability_rate:.1f}%")
    
    # 性能建议
    print(f"\n=== 性能改进建议 ===")
    if unstable_rune_detections > rune_center_detections * 0.5:
        print("⚠️  rune_center 检测稳定性较差，建议:")
        print("   1. 增加 rune_center 的训练样本数量")
        print("   2. 为 rune_center 添加真实的关键点标注")
        print("   3. 继续优化数据增强策略，增加旋转和颜色变化")
        print("   4. 考虑调整模型架构或增加训练轮数")
    else:
        print("✅ rune_center 检测稳定性良好")
    
    print(f"\n=== 测试完成 ===")

if __name__ == "__main__":
    test_rune_detection()
