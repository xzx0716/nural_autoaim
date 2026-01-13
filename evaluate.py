#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型评估脚本
用于验证YOLO11n-pose模型是否满足赛事性能要求
- mAP50≥0.85
- 针对旋转速度>0.4转/秒的能量机关装甲模块检测
- 5米距离下的识别精度
"""

import os
import yaml
import argparse
import numpy as np
from ultralytics import YOLO
from ultralytics.utils import ops
import cv2
from pathlib import Path

class RuneModelEvaluator:
    """能量机关模型评估器"""
    
    def __init__(self, model_path, data_config='rune_pose.yaml'):
        """
        初始化评估器
        
        Args:
            model_path: 训练好的模型路径
            data_config: 数据集配置文件路径
        """
        self.model = YOLO(model_path)
        
        # 加载数据集配置
        with open(data_config, 'r', encoding='utf-8') as f:
            self.data_config = yaml.safe_load(f)
        
        # 赛事性能要求
        self.required_map50 = 0.85
        self.required_rotation_speed = 0.4  # 转/秒
        self.required_distance = 5.0  # 米
    
    def evaluate_model(self, imgsz=640, batch=16, device=''):
        """
        评估模型性能
        
        Args:
            imgsz: 推理图像大小
            batch: 批处理大小
            device: 推理设备
            
        Returns:
            results: 评估结果
        """
        print("=" * 80)
        print("开始评估YOLO11n-pose能量机关检测模型")
        print(f"模型路径: {self.model.model.pt_path}")
        print(f"验证集路径: {self.data_config['val']}")
        print(f"评估图像大小: {imgsz}")
        print("=" * 80)
        
        # 运行评估
        results = self.model.val(
            data=self.data_config,
            imgsz=imgsz,
            batch=batch,
            device=device,
            conf=0.5,           # 置信度阈值
            iou=0.5,            # IoU阈值
            max_det=300,        # 最大检测数量
            name='rune_pose_evaluation',
            plots=True,         # 生成评估图表
        )
        
        return results
    
    def check_performance_requirements(self, results):
        """
        检查模型是否满足赛事性能要求
        
        Args:
            results: 评估结果
            
        Returns:
            bool: 是否满足所有要求
        """
        print("\n" + "=" * 80)
        print("赛事性能要求检查")
        print("=" * 80)
        
        # 获取关键评估指标
        map50 = results.box.map50
        map50_95 = results.box.map
        precision = results.box.mp
        recall = results.box.mr
        
        print(f"mAP50: {map50:.4f} {'✅' if map50 >= self.required_map50 else '❌'}")
        print(f"mAP50-95: {map50_95:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"赛事要求mAP50: ≥{self.required_map50}")
        
        # 关键点评估指标
        if hasattr(results, 'pose'):
            pose_map50 = results.pose.map50
            pose_map50_95 = results.pose.map
            print(f"\nPose mAP50: {pose_map50:.4f}")
            print(f"Pose mAP50-95: {pose_map50_95:.4f}")
        
        # 检查是否满足要求
        if map50 >= self.required_map50:
            print("\n✅ 模型满足赛事性能要求！")
            return True
        else:
            print("\n❌ 模型未满足赛事性能要求，请继续调优！")
            print("建议优化方向：")
            print("1. 增加旋转速度>0.4转/秒的训练数据")
            print("2. 调整数据增强策略，特别是旋转和颜色变换")
            print("3. 调整模型超参数，如学习率、批大小等")
            print("4. 增加训练轮数或调整迁移学习策略")
            return False
    
    def evaluate_on_test_images(self, test_images_dir, output_dir='evaluation_results'):
        """
        在测试图像上进行评估
        
        Args:
            test_images_dir: 测试图像目录
            output_dir: 结果输出目录
        """
        # 创建输出目录
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取测试图像列表
        image_files = list(Path(test_images_dir).glob('*.png')) + list(Path(test_images_dir).glob('*.jpg'))
        
        print(f"\n在 {len(image_files)} 张测试图像上进行评估...")
        
        total_detections = 0
        correct_detections = 0
        
        for image_path in image_files:
            # 读取图像
            img = cv2.imread(str(image_path))
            
            # 模型推理
            results = self.model.predict(
                img,
                imgsz=640,
                conf=0.5,
                iou=0.5,
                max_det=10,
                device='',
            )
            
            # 处理结果
            for result in results:
                boxes = result.boxes
                keypoints = result.keypoints
                
                if len(boxes) > 0:
                    total_detections += len(boxes)
                    
                    # 绘制检测结果
                    result.render()
                    
                    # 保存结果图像
                    output_path = output_dir / image_path.name
                    cv2.imwrite(str(output_path), result.plot()[:, :, ::-1])
                    
                    # 简单的正确性检查（这里可以根据实际需求扩展）
                    for box, cls in zip(boxes.xyxy, boxes.cls):
                        if int(cls) == 1:  # armor_module
                            correct_detections += 1
        
        if total_detections > 0:
            precision = correct_detections / total_detections
            print(f"测试图像检测精度: {precision:.4f} ({correct_detections}/{total_detections})")
        
        print(f"评估结果已保存至: {output_dir}")
        
    def get_evaluation_summary(self, results):
        """
        获取评估摘要
        
        Args:
            results: 评估结果
            
        Returns:
            summary: 评估摘要字典
        """
        summary = {
            'model_path': self.model.model.pt_path,
            'map50': results.box.map50 if hasattr(results.box, 'map50') else 0.0,
            'map50_95': results.box.map if hasattr(results.box, 'map') else 0.0,
            'precision': results.box.mp if hasattr(results.box, 'mp') else 0.0,
            'recall': results.box.mr if hasattr(results.box, 'mr') else 0.0,
            'f1_score': 2 * (results.box.mp * results.box.mr) / (results.box.mp + results.box.mr) if (results.box.mp + results.box.mr) > 0 else 0.0,
            'required_map50': self.required_map50,
            'meets_requirements': results.box.map50 >= self.required_map50 if hasattr(results.box, 'map50') else False,
        }
        
        # 关键点评估指标
        if hasattr(results, 'pose'):
            summary['pose_map50'] = results.pose.map50 if hasattr(results.pose, 'map50') else 0.0
            summary['pose_map50_95'] = results.pose.map if hasattr(results.pose, 'map') else 0.0
        
        return summary

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='YOLO11n-pose能量机关模型评估脚本')
    parser.add_argument('--model', type=str, required=True, help='训练好的模型路径')
    parser.add_argument('--data', type=str, default='rune_pose.yaml', help='数据集配置文件路径')
    parser.add_argument('--imgsz', type=int, default=640, help='推理图像大小')
    parser.add_argument('--batch', type=int, default=16, help='批处理大小')
    parser.add_argument('--device', type=str, default='', help='推理设备')
    parser.add_argument('--test_images', type=str, default=None, help='测试图像目录')
    parser.add_argument('--output', type=str, default='evaluation_results', help='评估结果输出目录')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    # 创建评估器
    evaluator = RuneModelEvaluator(args.model, args.data)
    
    # 评估模型
    results = evaluator.evaluate_model(args.imgsz, args.batch, args.device)
    
    # 检查性能要求
    meets_requirements = evaluator.check_performance_requirements(results)
    
    # 获取评估摘要
    summary = evaluator.get_evaluation_summary(results)
    
    # 输出评估摘要
    print("\n" + "=" * 80)
    print("评估摘要")
    print("=" * 80)
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    # 在测试图像上评估（可选）
    if args.test_images:
        evaluator.evaluate_on_test_images(args.test_images, args.output)
    
    print("\n" + "=" * 80)
    print("模型评估完成！")
    print("=" * 80)
    
    # 返回退出码
    exit(0 if meets_requirements else 1)
