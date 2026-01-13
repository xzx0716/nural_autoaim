#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型导出脚本
用于将训练好的YOLO11n-pose模型导出为ONNX、TensorRT等格式
适配NVIDIA Jetson Xavier系列嵌入式平台（32 TFLOPS GPU）
"""

import os
import argparse
import yaml
from ultralytics import YOLO
from ultralytics.utils import ops
import numpy as np
import torch
from pathlib import Path

class RuneModelExporter:
    """能量机关模型导出器"""
    
    def __init__(self, model_path, data_config='rune_pose.yaml'):
        """
        初始化导出器
        
        Args:
            model_path: 训练好的模型路径
            data_config: 数据集配置文件路径
        """
        self.model = YOLO(model_path)
        
        # 加载数据集配置
        with open(data_config, 'r', encoding='utf-8') as f:
            self.data_config = yaml.safe_load(f)
        
        # Jetson Xavier平台参数
        self.jetson_params = {
            'imgsz': 640,
            'batch': 1,
            'fp16': True,  # Jetson支持FP16
            'int8': False,  # 可选INT8量化
            'workspace': 8,  # TensorRT工作空间大小(GB)
        }
    
    def export_onnx(self, output_dir='exports', opset=12, simplify=True):
        """
        导出ONNX格式模型
        
        Args:
            output_dir: 输出目录
            opset: ONNX算子集版本
            simplify: 是否简化ONNX模型
            
        Returns:
            onnx_path: ONNX模型路径
        """
        print("=" * 80)
        print("导出ONNX格式模型")
        print("=" * 80)
        
        # 创建输出目录
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 导出ONNX模型
        onnx_path = self.model.export(
            format='onnx',
            imgsz=self.jetson_params['imgsz'],
            batch=self.jetson_params['batch'],
            opset=opset,
            simplify=simplify,
            dynamic=False,  # 静态批量大小，提高推理速度
            half=self.jetson_params['fp16'],
            workspace=self.jetson_params['workspace'],
            path=output_dir / f"{Path(self.model.model.pt_path).stem}.onnx"
        )
        
        print(f"ONNX模型导出成功: {onnx_path}")
        print(f"ONNX模型大小: {os.path.getsize(onnx_path) / (1024 * 1024):.2f} MB")
        
        return onnx_path
    
    def export_tensorrt(self, output_dir='exports', half=True, workspace=8, int8=False):
        """
        导出TensorRT格式模型
        
        Args:
            output_dir: 输出目录
            half: 是否使用FP16精度
            workspace: TensorRT工作空间大小(GB)
            int8: 是否使用INT8量化
            
        Returns:
            trt_path: TensorRT模型路径
        """
        print("\n" + "=" * 80)
        print("导出TensorRT格式模型")
        print("=" * 80)
        
        # 创建输出目录
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 导出TensorRT模型
        trt_path = self.model.export(
            format='engine',
            imgsz=self.jetson_params['imgsz'],
            batch=self.jetson_params['batch'],
            half=half,
            workspace=workspace,
            int8=int8,
            dynamic=False,  # 静态批量大小
            path=output_dir / f"{Path(self.model.model.pt_path).stem}.engine"
        )
        
        print(f"TensorRT模型导出成功: {trt_path}")
        print(f"TensorRT模型大小: {os.path.getsize(trt_path) / (1024 * 1024):.2f} MB")
        
        return trt_path
    
    def export_all_formats(self, output_dir='exports', opset=12):
        """
        导出所有支持的模型格式
        
        Args:
            output_dir: 输出目录
            opset: ONNX算子集版本
            
        Returns:
            exported_paths: 所有导出模型的路径字典
        """
        exported_paths = {}
        
        # 导出ONNX
        try:
            onnx_path = self.export_onnx(output_dir=output_dir, opset=opset)
            exported_paths['onnx'] = onnx_path
        except Exception as e:
            print(f"ONNX导出失败: {e}")
        
        # 导出TensorRT
        try:
            trt_path = self.export_tensorrt(output_dir=output_dir)
            exported_paths['tensorrt'] = trt_path
        except Exception as e:
            print(f"TensorRT导出失败: {e}")
            print("注意: TensorRT导出需要在支持CUDA的环境中运行")
        
        return exported_paths
    
    def validate_exported_model(self, model_path, format='onnx'):
        """
        验证导出的模型
        
        Args:
            model_path: 导出的模型路径
            format: 模型格式
            
        Returns:
            bool: 验证是否成功
        """
        print(f"\n验证{format}格式模型: {model_path}")
        
        try:
            # 加载导出的模型
            if format == 'onnx':
                import onnxruntime as ort
                
                # 创建ONNX Runtime会话
                session = ort.InferenceSession(model_path)
                
                # 创建测试输入
                input_name = session.get_inputs()[0].name
                input_shape = session.get_inputs()[0].shape
                input_data = np.random.rand(*input_shape).astype(np.float32)
                
                # 推理测试
                output = session.run(None, {input_name: input_data})
                print(f"ONNX模型推理成功，输出形状: {[o.shape for o in output]}")
                
            elif format == 'tensorrt':
                # TensorRT模型验证需要使用TensorRT API
                # 这里提供简单的文件存在性检查
                if os.path.exists(model_path):
                    print("TensorRT模型文件存在")
                    return True
                else:
                    print("TensorRT模型文件不存在")
                    return False
            
            else:
                print(f"不支持的模型格式: {format}")
                return False
                
            return True
            
        except Exception as e:
            print(f"模型验证失败: {e}")
            return False
    
    def get_export_summary(self, exported_paths):
        """
        获取导出摘要
        
        Args:
            exported_paths: 导出模型路径字典
            
        Returns:
            summary: 导出摘要字典
        """
        summary = {
            'original_model': self.model.model.pt_path,
            'exported_formats': list(exported_paths.keys()),
            'jetson_params': self.jetson_params,
            'models': {},
        }
        
        for fmt, path in exported_paths.items():
            summary['models'][fmt] = {
                'path': str(path),
                'size_mb': os.path.getsize(path) / (1024 * 1024) if os.path.exists(path) else 0,
                'valid': self.validate_exported_model(path, fmt),
            }
        
        return summary

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='YOLO11n-pose能量机关模型导出脚本')
    parser.add_argument('--model', type=str, required=True, help='训练好的模型路径')
    parser.add_argument('--data', type=str, default='rune_pose.yaml', help='数据集配置文件路径')
    parser.add_argument('--output', type=str, default='exports', help='导出目录')
    parser.add_argument('--format', type=str, default='all', choices=['onnx', 'tensorrt', 'all'], help='导出格式')
    parser.add_argument('--opset', type=int, default=12, help='ONNX算子集版本')
    parser.add_argument('--half', action='store_true', default=True, help='使用FP16精度')
    parser.add_argument('--int8', action='store_true', default=False, help='使用INT8量化')
    parser.add_argument('--workspace', type=int, default=8, help='TensorRT工作空间大小(GB)')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    # 创建导出器
    exporter = RuneModelExporter(args.model, args.data)
    
    # 更新Jetson参数
    exporter.jetson_params.update({
        'fp16': args.half,
        'int8': args.int8,
        'workspace': args.workspace,
    })
    
    exported_paths = {}
    
    # 导出模型
    if args.format == 'onnx':
        onnx_path = exporter.export_onnx(output_dir=args.output, opset=args.opset)
        exported_paths['onnx'] = onnx_path
    
    elif args.format == 'tensorrt':
        trt_path = exporter.export_tensorrt(output_dir=args.output)
        exported_paths['tensorrt'] = trt_path
    
    else:  # all formats
        exported_paths = exporter.export_all_formats(output_dir=args.output, opset=args.opset)
    
    # 验证导出模型
    print("\n" + "=" * 80)
    print("模型验证结果")
    print("=" * 80)
    
    for fmt, path in exported_paths.items():
        if exporter.validate_exported_model(path, fmt):
            print(f"✅ {fmt.upper()}模型验证通过: {path}")
        else:
            print(f"❌ {fmt.upper()}模型验证失败: {path}")
    
    # 获取导出摘要
    summary = exporter.get_export_summary(exported_paths)
    
    print("\n" + "=" * 80)
    print("导出摘要")
    print("=" * 80)
    print(f"原始模型: {summary['original_model']}")
    print(f"导出格式: {', '.join(summary['exported_formats'])}")
    print(f"Jetson参数: {summary['jetson_params']}")
    print("\n模型详情:")
    for fmt, info in summary['models'].items():
        print(f"- {fmt.upper()}:")
        print(f"  路径: {info['path']}")
        print(f"  大小: {info['size_mb']:.2f} MB")
        print(f"  状态: {'有效' if info['valid'] else '无效'}")
    
    print("\n" + "=" * 80)
    print("模型导出完成！")
    print("=" * 80)
