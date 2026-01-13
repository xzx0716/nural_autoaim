#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO11n-pose能量机关实时推理脚本
用于演示模型在视频或摄像头输入下的实时推理效果
- 支持视频文件和USB摄像头输入
- 实时显示检测结果（装甲模块检测框和WS2812灯珠关键点）
- 性能监控（FPS显示）
- 可视化参数可配置
"""

import os
import argparse
import cv2
import numpy as np
from ultralytics import YOLO
import time
from pathlib import Path

class RunePoseInferencer:
    """能量机关姿态检测推理器"""
    
    def __init__(self, model_path, device='', conf_threshold=0.5, iou_threshold=0.5):
        """
        初始化推理器
        
        Args:
            model_path: 训练好的模型路径
            device: 推理设备 (空字符串表示自动选择)
            conf_threshold: 置信度阈值
            iou_threshold: IoU阈值
        """
        # 加载模型
        self.model = YOLO(model_path)
        
        # 推理参数
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        
        # 初始化性能统计
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        # 可视化配置
        self.line_thickness = 2  # 线条粗细
        self.font_scale = 0.5     # 字体大小
        self.box_color = (255, 0, 0)  # 检测框颜色 (BGR)
        self.kpt_color = (0, 255, 0)   # 关键点颜色 (BGR)
        self.text_color = (255, 255, 255)  # 文本颜色 (BGR)
        self.text_bg_color = (0, 0, 0)    # 文本背景颜色 (BGR)
    
    def process_frame(self, frame):
        """
        处理单帧图像
        
        Args:
            frame: 输入图像 (BGR格式)
            
        Returns:
            frame: 带检测结果的输出图像
        """
        # 模型推理
        results = self.model.predict(
            frame,
            imgsz=640,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False
        )
        
        # 更新性能统计
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        if elapsed_time >= 1.0:
            self.fps = self.frame_count / elapsed_time
            self.frame_count = 0
            self.start_time = time.time()
        
        # 处理推理结果
        for result in results:
            # 绘制检测框
            if result.boxes is not None:
                for box in result.boxes:
                    # 转换为整数坐标
                    xyxy = box.xyxy[0].int().tolist()
                    confidence = box.conf[0].item()
                    class_id = box.cls[0].item()
                    
                    # 绘制矩形框
                    cv2.rectangle(
                        frame, 
                        (xyxy[0], xyxy[1]), 
                        (xyxy[2], xyxy[3]), 
                        self.box_color, 
                        self.line_thickness
                    )
                    
                    # 绘制类别和置信度
                    label = f"Armor: {confidence:.2f}"
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, 1)
                    label_width, label_height = label_size
                    
                    # 绘制文本背景
                    cv2.rectangle(
                        frame, 
                        (xyxy[0], xyxy[1] - label_height - 5), 
                        (xyxy[0] + label_width, xyxy[1]), 
                        self.text_bg_color, 
                        -1
                    )
                    
                    # 绘制文本
                    cv2.putText(
                        frame, 
                        label, 
                        (xyxy[0], xyxy[1] - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        self.font_scale, 
                        self.text_color, 
                        1
                    )
            
            # 绘制关键点
            if result.keypoints is not None:
                for kpt in result.keypoints:
                    # 关键点坐标
                    xy = kpt.xy[0].int().tolist()
                    # 关键点可见性
                    confidence = kpt.conf[0].tolist()
                    
                    # 绘制每个关键点
                    for i, (x, y) in enumerate(xy):
                        if confidence[i] > 0.5:  # 只绘制置信度高的关键点
                            # 绘制实心圆
                            cv2.circle(
                                frame, 
                                (x, y), 
                                5,  # 半径
                                self.kpt_color, 
                                -1   # 实心
                            )
                            
                            # 添加关键点索引
                            kpt_label = f"{i}"
                            cv2.putText(
                                frame, 
                                kpt_label, 
                                (x + 8, y - 8), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                0.4, 
                                self.text_color, 
                                1
                            )
        
        # 显示FPS
        fps_label = f"FPS: {self.fps:.2f}"
        cv2.putText(
            frame, 
            fps_label, 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.8, 
            (0, 255, 0), 
            2
        )
        
        return frame
    
    def run(self, input_source, output_path=None):
        """
        运行推理
        
        Args:
            input_source: 输入源 (摄像头ID或视频文件路径)
            output_path: 输出视频路径 (可选)
        """
        # 打开视频源
        if isinstance(input_source, int) or input_source.isdigit():
            # 摄像头输入
            cap = cv2.VideoCapture(int(input_source))
            print(f"正在使用摄像头: {input_source}")
        else:
            # 视频文件输入
            if not os.path.exists(input_source):
                print(f"错误: 视频文件不存在 - {input_source}")
                return
            cap = cv2.VideoCapture(input_source)
            print(f"正在处理视频: {input_source}")
        
        # 检查视频源是否打开成功
        if not cap.isOpened():
            print(f"错误: 无法打开视频源 - {input_source}")
            return
        
        # 获取视频属性
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 初始化视频写入器
        video_writer = None
        if output_path:
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"输出视频将保存到: {output_path}")
        
        # 显示窗口
        window_name = "YOLO11n-pose Rune Detection"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)
        
        print("按 'q' 键退出...")
        
        while True:
            # 读取一帧
            ret, frame = cap.read()
            if not ret:
                break
            
            # 处理帧
            processed_frame = self.process_frame(frame)
            
            # 显示结果
            cv2.imshow(window_name, processed_frame)
            
            # 写入视频
            if video_writer:
                video_writer.write(processed_frame)
            
            # 检查按键
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):  # 空格键暂停
                cv2.waitKey(0)
        
        # 清理资源
        cap.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()
        
        print("推理完成!")

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='YOLO11n-pose能量机关实时推理脚本')
    parser.add_argument('--model', type=str, required=True, help='训练好的模型路径')
    parser.add_argument('--input', type=str, default='0', help='输入源 (摄像头ID或视频文件路径)')
    parser.add_argument('--output', type=str, default=None, help='输出视频路径')
    parser.add_argument('--device', type=str, default='', help='推理设备 (cpu/gpu)')
    parser.add_argument('--conf', type=float, default=0.5, help='置信度阈值')
    parser.add_argument('--iou', type=float, default=0.5, help='IoU阈值')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    # 创建推理器
    inferencer = RunePoseInferencer(
        model_path=args.model,
        device=args.device,
        conf_threshold=args.conf,
        iou_threshold=args.iou
    )
    
    # 运行推理
    inferencer.run(
        input_source=args.input,
        output_path=args.output
    )
