#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模型融合脚本，用于融合多个YOLO模型的检测结果，提高rune_center的检测稳定性
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time
import os

class MultiModelFusion:
    """多模型融合类"""
    
    def __init__(self, model_paths, model_weights=None, conf_threshold=0.2, iou_threshold=0.4, fusion_strategy='weighted'):
        """
        初始化多模型融合
        
        Args:
            model_paths: 模型路径列表
            model_weights: 模型权重列表，与model_paths对应
            conf_threshold: 置信度阈值
            iou_threshold: IOU阈值
            fusion_strategy: 融合策略，可选值: 'weighted' (加权融合), 'voting' (投票融合), 'nms' (仅NMS)
        """
        self.models = []
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.fusion_strategy = fusion_strategy
        
        # 加载多个模型
        for path in model_paths:
            if os.path.exists(path):
                model = YOLO(path)
                self.models.append(model)
                print(f"✅ 加载模型: {path}")
            else:
                print(f"❌ 模型文件不存在: {path}")
        
        if not self.models:
            raise ValueError("没有加载到有效的模型")
        
        # 设置模型权重
        if model_weights is None:
            # 默认等权重
            self.model_weights = [1.0 / len(self.models) for _ in self.models]
        else:
            # 确保权重长度与模型数量一致
            if len(model_weights) != len(self.models):
                print(f"⚠️  模型权重数量与模型数量不一致，使用默认等权重")
                self.model_weights = [1.0 / len(self.models) for _ in self.models]
            else:
                self.model_weights = model_weights
        
        print(f"\n✅ 成功加载 {len(self.models)} 个模型")
        print(f"📊 融合策略: {fusion_strategy}")
        print(f"⚖️  模型权重: {self.model_weights}")
    
    def non_max_suppression(self, boxes, confidences, iou_threshold):
        """
        非极大值抑制
        
        Args:
            boxes: 边界框列表
            confidences: 置信度列表
            iou_threshold: IOU阈值
            
        Returns:
            过滤后的边界框索引
        """
        if len(boxes) == 0:
            return []
        
        # 转换为numpy数组
        boxes = np.array(boxes)
        confidences = np.array(confidences)
        
        # 获取边界框坐标
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        # 计算面积
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        
        # 按置信度排序
        order = confidences.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            # 计算与其他边界框的IOU
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            
            # 保留IOU小于阈值的边界框
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        
        return keep
    
    def fuse_detections(self, detections_list, img=None):
        """
        融合多个模型的检测结果
        
        Args:
            detections_list: 多个模型的检测结果列表
            img: 输入图像，用于场景复杂度计算
            
        Returns:
            融合后的检测结果
        """
        if not detections_list:
            return []
        
        # 收集所有检测结果
        all_boxes = []
        all_confidences = []
        all_classes = []
        all_keypoints = []
        all_model_indices = []
        
        for model_idx, detections in enumerate(detections_list):
            for det in detections:
                box = det['box']
                conf = det['conf']
                cls = det['class']
                keypoints = det['keypoints']
                
                all_boxes.append(box)
                all_confidences.append(conf)
                all_classes.append(cls)
                all_keypoints.append(keypoints)
                all_model_indices.append(model_idx)
        
        if not all_boxes:
            return []
        
        # 根据融合策略处理
        if self.fusion_strategy == 'nms':
            # 仅使用非极大值抑制
            keep_indices = self.non_max_suppression(all_boxes, all_confidences, self.iou_threshold)
            
            fused_results = []
            for i in keep_indices:
                result = {
                    'box': all_boxes[i],
                    'conf': all_confidences[i],
                    'class': all_classes[i],
                    'keypoints': all_keypoints[i]
                }
                fused_results.append(result)
                
        elif self.fusion_strategy == 'weighted':
            # 加权融合：基于模型权重和置信度
            fused_results = self._weighted_fusion(all_boxes, all_confidences, all_classes, all_keypoints, all_model_indices, img)
            
        elif self.fusion_strategy == 'voting':
            # 投票融合：基于多个模型的一致检测
            fused_results = self._voting_fusion(all_boxes, all_confidences, all_classes, all_keypoints)
            
        else:
            # 默认使用非极大值抑制
            keep_indices = self.non_max_suppression(all_boxes, all_confidences, self.iou_threshold)
            
            fused_results = []
            for i in keep_indices:
                result = {
                    'box': all_boxes[i],
                    'conf': all_confidences[i],
                    'class': all_classes[i],
                    'keypoints': all_keypoints[i]
                }
                fused_results.append(result)
        
        return fused_results
    
    def calculate_scene_complexity(self, img):
        """
        计算场景复杂度
        
        Args:
            img: 输入图像
            
        Returns:
            场景复杂度得分 (0-1)
        """
        if img is None:
            return 0.5
        
        # 转换为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 计算边缘密度
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])
        
        # 计算纹理复杂度
        texture = cv2.Laplacian(gray, cv2.CV_64F).var()
        texture_complexity = min(texture / 1000, 1.0)
        
        # 计算亮度变化
        brightness = np.mean(gray)
        brightness_variation = abs(brightness - 128) / 128
        
        # 综合计算场景复杂度
        complexity = 0.3 * edge_density + 0.4 * texture_complexity + 0.3 * brightness_variation
        
        return min(max(complexity, 0), 1)
    
    def adjust_model_weights(self, model_performance):
        """
        根据模型性能动态调整权重
        
        Args:
            model_performance: 模型性能列表，每个元素是 (model_idx, accuracy, reliability)
            
        Returns:
            调整后的权重列表
        """
        if not model_performance:
            return self.model_weights
        
        # 计算每个模型的综合得分
        model_scores = []
        for model_idx, accuracy, reliability in model_performance:
            score = 0.6 * accuracy + 0.4 * reliability
            model_scores.append((model_idx, score))
        
        # 归一化得分作为新权重
        total_score = sum(score for _, score in model_scores)
        if total_score == 0:
            return self.model_weights
        
        new_weights = [0] * len(self.model_weights)
        for model_idx, score in model_scores:
            new_weights[model_idx] = score / total_score
        
        return new_weights
    
    def _weighted_fusion(self, boxes, confidences, classes, keypoints, model_indices, img=None):
        """
        加权融合策略
        
        Args:
            boxes: 边界框列表
            confidences: 置信度列表
            classes: 类别列表
            keypoints: 关键点列表
            model_indices: 模型索引列表
            img: 输入图像，用于场景复杂度计算
            
        Returns:
            融合后的检测结果
        """
        # 应用非极大值抑制，获取候选检测
        keep_indices = self.non_max_suppression(boxes, confidences, self.iou_threshold)
        
        # 计算场景复杂度
        scene_complexity = self.calculate_scene_complexity(img) if img is not None else 0.5
        
        # 对每个候选检测，计算加权置信度
        weighted_results = []
        for i in keep_indices:
            # 获取模型权重
            model_idx = model_indices[i]
            model_weight = self.model_weights[model_idx]
            
            # 根据场景复杂度调整权重
            if scene_complexity > 0.7:
                # 复杂场景下增加高可靠性模型的权重
                model_weight *= (1 + 0.3 * scene_complexity)
            
            # 计算加权置信度 - 确保置信度不会过低
            base_conf = confidences[i]
            
            # 使用加权平均而不是简单相乘
            # 模型权重作为加权因子，同时保持原始置信度的重要性
            weighted_conf = base_conf * 0.6 + base_conf * model_weight * 0.4
            
            # 根据场景复杂度调整置信度
            if classes[i] == 'rune_center':
                # 场景复杂度调整 - 适用于所有复杂度级别
                if scene_complexity > 0.5:
                    # 复杂场景：增加置信度
                    adjusted_conf = weighted_conf * (1 + 0.4 * scene_complexity)
                else:
                    # 简单场景：显著增加置信度
                    adjusted_conf = weighted_conf * (1 + 0.3 * (1 - scene_complexity))
                
                # 确保置信度不会过高，同时保证最低置信度
                adjusted_conf = min(max(adjusted_conf, 0.4), 1.0)
            else:
                adjusted_conf = weighted_conf
            
            result = {
                'box': boxes[i],
                'conf': adjusted_conf,
                'class': classes[i],
                'keypoints': keypoints[i],
                'scene_complexity': scene_complexity
            }
            weighted_results.append(result)
        
        # 按置信度排序
        weighted_results.sort(key=lambda x: x['conf'], reverse=True)
        
        return weighted_results
    
    def _voting_fusion(self, boxes, confidences, classes, keypoints):
        """
        投票融合策略
        
        Args:
            boxes: 边界框列表
            confidences: 置信度列表
            classes: 类别列表
            keypoints: 关键点列表
            
        Returns:
            融合后的检测结果
        """
        if not boxes:
            return []
        
        # 聚类相似的检测结果
        clusters = []
        for i, (box, conf, cls, kpt) in enumerate(zip(boxes, confidences, classes, keypoints)):
            # 如果置信度低于阈值，跳过
            if conf < self.conf_threshold:
                continue
            
            # 检查是否与现有聚类匹配
            matched = False
            for cluster in clusters:
                cluster_box = cluster['box']
                # 计算IOU
                iou = self._calculate_iou(box, cluster_box)
                if iou >= self.iou_threshold and cls == cluster['class']:
                    # 添加到现有聚类
                    cluster['detections'].append({
                        'box': box,
                        'conf': conf,
                        'keypoints': kpt
                    })
                    matched = True
                    break
            
            if not matched:
                # 创建新聚类
                clusters.append({
                    'class': cls,
                    'box': box,
                    'detections': [{
                        'box': box,
                        'conf': conf,
                        'keypoints': kpt
                    }]
                })
        
        # 处理聚类结果
        fused_results = []
        for cluster in clusters:
            # 只保留至少有2个模型检测到的结果
            if len(cluster['detections']) >= 2:
                # 计算平均边界框
                avg_box = self._calculate_average_box([det['box'] for det in cluster['detections']])
                # 计算平均置信度
                avg_conf = sum(det['conf'] for det in cluster['detections']) / len(cluster['detections'])
                # 选择关键点质量最好的检测
                best_kpt = self._select_best_keypoints([det['keypoints'] for det in cluster['detections']])
                
                result = {
                    'box': avg_box,
                    'conf': avg_conf,
                    'class': cluster['class'],
                    'keypoints': best_kpt
                }
                fused_results.append(result)
        
        # 按置信度排序
        fused_results.sort(key=lambda x: x['conf'], reverse=True)
        
        return fused_results
    
    def _calculate_iou(self, box1, box2):
        """
        计算两个边界框的IOU
        
        Args:
            box1: 第一个边界框 [x1, y1, x2, y2]
            box2: 第二个边界框 [x1, y1, x2, y2]
            
        Returns:
            IOU值
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
        box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
        box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
        
        iou = inter_area / float(box1_area + box2_area - inter_area)
        return iou
    
    def _calculate_average_box(self, boxes):
        """
        计算多个边界框的平均值
        
        Args:
            boxes: 边界框列表
            
        Returns:
            平均边界框
        """
        if not boxes:
            return []
        
        x1 = sum(box[0] for box in boxes) / len(boxes)
        y1 = sum(box[1] for box in boxes) / len(boxes)
        x2 = sum(box[2] for box in boxes) / len(boxes)
        y2 = sum(box[3] for box in boxes) / len(boxes)
        
        return [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]
    
    def _select_best_keypoints(self, keypoints_list):
        """
        选择质量最好的关键点
        
        Args:
            keypoints_list: 关键点列表
            
        Returns:
            质量最好的关键点
        """
        if not keypoints_list:
            return []
        
        # 选择有效关键点数量最多的
        best_kpt = None
        max_valid = -1
        
        for kpt in keypoints_list:
            valid_count = sum(1 for p in kpt if p[0] != 0 and p[1] != 0)
            if valid_count > max_valid:
                max_valid = valid_count
                best_kpt = kpt
        
        return best_kpt if best_kpt else keypoints_list[0]
    
    def detect(self, img):
        """
        使用多模型进行检测并融合结果
        
        Args:
            img: 输入图像
            
        Returns:
            融合后的检测结果
        """
        # 分别处理rune_center和armor_module
        rune_detections_list = []
        armor_detections = []
        
        # 使用每个模型进行检测
        for model in self.models:
            results = model(img, conf=self.conf_threshold, iou=self.iou_threshold)
            
            # 解析检测结果
            for result in results:
                boxes = result.boxes
                keypoints = result.keypoints
                
                for j, box in enumerate(boxes):
                    conf = box.conf[0].item()
                    cls = int(box.cls[0].item())
                    cls_name = model.names[cls]
                    
                    # 只处理rune_center和armor_module
                    if cls_name not in ['rune_center', 'armor_module']:
                        continue
                    
                    # 获取边界框
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    
                    # 获取关键点
                    kpt = []
                    if keypoints is not None:
                        kpt = keypoints[j].xy[0].tolist()
                    
                    detection = {
                        'box': [x1, y1, x2, y2],
                        'conf': conf,
                        'class': cls_name,
                        'keypoints': kpt
                    }
                    
                    if cls_name == 'rune_center':
                        # 对rune_center使用多模型融合
                        if not rune_detections_list:
                            # 初始化每个模型的检测结果列表
                            rune_detections_list = [[] for _ in self.models]
                        rune_detections_list[self.models.index(model)].append(detection)
                    else:  # armor_module
                        # 对armor_module只使用最佳模型的结果
                        if model == self.models[0]:  # 第一个模型是best.pt，使用它的结果
                            # 应用原始的armor_module检测阈值
                            if conf >= 0.5:
                                armor_detections.append(detection)
        
        # 融合rune_center的检测结果
        fused_rune_results = []
        if rune_detections_list:
            fused_rune_results = self.fuse_detections(rune_detections_list, img)
        
        # 对armor_module应用非极大值抑制，避免重复检测
        if armor_detections:
            armor_boxes = [det['box'] for det in armor_detections]
            armor_confidences = [det['conf'] for det in armor_detections]
            keep_indices = self.non_max_suppression(armor_boxes, armor_confidences, self.iou_threshold)
            armor_detections = [armor_detections[i] for i in keep_indices]
        
        # 合并结果
        final_results = armor_detections + fused_rune_results
        
        return final_results

def evaluate_model_performance(fusion, test_images):
    """评估模型性能"""
    model_performance = []
    
    for model_idx, model in enumerate(fusion.models):
        correct_detections = 0
        total_detections = 0
        reliable_detections = 0
        
        for img_path in test_images[:10]:  # 使用前10张图像评估
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            # 使用单个模型进行检测
            results = model(img, conf=0.15, iou=0.4)
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    conf = box.conf[0].item()
                    cls = int(box.cls[0].item())
                    cls_name = model.names[cls]
                    
                    if cls_name == 'rune_center':
                        total_detections += 1
                        if conf > 0.6:
                            correct_detections += 1
                        if conf > 0.7:
                            reliable_detections += 1
        
        # 计算准确率和可靠性
        accuracy = correct_detections / total_detections if total_detections > 0 else 0
        reliability = reliable_detections / total_detections if total_detections > 0 else 0
        
        model_performance.append((model_idx, accuracy, reliability))
        print(f"模型 {model_idx} 性能: 准确率={accuracy:.3f}, 可靠性={reliability:.3f}")
    
    return model_performance

def test_multi_model_fusion():
    """测试多模型融合"""
    # 模型路径列表 - 添加不同训练轮次和数据增强策略的模型
    model_paths = [
        "runs/pose/rune_pose_model_stage2/weights/best.pt",
        "runs/pose/rune_pose_model_stage2/weights/last.pt",
        "runs/pose/rune_pose_model_stage2/weights/epoch70.pt",
        "runs/pose/rune_pose_model_stage2/weights/epoch60.pt",
        "runs/pose/rune_pose_model_stage2/weights/epoch50.pt"
    ]
    
    # 模型权重 - best.pt权重最高，其次是last.pt，然后是不同训练轮次的模型
    model_weights = [0.4, 0.3, 0.1, 0.1, 0.1]
    
    # 初始化多模型融合
    try:
        fusion = MultiModelFusion(
            model_paths,
            model_weights=model_weights,
            conf_threshold=0.15,
            iou_threshold=0.4,
            fusion_strategy='weighted'  # 使用加权融合策略
        )
    except ValueError as e:
        print(f"❌ 初始化失败: {e}")
        return
    
    # 准备测试图像
    test_images = []
    val_images_dir = "images/val"
    if os.path.exists(val_images_dir):
        test_images = [os.path.join(val_images_dir, f) for f in os.listdir(val_images_dir) 
                      if f.endswith('.jpg') or f.endswith('.png')]
    
    if not test_images:
        print(f"❌ 没有找到测试图像: {val_images_dir}")
        return
    
    print(f"✅ 找到测试图像数量: {len(test_images)}")
    
    # 评估模型性能并动态调整权重
    print(f"\n=== 评估模型性能 ===")
    model_performance = evaluate_model_performance(fusion, test_images)
    
    # 动态调整模型权重
    if model_performance:
        new_weights = fusion.adjust_model_weights(model_performance)
        print(f"\n✅ 动态调整模型权重: {new_weights}")
        fusion.model_weights = new_weights
    
    # 测试结果统计
    rune_center_detections = 0
    armor_module_detections = 0
    unstable_rune_detections = 0
    total_scene_complexity = 0
    
    print(f"\n=== 开始测试多模型融合效果 ===")
    
    for i, img_path in enumerate(test_images[:20]):  # 测试前20张图像
        print(f"\n测试图像 {i+1}/{20}: {img_path}")
        
        try:
            # 读取图像
            img = cv2.imread(img_path)
            if img is None:
                print(f"  ❌ 无法读取图像: {img_path}")
                continue
            
            # 使用多模型融合进行检测
            start_time = time.time()
            results = fusion.detect(img)
            end_time = time.time()
            
            print(f"  ⏱️  检测时间: {(end_time - start_time):.3f}秒")
            
            # 处理检测结果
            for result in results:
                box = result['box']
                conf = result['conf']
                cls_name = result['class']
                keypoints = result['keypoints']
                scene_complexity = result.get('scene_complexity', 0.5)
                
                # 计算边界框大小
                box_size = (box[2] - box[0]) * (box[3] - box[1])
                
                if cls_name == "rune_center":
                    rune_center_detections += 1
                    total_scene_complexity += scene_complexity
                    print(f"  ✅ 检测到 rune_center - 置信度: {conf:.3f}, 大小: {box_size}, 场景复杂度: {scene_complexity:.2f}")
                    
                    # 检查置信度是否稳定
                    if conf < 0.6:
                        unstable_rune_detections += 1
                        print(f"    ⚠️  置信度较低: {conf:.3f}")
                    
                    # 检查关键点
                    if keypoints:
                        # 计算有效关键点数量
                        valid_kpts = sum(1 for k in keypoints if k[0] != 0 and k[1] != 0)
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
        avg_scene_complexity = total_scene_complexity / rune_center_detections
        print(f"rune_center 检测稳定率: {stability_rate:.1f}%")
        print(f"平均场景复杂度: {avg_scene_complexity:.2f}")
    
    # 性能建议
    print(f"\n=== 性能改进建议 ===")
    if unstable_rune_detections > rune_center_detections * 0.5:
        print("⚠️  rune_center 检测稳定性较差，建议:")
        print("   1. 增加更多模型到融合系统中")
        print("   2. 调整融合策略和阈值")
        print("   3. 进一步优化单个模型的性能")
    else:
        print("✅ rune_center 检测稳定性良好")
    
    print(f"\n=== 测试完成 ===")

if __name__ == "__main__":
    test_multi_model_fusion()
