#!/usr/bin/env python3
import os

def check_pose_labels(label_dir):
    """
    检查姿态标签是否符合YOLO11-pose要求
    
    YOLO11-pose要求：
    - 类别0 (rune_center)：只有边界框，格式为 [class, x, y, w, h]
    - 类别1 (armor_module)：边界框 + 9个关键点，格式为 [class, x, y, w, h, kpt1_x, kpt1_y, kpt1_v, ...]
    - 每个关键点包含3个值：x坐标, y坐标, 可见性(v=0或1)
    - 总关键点数量：9个
    """
    print(f"检查目录: {label_dir}")
    
    # 统计信息
    total_files = 0
    total_lines = 0
    valid_pose_lines = 0
    invalid_pose_lines = 0
    
    # 获取所有标签文件
    label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
    total_files = len(label_files)
    
    # 检查每个标签文件
    for i, label_file in enumerate(label_files):
        file_path = os.path.join(label_dir, label_file)
        
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            total_lines += 1
            parts = line.split()
            
            if len(parts) < 5:
                print(f"文件 {label_file}: 行 '{line}' - 标签不完整，至少需要5个值")
                invalid_pose_lines += 1
                continue
            
            cls = int(parts[0])
            
            if cls == 0:  # rune_center - 只有边界框
                if len(parts) == 5:
                    # 正确的格式
                    pass
                else:
                    print(f"文件 {label_file}: 行 '{line}' - 类别0应该只有5个值，实际有{len(parts)}个")
                    invalid_pose_lines += 1
            elif cls == 1:  # armor_module - 边界框 + 关键点
                expected_len = 5 + 9 * 3  # 5个边界框值 + 9个关键点×3
                if len(parts) == expected_len:
                    # 检查关键点可见性值
                    for j in range(5 + 2, expected_len, 3):
                        v = parts[j]
                        if v not in ['0', '1']:
                            print(f"文件 {label_file}: 行 '{line}' - 关键点可见性值 {v} 无效，必须是0或1")
                            invalid_pose_lines += 1
                            break
                    else:
                        # 所有可见性值都有效
                        valid_pose_lines += 1
                else:
                    print(f"文件 {label_file}: 行 '{line}' - 类别1应该有{expected_len}个值，实际有{len(parts)}个")
                    invalid_pose_lines += 1
            else:
                print(f"文件 {label_file}: 行 '{line}' - 未知类别 {cls}")
                invalid_pose_lines += 1
    
    print(f"\n=== 检查结果 ===")
    print(f"总文件数: {total_files}")
    print(f"总行数: {total_lines}")
    print(f"有效姿态标签行数: {valid_pose_lines}")
    print(f"无效姿态标签行数: {invalid_pose_lines}")
    print(f"姿态标签有效率: {(valid_pose_lines / total_lines * 100):.2f}%")
    
    return valid_pose_lines, invalid_pose_lines

# 检查训练集和验证集标签
print("\n=== 检查训练集标签 ===")
train_valid, train_invalid = check_pose_labels('labels/train')

print("\n=== 检查验证集标签 ===")
val_valid, val_invalid = check_pose_labels('labels/val')

print(f"\n=== 总体结果 ===")
total_valid = train_valid + val_valid
total_invalid = train_invalid + val_invalid
total = total_valid + total_invalid
print(f"总有效姿态标签: {total_valid}")
print(f"总无效姿态标签: {total_invalid}")
print(f"总体有效率: {(total_valid / total * 100):.2f}%")

if total_invalid == 0:
    print("\n✅ 所有姿态标签格式正确，符合YOLO11-pose要求！")
else:
    print("\n❌ 发现无效姿态标签，请检查并修复！")