#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复标签文件格式问题
1. 第0类（rune_center）：只保留边界框信息，删除多余的关键点信息
2. 第1类（armor_module）：保留边界框 + 9个关键点信息（每个关键点3个值）
"""

import os
import glob

# 定义标签目录
label_dirs = ['labels/train', 'labels/val']

# 定义关键点数量
NUM_KEYPOINTS = 9
KEYPOINTS_PER_ROW = NUM_KEYPOINTS * 3  # 每个关键点有x, y, v三个值

for label_dir in label_dirs:
    # 获取所有标签文件
    label_files = glob.glob(os.path.join(label_dir, '*.txt'))
    
    for label_file in label_files:
        print(f"正在处理: {label_file}")
        
        # 读取标签文件内容
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        fixed_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # 分割标签行
            parts = line.split()
            if not parts:
                continue
                
            # 获取类别
            cls = int(parts[0])
            
            # 检查边界框信息是否完整
            if len(parts) < 5:
                print(f"  跳过不完整的标签行: {line}")
                continue
                
            # 提取边界框信息
            bbox = parts[1:5]
            
            if cls == 0:  # rune_center - 边界框 + 默认关键点信息
                # 为类别0添加9个关键点的默认值，确保所有标签行长度一致
                default_kpts = ['0'] * (9 * 3)  # 9个关键点，每个关键点3个默认值(0, 0, 0)
                fixed_line = f"{cls} {' '.join(bbox)} {' '.join(default_kpts)}"
            elif cls == 1:  # armor_module - 边界框 + 关键点信息
                # 强制只保留9个关键点的信息
                fixed_kpts = []
                # 确保我们有足够的关键点数据
                if len(parts) >= 5 + 27:  # 4个边界框值 + 9个关键点×3=27个值
                    # 提取9个关键点的信息
                    for i in range(9):
                        idx = 5 + i * 3
                        if idx + 2 < len(parts):
                            # 保留x和y坐标
                            fixed_kpts.append(parts[idx])
                            fixed_kpts.append(parts[idx+1])
                            # 修复可见性值（确保为0或1）
                            try:
                                v = float(parts[idx+2])
                                if v > 0:
                                    fixed_kpts.append('1')
                                else:
                                    fixed_kpts.append('0')
                            except:
                                fixed_kpts.append('0')
                        else:
                            # 如果关键点信息不完整，使用默认值
                            fixed_kpts.extend(['0', '0', '0'])
                    fixed_line = f"{cls} {' '.join(bbox)} {' '.join(fixed_kpts)}"
                else:
                    # 如果关键点信息不完整，跳过该行
                    print(f"  跳过关键点信息不完整的标签行: {line}")
                    continue
            else:
                # 未知类别，跳过
                print(f"  跳过未知类别: {cls}")
                continue
                
            fixed_lines.append(fixed_line)
        
        # 保存修复后的标签文件
        with open(label_file, 'w') as f:
            f.write('\n'.join(fixed_lines) + '\n')
        
        print(f"  修复完成，共修复 {len(fixed_lines)} 行")

print("所有标签文件修复完成！")
