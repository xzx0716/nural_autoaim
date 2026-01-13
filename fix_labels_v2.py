#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复标签文件格式问题 v2
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
            try:
                cls = int(parts[0])
            except:
                print(f"  跳过无效类别: {parts[0]}")
                continue
            
            # 检查边界框信息是否完整
            if len(parts) < 5:
                print(f"  跳过不完整的标签行: {line}")
                continue
                
            # 提取边界框信息
            bbox = parts[1:5]
            
            if cls == 0:  # rune_center - 只有边界框信息
                # 只保留边界框信息
                fixed_line = f"{cls} {' '.join(bbox)}"
                fixed_lines.append(fixed_line)
            elif cls == 1:  # armor_module - 边界框 + 关键点信息
                # 只保留9个关键点的信息，每个关键点3个值
                kpt_values = []
                
                # 从第5个元素开始，只取前27个值（9个关键点×3个值）
                if len(parts) >= 5:
                    # 只提取需要的部分
                    raw_kpts = parts[5:5+27]
                    
                    # 确保我们有足够的数据
                    if len(raw_kpts) >= 27:
                        # 处理每个关键点
                        for i in range(0, 27, 3):
                            # 保留x和y坐标
                            kpt_values.append(raw_kpts[i])
                            kpt_values.append(raw_kpts[i+1])
                            
                            # 修复可见性值
                            try:
                                v = float(raw_kpts[i+2])
                                if v > 0:
                                    kpt_values.append('1')
                                else:
                                    kpt_values.append('0')
                            except:
                                kpt_values.append('0')
                        
                        fixed_line = f"{cls} {' '.join(bbox)} {' '.join(kpt_values)}"
                        fixed_lines.append(fixed_line)
                    else:
                        print(f"  关键点信息不完整，跳过: {line}")
                else:
                    print(f"  缺少关键点信息，跳过: {line}")
            else:
                print(f"  未知类别，跳过: {cls}")
        
        # 保存修复后的标签文件
        with open(label_file, 'w') as f:
            f.write('\n'.join(fixed_lines) + '\n')
        
        print(f"  修复完成，共修复 {len(fixed_lines)} 行")

print("所有标签文件修复完成！")
