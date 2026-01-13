#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查标签文件格式，特别是rune_center的关键点数据
"""

import os
import glob

def check_labels():
    # 检查训练集标签
    train_files = glob.glob('labels/train/*.txt')
    val_files = glob.glob('labels/val/*.txt')
    
    print(f"\n=== 检查标签文件格式 ===")
    print(f"训练集标签文件数: {len(train_files)}")
    print(f"验证集标签文件数: {len(val_files)}")
    
    # 检查前5个训练集标签文件
    print(f"\n=== 检查前5个训练集标签文件 ===")
    for file_path in train_files[:5]:
        print(f"\n文件: {file_path}")
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                cls = int(parts[0])
                
                # 检查标签格式
                if len(parts) != 32:  # class_id + 4个bbox值 + 9个关键点×3=27个值
                    print(f"  ❌ 标签格式错误: {line} (长度: {len(parts)})")
                else:
                    cls_name = "rune_center" if cls == 0 else "armor_module"
                    print(f"  ✅ {cls_name} - 格式正确")
                    
                    # 检查关键点数据
                    kpts = parts[5:32]
                    all_zero = all(float(x) == 0 for x in kpts)
                    if all_zero:
                        print(f"  ⚠️  所有关键点值为0")
                    else:
                        non_zero_count = sum(1 for x in kpts if float(x) != 0)
                        print(f"  ✅ 非零关键点值数量: {non_zero_count}")
                        
        except Exception as e:
            print(f"  ❌ 读取文件错误: {e}")
    
    # 统计各类别的数量
    print(f"\n=== 统计各类别标签数量 ===")
    rune_center_count = 0
    armor_module_count = 0
    
    for file_path in train_files + val_files:
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                cls = int(parts[0])
                
                if cls == 0:
                    rune_center_count += 1
                elif cls == 1:
                    armor_module_count += 1
        except:
            continue
    
    print(f"rune_center标签数量: {rune_center_count}")
    print(f"armor_module标签数量: {armor_module_count}")
    
    # 检查rune_center标签的关键点数据分布
    print(f"\n=== 检查rune_center关键点数据分布 ===")
    rune_with_kpts = 0
    rune_all_zero = 0
    
    for file_path in train_files + val_files:
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                cls = int(parts[0])
                
                if cls == 0:
                    kpts = parts[5:32]
                    all_zero = all(float(x) == 0 for x in kpts)
                    
                    if all_zero:
                        rune_all_zero += 1
                    else:
                        rune_with_kpts += 1
        except:
            continue
    
    print(f"rune_center带非零关键点数据: {rune_with_kpts}")
    print(f"rune_center全零关键点数据: {rune_all_zero}")

if __name__ == "__main__":
    check_labels()
