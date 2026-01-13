import torch
import os

def verify_checkpoint(checkpoint_path):
    """
    验证检查点文件是否损坏（包含NaN或Inf值）
    
    Args:
        checkpoint_path: 检查点文件路径
    
    Returns:
        bool: True表示损坏，False表示正常
    """
    print(f"验证检查点文件: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ 文件不存在: {checkpoint_path}")
        return True
    
    try:
        # 加载检查点（使用weights_only=False以兼容Ultralytics模型）
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        print(f"✅ 成功加载检查点")
        
        # 打印检查点结构
        print(f"✅ 检查点包含的键: {list(checkpoint.keys())}")
        
        # 检查是否包含模型状态字典
        if 'model' in checkpoint:
            model_data = checkpoint['model']
            print(f"✅ 找到 'model' 键")
            
            # 检查模型数据类型
            if isinstance(model_data, dict):
                print(f"✅ 模型数据是字典，包含 {len(model_data)} 个参数")
                # 检查权重是否包含NaN或Inf
                has_nan = False
                has_inf = False
                
                for key, value in model_data.items():
                    if torch.is_tensor(value):
                        if torch.isnan(value).any():
                            has_nan = True
                            print(f"❌ 权重 '{key}' 包含 NaN 值")
                        if torch.isinf(value).any():
                            has_inf = True
                            print(f"❌ 权重 '{key}' 包含 Inf 值")
                
                if has_nan or has_inf:
                    print(f"\n❌ 检查点文件已损坏，包含 {'NaN' if has_nan else ''}{' 和 ' if has_nan and has_inf else ''}{'Inf' if has_inf else ''} 值")
                    return True
                else:
                    print("\n✅ 检查点文件正常，无NaN或Inf值")
                    return False
            else:
                print(f"⚠️  模型数据类型不是字典: {type(model_data)}")
                # 尝试直接检查模型参数
                try:
                    if hasattr(model_data, 'state_dict'):
                        state_dict = model_data.state_dict()
                        print(f"✅ 从模型获取状态字典，包含 {len(state_dict)} 个参数")
                        has_nan = False
                        has_inf = False
                        
                        for key, value in state_dict.items():
                            if torch.is_tensor(value):
                                if torch.isnan(value).any():
                                    has_nan = True
                                    print(f"❌ 权重 '{key}' 包含 NaN 值")
                                if torch.isinf(value).any():
                                    has_inf = True
                                    print(f"❌ 权重 '{key}' 包含 Inf 值")
                        
                        if has_nan or has_inf:
                            print(f"\n❌ 检查点文件已损坏，包含 {'NaN' if has_nan else ''}{' 和 ' if has_nan and has_inf else ''}{'Inf' if has_inf else ''} 值")
                            return True
                        else:
                            print("\n✅ 检查点文件正常，无NaN或Inf值")
                            return False
                    else:
                        print(f"❌ 无法获取模型状态字典")
                        return True
                except Exception as e:
                    print(f"❌ 检查模型参数时发生错误: {e}")
                    return True
        else:
            print(f"❌ 检查点中没有 'model' 键")
            return True
            
    except Exception as e:
        print(f"❌ 加载检查点时发生错误: {e}")
        return True

if __name__ == '__main__':
    # 验证可能损坏的检查点文件
    checkpoint_paths = [
        'runs/pose/rune_pose_model_stage2/weights/last.pt',
        'runs/pose/rune_pose_model_stage2/weights/best.pt',
        'runs/pose/rune_pose_model_stage1/weights/last.pt',
        'runs/pose/rune_pose_model_stage1/weights/best.pt'
    ]
    
    for path in checkpoint_paths:
        print("=" * 80)
        is_corrupted = verify_checkpoint(path)
        print(f"结果: {'损坏' if is_corrupted else '正常'}")
        print("=" * 80)
        print()
