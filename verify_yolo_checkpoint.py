from ultralytics import YOLO
import os

def verify_yolo_checkpoint(checkpoint_path):
    """
    使用Ultralytics YOLO官方方法验证检查点文件是否损坏
    
    Args:
        checkpoint_path: 检查点文件路径
    
    Returns:
        bool: True表示损坏，False表示正常
    """
    print(f"验证YOLO检查点文件: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ 文件不存在: {checkpoint_path}")
        return True
    
    try:
        # 使用YOLO官方方法加载检查点
        model = YOLO(checkpoint_path)
        print(f"✅ 成功使用YOLO加载检查点")
        
        # 尝试获取模型状态字典
        state_dict = model.model.state_dict()
        print(f"✅ 成功获取模型状态字典，包含 {len(state_dict)} 个参数")
        
        # 检查状态字典是否包含NaN或Inf值
        import torch
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
            
    except Exception as e:
        print(f"❌ 加载检查点时发生错误: {e}")
        return True

if __name__ == '__main__':
    # 验证可能损坏的检查点文件
    checkpoint_paths = [
        'runs/pose/rune_pose_model_stage1_attention/weights/last.pt',
        'runs/pose/rune_pose_model_stage1_attention/weights/best.pt',
        'runs/pose/rune_pose_model_stage1_attention/weights/epoch0.pt',
        'runs/pose/rune_pose_model_stage2/weights/last.pt',
        'runs/pose/rune_pose_model_stage2/weights/best.pt',
        'runs/pose/rune_pose_model_stage1/weights/last.pt',
        'runs/pose/rune_pose_model_stage1/weights/best.pt'
    ]
    
    for path in checkpoint_paths:
        print("=" * 80)
        is_corrupted = verify_yolo_checkpoint(path)
        print(f"结果: {'损坏' if is_corrupted else '正常'}")
        print("=" * 80)
        print()
