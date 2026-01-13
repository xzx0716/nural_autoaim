#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO11n-pose Training Script for Rune Detection
赛事专属配置：
- 适配灯环一体化PCB设计的WS2812灯珠检测
- 旋转速度>0.4转/秒的能量机关识别
- 5米距离命中率考核
- 嵌入式部署优化
"""

import os
import yaml
import torch
# 启用cuDNN并设置为确定性模式，提高稳定性
torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
from ultralytics import YOLO
from ultralytics.utils import callbacks
import datetime
import subprocess
import time
import webbrowser

# 加载数据集配置
with open('rune_pose.yaml', 'r', encoding='utf-8') as f:
    data_config = yaml.safe_load(f)

# 训练参数配置
train_params = {
    'data': 'rune_pose.yaml',
    'imgsz': 640,                # 图像大小
    'batch': 8,                  # 批处理大小（降低以减少GPU内存使用）
    'lr0': 0.005,                # 初始学习率（降低以提高稳定性）
    'lrf': 0.005,                # 最终学习率
    'momentum': 0.937,           # 动量
    'weight_decay': 0.0005,      # 权重衰减
    'warmup_epochs': 3,          # 预热轮数
    'warmup_momentum': 0.8,      # 预热动量
    'warmup_bias_lr': 0.05,      # 预热偏置学习率
    'box': 7.5,                  # 边界框损失权重
    'cls': 0.5,                  # 类别损失权重
    'dfl': 1.5,                  # 分布焦点损失权重
    'pose': 12.0,                # 姿态关键点损失权重
    'kobj': 1.0,                 # 关键点可见性损失权重

    'nbs': 32,                   # 标称批大小（降低以匹配新的batch size）
    'overlap_mask': True,        # 重叠掩码
    'mask_ratio': 4,             # 掩码比例
    'dropout': 0.0,              # Dropout概率
    'val': True,                 # 训练期间验证
    'save_period': 10,           # 每10轮保存一次检查点
    'exist_ok': True,            # 覆盖现有目录
    'pretrained': True,          # 使用预训练权重
    'optimizer': 'SGD',          # 优化器
    'verbose': True,             # 详细输出
    'deterministic': True,       # 确定性训练
    'seed': 42,                  # 随机种子
    'workers': 2,                # 数据加载线程数（降低以减少资源占用）
    'amp': False,                # 禁用混合精度训练以提高稳定性
    'fraction': 1.0,             # 训练集使用比例
    'profile': False,            # 性能分析
    'cos_lr': False,             # 余弦学习率调度
    'close_mosaic': 10,          # 最后10轮关闭马赛克增强
    'resume': False,             # 从断点恢复训练
    'single_cls': False,         # 单类别训练
    'device': '',                # 训练设备
    'visualize': False,          # 可视化训练
    'plots': True,               # 生成训练图表
}

# 导入自定义数据增强配置
from data_augmentation import high_level_augmentation

# 数据增强配置
data_augmentation = {
    'augment': True,
    'hsv_h': 0.2,                # 色相调整 - 适应灯珠颜色变化
    'hsv_s': 0.5,                # 饱和度调整 - 增强灯珠发光效果
    'hsv_v': 0.5,                # 亮度调整 - 模拟不同光照条件
    'degrees': 15.0,             # 旋转角度 - 适应旋转速度>0.4转/秒的能量机关
    'translate': 0.15,           # 平移比例 - 适应5米距离视角变化
    'scale': 0.7,                # 缩放比例 - 扩大尺度变化范围
    'shear': 0.0,                # 剪切角度
    'perspective': 0.0,          # 透视变换
    'flipud': 0.0,               # 上下翻转概率（保持物理合理性）
    'fliplr': 0.5,               # 左右翻转概率
    'mosaic': 1.0,               # 马赛克增强概率
    'mixup': 0.2,                # 混合增强概率 - 提高泛化能力
    'copy_paste': 0.0,           # 复制粘贴增强概率
    'auto_augment': 'randaugment',  # 自动增强策略
    'erasing': 0.2,              # 随机擦除概率 - 模拟遮挡
    'rect': False,               # 矩形训练
}

# 合并训练参数
train_params.update(data_augmentation)

# 确保TensorBoard日志目录存在
log_dir = os.path.join('runs', 'pose', 'tensorboard')
os.makedirs(log_dir, exist_ok=True)

# 创建TensorBoard回调函数
class TensorBoardCallback:
    """自定义TensorBoard回调函数，增强日志记录"""
    
    def __init__(self, log_dir):
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(log_dir)
        self.step = 0
    
    def __call__(self, trainer):
        """训练过程中调用的回调函数"""
        self.step += 1
        
        # 记录训练损失
        if hasattr(trainer, 'train_loss'):
            self.writer.add_scalar('train/loss', trainer.train_loss, self.step)
        
        # 记录学习率
        if hasattr(trainer, 'optimizer'):
            lr = trainer.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('train/lr', lr, self.step)
        
        # 记录验证指标
        if hasattr(trainer, 'metrics'):
            metrics = trainer.metrics
            for key, value in metrics.items():
                if isinstance(value, (float, int)):
                    self.writer.add_scalar(f'val/{key}', value, self.step)
        
        # 记录验证图像
        if hasattr(trainer, 'best_result') and trainer.best_result is not None:
            result = trainer.best_result
            if hasattr(result, 'plot'):
                import cv2
                import numpy as np
                
                # 获取绘制的图像
                plot_img = result.plot()
                # 转换为RGB格式
                plot_img_rgb = cv2.cvtColor(np.array(plot_img), cv2.COLOR_BGR2RGB)
                # 记录到TensorBoard
                self.writer.add_image('val/predictions', plot_img_rgb, self.step)
    
    def close(self):
        """关闭TensorBoard写入器"""
        self.writer.close()

# 创建TensorBoard回调实例
tensorboard_callback = TensorBoardCallback(log_dir)

# 添加TensorBoard相关参数
train_params.update({
    'plots': True,  # 生成训练图表
    'save': True,   # 保存模型检查点
})

def start_tensorboard(log_dir='runs/pose/tensorboard', port=6007):
    """
    启动TensorBoard服务
    
    Args:
        log_dir: 日志目录
        port: TensorBoard端口
    """
    try:
        # 检查是否已安装TensorBoard
        import tensorboard
        print(f"TensorBoard版本: {tensorboard.__version__}")
        
        # 启动TensorBoard进程
        cmd = [
            'tensorboard',
            '--logdir', log_dir,
            '--port', str(port),
            '--host', '0.0.0.0'
        ]
        
        # 在后台启动TensorBoard
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # 等待TensorBoard启动
        time.sleep(3)
        
        # 检查进程状态
        if process.poll() is None:
            tensorboard_url = f"http://localhost:{port}"
            print(f"\n✅ TensorBoard已启动: {tensorboard_url}")
            print(f"📊 日志目录: {log_dir}")
            print("💡 提示: 训练开始后，TensorBoard将显示实时训练指标")
            
            # 尝试自动打开浏览器
            try:
                webbrowser.open(tensorboard_url)
            except:
                print("⚠️  无法自动打开浏览器，请手动访问上述URL")
            
            return process
        else:
            stderr = process.stderr.read()
            print(f"❌ TensorBoard启动失败: {stderr}")
            return None
            
    except ImportError:
        print("❌ TensorBoard未安装，请运行 'pip install tensorboard' 进行安装")
        return None
    except Exception as e:
        print(f"❌ 启动TensorBoard时发生错误: {e}")
        return None

def train_rune_pose():
    """训练YOLO11n-pose模型用于能量机关检测"""
    print("=" * 80)
    print("开始训练YOLO11n-pose能量机关检测模型")
    print(f"数据集路径: {data_config['path']}")
    # 计算训练和验证图像数量
    train_images = os.listdir(os.path.join(data_config['path'], 'images', 'train'))
    val_images = os.listdir(os.path.join(data_config['path'], 'images', 'val'))
    print(f"训练图像数: {len(train_images)}")
    print(f"验证图像数: {len(val_images)}")
    print(f"类别数: {data_config['nc']}, 类别: {data_config['names']}")
    print(f"关键点数量: {data_config['kpt_shape'][0]}")
    print("=" * 80)

    # 启动TensorBoard服务
    tensorboard_process = start_tensorboard(log_dir=log_dir, port=6006)

    # 加载阶段1的最佳模型权重
    model = YOLO('runs/pose/rune_pose_model_stage1/weights/last.pt')
    
    # 显示CUDA设备信息
    if torch.cuda.is_available():
        print(f"\nGPU设备: {torch.cuda.get_device_name(0)}")
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"PyTorch版本: {torch.__version__}")
    else:
        print("\n⚠️  CUDA不可用，将使用CPU进行训练（速度较慢）")
    
    print(f"\n训练参数:")
    print(f"- 图像大小: {train_params['imgsz']}")
    print(f"- 总训练轮数: 30（阶段2）")
    print(f"- 批处理大小: {train_params['batch']}")
    print(f"- 优化器: {train_params['optimizer']}")
    print(f"- 学习率: {train_params['lr0']}")
    print(f"- 混合精度训练: {'启用' if train_params['amp'] else '禁用'}")

    # 阶段2：解冻所有参数训练30轮
    print("\n阶段2：解冻所有参数训练30轮")
    for param in model.parameters():
        param.requires_grad = True
    
    # 注册TensorBoard回调
    model.add_callback('on_train_epoch_end', tensorboard_callback)
    
    # 更新训练参数，不使用resume=True，避免权限冲突
    stage2_params = train_params.copy()
    stage2_params['resume'] = False

    # 训练30轮，使用新的训练目录
    model.train(
        **stage2_params,
        epochs=30,
        name='rune_pose_model_stage2'
    )

    # 关闭TensorBoard写入器
    tensorboard_callback.close()
    
    # 关闭TensorBoard进程
    if tensorboard_process:
        try:
            tensorboard_process.terminate()
            tensorboard_process.wait(timeout=5)
            print("✅ TensorBoard进程已关闭")
        except:
            tensorboard_process.kill()
            print("⚠️  TensorBoard进程已强制关闭")

    print("\n" + "=" * 80)
    print("训练完成！")
    print("模型输出路径: runs/pose/rune_pose_model_stage2")
    print("\n训练结果总结:")
    print("- TensorBoard日志: runs/pose/tensorboard")
    print("- 模型检查点: runs/pose/rune_pose_model_stage2")
    print("- 评估图表: runs/pose/rune_pose_model_stage2/val_batch0_pred.jpg")
    print("=" * 80)

if __name__ == '__main__':
    train_rune_pose()
