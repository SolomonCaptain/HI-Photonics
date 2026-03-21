"""
Windows 平台优化的逆向设计模型训练示例

演示如何训练 TNN 模型并保存为 safetensors 格式。
"""

import sys
import platform
import gc
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

# 添加项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Windows 平台设置
if platform.system() == 'Windows':
    torch.multiprocessing.set_sharing_strategy('file_system')

from data.loaders.pipeline import SyntheticDataset, create_dataloaders
from models.inverse.tnn import TandemNetwork, TandemNetworkConfig
from models.safetensor_utils import convert_torch_to_safetensors, get_safetensors_info


def setup_training_environment() -> Dict[str, Any]:
    """
    设置训练环境
    
    Returns:
        环境配置信息
    """
    env_info = {
        'platform': platform.system(),
        'python_version': platform.python_version(),
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    if torch.cuda.is_available():
        env_info['cuda_version'] = torch.version.cuda
        env_info['gpu_name'] = torch.cuda.get_device_name(0)
        env_info['gpu_memory'] = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    # Windows 特定设置
    if platform.system() == 'Windows':
        env_info['num_workers'] = 0
        print("[Windows] 数据加载器使用 num_workers=0")
    else:
        import os
        env_info['num_workers'] = min(4, os.cpu_count() // 2 or 1)
    
    print("\n" + "=" * 50)
    print("训练环境配置")
    print("=" * 50)
    for key, value in env_info.items():
        print(f"  {key}: {value}")
    print("=" * 50 + "\n")
    
    return env_info


def train_with_mixed_precision(
    model: nn.Module,
    train_loader,
    val_loader,
    epochs: int = 100,
    lr: float = 1e-3,
    patience: int = 15,
    device: str = 'cuda',
    use_amp: bool = True
) -> Dict[str, Any]:
    """
    混合精度训练（减少内存，加速训练）
    
    Args:
        model: 模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        epochs: 训练轮数
        lr: 学习率
        patience: 早停耐心值
        device: 计算设备
        use_amp: 是否使用混合精度
        
    Returns:
        训练历史
    """
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
    
    scaler = GradScaler() if use_amp and device == 'cuda' else None
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rate': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            design = batch['design'].to(device)
            performance = batch['performance'].to(device)
            
            optimizer.zero_grad()
            
            if scaler is not None:
                # 混合精度训练
                with autocast():
                    pred = model(design)
                    loss = F.mse_loss(pred, performance)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                # 常规训练
                pred = model(design)
                loss = F.mse_loss(pred, performance)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            train_loss += loss.item()
            
            # Windows 内存优化：定期清理
            if platform.system() == 'Windows' and batch_idx % 50 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        train_loss /= len(train_loader)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                design = batch['design'].to(device)
                performance = batch['performance'].to(device)
                
                if scaler is not None:
                    with autocast():
                        pred = model(design)
                        loss = F.mse_loss(pred, performance)
                else:
                    pred = model(design)
                    loss = F.mse_loss(pred, performance)
                
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        # 更新学习率
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['learning_rate'].append(current_lr)
        
        # 打印进度
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.6f} | "
              f"Val Loss: {val_loss:.6f} | "
              f"LR: {current_lr:.2e}")
        
        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            print(f"  -> 保存最佳模型 (val_loss: {val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n早停触发！最佳验证损失: {best_val_loss:.6f}")
                break
        
        # 定期内存清理
        if epoch % 10 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # 恢复最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return history


def train_tnn_model(
    output_dir: str = "op_models/pretrained",
    num_samples: int = 3000,
    design_shape: Tuple[int, int] = (100, 22),
    performance_dim: int = 3,
    batch_size: int = 32,
    forward_epochs: int = 25,
    inverse_epochs: int = 50,
    save_safetensors: bool = True
) -> TandemNetwork:
    """
    训练 TNN 模型并保存
    
    Args:
        output_dir: 输出目录
        num_samples: 训练样本数
        design_shape: 设计形状
        performance_dim: 性能维度
        batch_size: 批次大小
        forward_epochs: 前向网络训练轮数
        inverse_epochs: 逆向网络训练轮数
        save_safetensors: 是否保存为 safetensors 格式
        
    Returns:
        训练完成的模型
    """
    # 设置环境
    env_info = setup_training_environment()
    
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建数据集
    print(f"\n创建数据集 ({num_samples} 样本)...")
    dataset = SyntheticDataset(
        num_samples=num_samples,
        design_shape=design_shape,
        performance_dim=performance_dim,
        noise_level=0.05,
        seed=42
    )
    
    # 创建数据加载器（自动优化 Windows）
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset,
        batch_size=batch_size,
        train_ratio=0.8,
        val_ratio=0.1
    )
    
    print(f"训练集: {len(train_loader.dataset)} 样本")
    print(f"验证集: {len(val_loader.dataset)} 样本")
    print(f"测试集: {len(test_loader.dataset)} 样本")
    
    # 创建模型配置
    config = TandemNetworkConfig(
        name="tnn_windows_trained",
        pretrain_forward=True,
        freeze_forward=True
    )
    
    # 创建模型
    print("\n初始化 TNN 模型...")
    model = TandemNetwork(config)
    print(f"模型参数量: {model.count_parameters():,}")
    
    # 训练前向网络
    print("\n" + "=" * 50)
    print("阶段 1: 训练前向网络")
    print("=" * 50)
    forward_history = train_with_mixed_precision(
        model.forward_net,
        train_loader,
        val_loader,
        epochs=forward_epochs,
        lr=1e-3,
        patience=10,
        device=env_info['device'],
        use_amp=env_info['cuda_available']
    )
    model.forward_trained = True
    model.training_history['forward'] = forward_history
    
    # 冻结前向网络
    model.forward_net.freeze()
    print("\n前向网络已冻结")
    
    # 训练逆向网络
    print("\n" + "=" * 50)
    print("阶段 2: 训练逆向网络")
    print("=" * 50)
    inverse_history = train_with_mixed_precision(
        model.inverse_net,
        train_loader,
        val_loader,
        epochs=inverse_epochs,
        lr=1e-4,
        patience=15,
        device=env_info['device'],
        use_amp=env_info['cuda_available']
    )
    model.inverse_trained = True
    model.training_history['inverse'] = inverse_history
    
    # 测试逆向设计
    print("\n" + "=" * 50)
    print("测试逆向设计")
    print("=" * 50)
    model.eval()
    with torch.no_grad():
        # 随机选择一个目标性能
        test_batch = next(iter(test_loader))
        target_performance = test_batch['performance'][:1].to(env_info['device'])
        
        # 执行逆向设计
        generated_design = model.inverse_net(target_performance)
        predicted_performance = model.forward_net(generated_design)
        
        print(f"目标性能: {target_performance[0].cpu().numpy()}")
        print(f"预测性能: {predicted_performance[0].cpu().numpy()}")
        print(f"设计形状: {generated_design[0].shape}")
    
    # 保存模型
    print("\n" + "=" * 50)
    print("保存模型")
    print("=" * 50)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存为 PyTorch 格式
    torch_path = output_dir / f"tnn_model_{timestamp}.pth"
    model.save(torch_path, format="torch")
    print(f"PyTorch 模型保存至: {torch_path}")
    
    # 保存为 safetensors 格式
    if save_safetensors:
        safetensors_path = output_dir / f"tnn_model_{timestamp}.safetensors"
        
        # 分别保存前向和逆向网络
        safetensors_forward = output_dir / f"tnn_forward_{timestamp}.safetensors"
        safetensors_inverse = output_dir / f"tnn_inverse_{timestamp}.safetensors"
        
        # 保存前向网络
        import safetensors.torch
        safetensors.torch.save_file(
            model.forward_net.state_dict(),
            str(safetensors_forward)
        )
        
        # 保存逆向网络
        safetensors.torch.save_file(
            model.inverse_net.state_dict(),
            str(safetensors_inverse)
        )
        
        # 保存元数据
        import json
        metadata = {
            'model_name': config.name,
            'training_history': model.training_history,
            'design_shape': list(design_shape),
            'performance_dim': performance_dim,
            'saved_at': datetime.now().isoformat(),
            'platform': platform.system(),
        }
        
        for net_name, net_path in [('forward', safetensors_forward), ('inverse', safetensors_inverse)]:
            metadata_path = net_path.with_suffix('.json')
            net_metadata = {**metadata, 'network': net_name}
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(net_metadata, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"Safetensors 模型保存至:")
        print(f"  - 前向网络: {safetensors_forward}")
        print(f"  - 逆向网络: {safetensors_inverse}")
    
    print("\n训练完成!")
    return model


def load_safetensors_tnn(
    forward_path: str,
    inverse_path: str,
    config: Optional[TandemNetworkConfig] = None
) -> TandemNetwork:
    """
    从 safetensors 加载 TNN 模型
    
    Args:
        forward_path: 前向网络路径
        inverse_path: 逆向网络路径
        config: 模型配置
        
    Returns:
        加载的模型
    """
    import safetensors.torch
    import json
    
    if config is None:
        # 尝试从元数据加载配置
        metadata_path = Path(forward_path).with_suffix('.json')
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            config = TandemNetworkConfig(name=metadata.get('model_name', 'loaded_tnn'))
        else:
            config = TandemNetworkConfig(name="loaded_tnn")
    
    # 创建模型
    model = TandemNetwork(config)
    
    # 加载权重
    forward_state = safetensors.torch.load_file(forward_path)
    inverse_state = safetensors.torch.load_file(inverse_path)
    
    model.forward_net.load_state_dict(forward_state)
    model.inverse_net.load_state_dict(inverse_state)
    
    model.forward_trained = True
    model.inverse_trained = True
    
    print(f"模型加载完成:")
    print(f"  - 前向网络: {forward_path}")
    print(f"  - 逆向网络: {inverse_path}")
    
    return model


if __name__ == "__main__":
    # 训练模型
    model = train_tnn_model(
        output_dir="op_models/pretrained",
        num_samples=3000,
        design_shape=(100, 22),
        performance_dim=3,
        batch_size=32,
        forward_epochs=25,
        inverse_epochs=50,
        save_safetensors=True
    )
    
    # 验证加载
    print("\n" + "=" * 50)
    print("验证 safetensors 模型加载")
    print("=" * 50)
    
    # 找到最新保存的模型
    import glob
    safetensors_files = list(Path("op_models/pretrained").glob("tnn_forward_*.safetensors"))
    if safetensors_files:
        latest_forward = max(safetensors_files, key=lambda p: p.stat().st_mtime)
        latest_inverse = latest_forward.parent / latest_forward.name.replace("forward", "inverse")
        
        if latest_inverse.exists():
            loaded_model = load_safetensors_tnn(str(latest_forward), str(latest_inverse))
            print(f"加载成功！参数量: {loaded_model.count_parameters():,}")
