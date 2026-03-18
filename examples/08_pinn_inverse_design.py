"""
物理信息神经网络 (PINN) 逆向设计示例

本示例展示如何使用 PINN 进行光子器件的:
1. 正向问题求解：预测电磁场分布
2. 逆向设计：从目标场分布反推设计参数
3. 物理约束训练：无条件/少数据训练

PINN 优势:
- 数据高效：物理约束替代部分仿真数据
- 泛化能力强：自动满足物理定律
- 可解释性：物理残差可诊断问题
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from models.inverse.pinn import (
    PhysicsInformedNet,
    SirenNet,
    MaxwellPINN,
    PhotonicsPINN,
    PINNSolver,
    PINNConfig,
    MaxwellConfig,
    PhysicsLossConfig,
    create_pinn_for_photonics
)
from models.training.losses import (
    PDEResidualLoss,
    HelmholtzLoss,
    MaxwellLoss,
    BoundaryConditionLoss,
    PINNCombinedLoss
)
from core.nodes.neural_network import (
    PINNNode,
    CoordinateNode,
    create_pinn_node_for_photonics
)


def set_seed(seed: int = 42):
    """设置随机种子"""
    torch.manual_seed(seed)
    np.random.seed(seed)


def example_1_basic_pinn():
    """
    示例 1: 基础 PINN 使用
    
    求解 Helmholtz 方程: (∇² + k²)u = 0
    """
    print("=" * 60)
    print("示例 1: 基础 PINN - Helmholtz 方程求解")
    print("=" * 60)
    
    # 创建 PINN
    config = PINNConfig(
        spatial_dim=2,
        field_components=1,
        hidden_dims=[64, 128, 256, 128, 64],
        activation='tanh',
        use_fourier=True,
        fourier_dim=64
    )
    
    pinn = PhysicsInformedNet(config)
    print(f"PINN 参数量: {sum(p.numel() for p in pinn.parameters()):,}")
    
    # 创建求解器
    solver = PINNSolver(pinn, optimizer='adam', lr=1e-3)
    
    # 定义计算域
    bounds = [(-1, 1), (-1, 1)]
    
    # 采样配点
    n_collocation = 2000
    collocation_points = torch.rand(n_collocation, 2)
    collocation_points[:, 0] = collocation_points[:, 0] * 2 - 1
    collocation_points[:, 1] = collocation_points[:, 1] * 2 - 1
    
    # 采样边界点
    n_boundary = 200
    boundary_points = []
    for x_val in [-1, 1]:
        points = torch.rand(n_boundary // 4, 2)
        points[:, 0] = x_val
        points[:, 1] = points[:, 1] * 2 - 1
        boundary_points.append(points)
    for y_val in [-1, 1]:
        points = torch.rand(n_boundary // 4, 2)
        points[:, 1] = y_val
        points[:, 0] = points[:, 0] * 2 - 1
        boundary_points.append(points)
    boundary_points = torch.cat(boundary_points, dim=0)
    
    # 训练
    print("\n开始训练...")
    history = solver.train(
        n_iterations=2000,
        collocation_points=collocation_points,
        boundary_points=boundary_points,
        log_interval=500
    )
    
    # 可视化结果
    with torch.no_grad():
        # 生成网格
        x = torch.linspace(-1, 1, 100)
        y = torch.linspace(-1, 1, 100)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        coords = torch.stack([X.flatten(), Y.flatten()], dim=-1)
        
        # 预测
        solution = pinn(coords).view(100, 100)
    
    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 解的可视化
    im = axes[0].imshow(solution.numpy(), extent=[-1, 1, -1, 1], origin='lower', cmap='RdBu')
    axes[0].set_title('PINN Solution')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    plt.colorbar(im, ax=axes[0])
    
    # 损失曲线
    axes[1].plot(history['loss'])
    axes[1].set_title('Training Loss')
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Loss')
    axes[1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('outputs/pinn_basic_solution.png', dpi=150)
    plt.close()
    
    print("结果已保存到 outputs/pinn_basic_solution.png")
    print(f"最终损失: {history['loss'][-1]:.6f}")


def example_2_siren_network():
    """
    示例 2: SIREN 网络
    
    使用正弦激活函数的网络，适合表示高频信号。
    """
    print("\n" + "=" * 60)
    print("示例 2: SIREN 网络 - 高频场表示")
    print("=" * 60)
    
    # 创建 SIREN
    config = PINNConfig(
        spatial_dim=2,
        field_components=1,
        hidden_dims=[64, 128, 128, 64],
        activation='sine',
        use_fourier=False
    )
    
    siren = SirenNet(config, w0=30.0)
    print(f"SIREN 参数量: {sum(p.numel() for p in siren.parameters()):,}")
    
    # 比较不同激活函数
    configs = {
        'SIREN (sine)': PINNConfig(
            spatial_dim=2, field_components=1,
            hidden_dims=[64, 128, 128, 64],
            activation='sine', use_fourier=False
        ),
        'Tanh': PINNConfig(
            spatial_dim=2, field_components=1,
            hidden_dims=[64, 128, 128, 64],
            activation='tanh', use_fourier=False
        ),
        'GELU': PINNConfig(
            spatial_dim=2, field_components=1,
            hidden_dims=[64, 128, 128, 64],
            activation='gelu', use_fourier=False
        )
    }
    
    results = {}
    
    for name, cfg in configs.items():
        if 'SIREN' in name:
            model = SirenNet(cfg, w0=30.0)
        else:
            model = PhysicsInformedNet(cfg)
        
        solver = PINNSolver(model, lr=1e-3)
        
        # 简化训练
        n_points = 1000
        collocation = torch.rand(n_points, 2) * 2 - 1
        
        history = solver.train(
            n_iterations=1000,
            collocation_points=collocation,
            log_interval=500
        )
        
        results[name] = history['loss'][-1]
        print(f"{name}: 最终损失 = {history['loss'][-1]:.6f}")
    
    print("\n不同激活函数的比较:")
    for name, loss in results.items():
        print(f"  {name}: {loss:.6f}")


def example_3_maxwell_pinn():
    """
    示例 3: Maxwell 方程 PINN
    
    求解电磁场分布。
    """
    print("\n" + "=" * 60)
    print("示例 3: Maxwell PINN - 电磁场求解")
    print("=" * 60)
    
    # 创建 Maxwell PINN
    config = MaxwellConfig(
        spatial_dim=2,
        field_components=6,  # Ex, Ey, Ez, Hx, Hy, Hz
        wavelength=1.55e-6,
        epsilon_r=12.0,
        hidden_dims=[64, 128, 256, 256, 128, 64],
        activation='tanh',
        use_fourier=True
    )
    
    pinn = MaxwellPINN(config)
    print(f"Maxwell PINN 参数量: {sum(p.numel() for p in pinn.parameters()):,}")
    print(f"波长: {config.wavelength * 1e6:.2f} μm")
    print(f"相对介电常数: {config.epsilon_r}")
    
    # 创建坐标节点
    coords = CoordinateNode(
        name='coords',
        bounds=[(-2e-6, 2e-6), (-2e-6, 2e-6)],
        resolution=(50, 50)
    )
    
    # 创建 PINN 节点
    pinn_node = PINNNode('maxwell_pinn', pinn, coords)
    
    # 预测场
    print("\n预测电磁场分布...")
    fields = pinn_node.forward()
    print(f"场预测形状: {fields['E'].shape}, {fields['H'].shape}")
    
    # 计算 Maxwell 残差
    print("计算 Maxwell 方程残差...")
    collocation = coords.sample_random(500)
    residual = pinn_node.compute_residual(collocation)
    print(f"E 残差范数: {residual['curl_E_residual'].norm():.6f}")
    print(f"H 残差范数: {residual['curl_H_residual'].norm():.6f}")
    
    # 可视化场分布
    with torch.no_grad():
        grid_coords = coords.get_grid()
        fields_grid = pinn(grid_coords)
        
        E = fields_grid['E'].view(50, 50, 3)
        H = fields_grid['H'].view(50, 50, 3)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 电场分量
    for i, (ax, comp) in enumerate(zip(axes[0], ['Ex', 'Ey', 'Ez'])):
        im = ax.imshow(E[:, :, i].numpy(), cmap='RdBu', origin='lower')
        ax.set_title(comp)
        plt.colorbar(im, ax=ax)
    
    # 磁场分量
    for i, (ax, comp) in enumerate(zip(axes[1], ['Hx', 'Hy', 'Hz'])):
        im = ax.imshow(H[:, :, i].numpy(), cmap='RdBu', origin='lower')
        ax.set_title(comp)
        plt.colorbar(im, ax=ax)
    
    plt.suptitle('Maxwell PINN 场分布预测')
    plt.tight_layout()
    plt.savefig('outputs/pinn_maxwell_fields.png', dpi=150)
    plt.close()
    
    print("场分布已保存到 outputs/pinn_maxwell_fields.png")


def example_4_physics_loss():
    """
    示例 4: 物理约束损失函数
    
    展示各种物理约束损失的使用。
    """
    print("\n" + "=" * 60)
    print("示例 4: 物理约束损失函数")
    print("=" * 60)
    
    # Helmholtz 损失
    helmholtz_loss = HelmholtzLoss(k=1.0, weight=1.0)
    
    u = torch.randn(100, 1)
    laplacian = torch.randn(100, 1)
    
    loss = helmholtz_loss(u, laplacian)
    print(f"Helmholtz 损失: {loss.item():.6f}")
    
    # Maxwell 损失
    maxwell_loss = MaxwellLoss(omega=1.0, epsilon=1.0, mu=1.0)
    
    curl_E = torch.randn(100, 3)
    curl_H = torch.randn(100, 3)
    E = torch.randn(100, 3)
    H = torch.randn(100, 3)
    
    losses = maxwell_loss(curl_E, curl_H, E, H)
    print(f"Maxwell E 损失: {losses['maxwell_E'].item():.6f}")
    print(f"Maxwell H 损失: {losses['maxwell_H'].item():.6f}")
    
    # 边界条件损失
    bc_loss = BoundaryConditionLoss(bc_type='dirichlet', weight=1.0)
    
    pred = torch.randn(100, 1)
    target = torch.zeros(100, 1)
    
    loss = bc_loss(pred, target)
    print(f"Dirichlet 边界条件损失: {loss.item():.6f}")
    
    # PINN 组合损失
    pinn_loss = PINNCombinedLoss(
        physics_weight=1.0,
        bc_weight=1.0,
        data_weight=1.0,
        adaptive_weights=True
    )
    
    physics_residual = torch.randn(100, 1)
    bc_pred = torch.randn(50, 1)
    bc_target = torch.zeros(50, 1)
    data_pred = torch.randn(30, 1)
    data_target = torch.randn(30, 1)
    
    losses = pinn_loss(
        physics_residual=physics_residual,
        bc_pred=bc_pred,
        bc_target=bc_target,
        data_pred=data_pred,
        data_target=data_target
    )
    
    print("\nPINN 组合损失:")
    for key, value in losses.items():
        print(f"  {key}: {value.item():.6f}")


def example_5_graph_integration():
    """
    示例 5: 计算图集成
    
    展示 PINN 如何集成到计算图框架中。
    """
    print("\n" + "=" * 60)
    print("示例 5: 计算图集成")
    print("=" * 60)
    
    # 创建坐标节点
    coords = CoordinateNode(
        name='coordinates',
        bounds=[(-1, 1), (-1, 1)],
        resolution=(50, 50)
    )
    
    # 创建 PINN 节点
    pinn_node = create_pinn_node_for_photonics(
        spatial_dim=2,
        field_components=3,
        wavelength=1.55e-6,
        coordinate_node=coords
    )
    
    print(f"坐标节点: {coords.name}")
    print(f"坐标形状: {coords.forward().shape}")
    
    # 前向预测
    fields = pinn_node.forward()
    print(f"PINN 输出形状: {fields.shape}")
    
    # 计算物理损失
    sample_coords = coords.sample_random(500)
    physics_loss = pinn_node.compute_physics_loss(sample_coords)
    print(f"物理损失: {physics_loss.item():.6f}")
    
    # 获取场梯度
    gradient = pinn_node.get_field_gradient(sample_coords[:10], component=0)
    print(f"场梯度形状: {gradient.shape}")
    
    # 边界采样
    boundary = coords.sample_boundary(25)
    print(f"边界点形状: {boundary.shape}")


def example_6_inverse_design():
    """
    示例 6: 逆向设计
    
    从目标场分布反推设计参数。
    """
    print("\n" + "=" * 60)
    print("示例 6: 逆向设计")
    print("=" * 60)
    
    # 创建 PINN（带设计参数）
    config = PINNConfig(
        spatial_dim=2,
        design_dim=5,  # 5 个设计参数
        field_components=3,
        hidden_dims=[64, 128, 128, 64],
        activation='tanh'
    )
    
    pinn = PhotonicsPINN(config)
    
    # 创建坐标节点
    coords = CoordinateNode(
        name='coords',
        bounds=[(-1, 1), (-1, 1)],
        resolution=(30, 30)
    )
    
    # 创建 PINN 节点
    pinn_node = PINNNode('pinn', pinn, coords)
    
    # 生成目标场（模拟真实场分布）
    target_design = torch.tensor([[0.3, 0.7, 0.5, 0.2, 0.8]])
    target_coords = coords.forward()
    
    with torch.no_grad():
        target_field = pinn(target_coords, target_design)
    
    print(f"目标设计参数: {target_design}")
    print(f"目标场形状: {target_field.shape}")
    
    # 逆向设计
    print("\n开始逆向设计优化...")
    recovered_design = pinn_node.inverse_design(
        target_field=target_field,
        coordinates=target_coords,
        n_iterations=500,
        lr=0.01
    )
    
    print(f"恢复的设计参数: {recovered_design}")
    print(f"设计参数误差: {F.mse_loss(recovered_design, target_design).item():.6f}")
    
    # 验证恢复的设计
    with torch.no_grad():
        recovered_field = pinn(target_coords, recovered_design)
        field_error = F.mse_loss(recovered_field, target_field)
    
    print(f"场重建误差: {field_error.item():.6f}")


def example_7_training_with_physics():
    """
    示例 7: 物理约束训练
    
    展示如何使用物理约束进行训练。
    """
    print("\n" + "=" * 60)
    print("示例 7: 物理约束训练")
    print("=" * 60)
    
    # 创建 PINN
    config = PINNConfig(
        spatial_dim=2,
        field_components=1,
        hidden_dims=[64, 128, 64],
        activation='tanh'
    )
    
    pinn = PhysicsInformedNet(config)
    
    # 创建求解器
    solver = PINNSolver(pinn, optimizer='adam', lr=1e-3)
    
    # 准备数据
    bounds = [(-1, 1), (-1, 1)]
    
    # 配点（物理约束区域）
    n_collocation = 1000
    collocation = torch.rand(n_collocation, 2) * 2 - 1
    
    # 边界点
    n_boundary = 100
    boundary = torch.cat([
        torch.cat([torch.full((25, 1), -1.0), torch.rand(25, 1) * 2 - 1], dim=1),
        torch.cat([torch.full((25, 1), 1.0), torch.rand(25, 1) * 2 - 1], dim=1),
        torch.cat([torch.rand(25, 1) * 2 - 1, torch.full((25, 1), -1.0)], dim=1),
        torch.cat([torch.rand(25, 1) * 2 - 1, torch.full((25, 1), 1.0)], dim=1),
    ])
    
    # 少量标签数据（模拟仿真数据）
    n_labeled = 50
    labeled_coords = torch.rand(n_labeled, 2) * 2 - 1
    # 模拟标签（实际应用中来自仿真）
    labeled_fields = torch.sin(3.14159 * labeled_coords[:, 0:1]) * torch.cos(3.14159 * labeled_coords[:, 1:2])
    labeled_data = {'coords': labeled_coords, 'fields': labeled_fields}
    
    print(f"配点数: {n_collocation}")
    print(f"边界点数: {n_boundary}")
    print(f"标签数据: {n_labeled}")
    
    # 训练
    history = solver.train(
        n_iterations=2000,
        collocation_points=collocation,
        boundary_points=boundary,
        labeled_data=labeled_data,
        log_interval=500
    )
    
    # 比较不同训练策略
    strategies = {
        '仅物理约束': {'physics': True, 'bc': True, 'data': False},
        '物理+边界': {'physics': True, 'bc': True, 'data': False},
        '物理+数据': {'physics': True, 'bc': True, 'data': True}
    }
    
    print("\n不同训练策略比较:")
    for name, use in strategies.items():
        pinn_temp = PhysicsInformedNet(config)
        solver_temp = PINNSolver(pinn_temp, lr=1e-3)
        
        history_temp = solver_temp.train(
            n_iterations=1000,
            collocation_points=collocation if use['physics'] else None,
            boundary_points=boundary if use['bc'] else None,
            labeled_data=labeled_data if use['data'] else None,
            log_interval=1000
        )
        
        print(f"  {name}: 最终损失 = {history_temp['loss'][-1]:.6f}")


def main():
    """主函数"""
    set_seed(42)
    
    # 创建输出目录
    os.makedirs('outputs', exist_ok=True)
    
    print("PINN (物理信息神经网络) 示例")
    print("=" * 60)
    
    # 运行示例
    example_1_basic_pinn()
    example_2_siren_network()
    example_3_maxwell_pinn()
    example_4_physics_loss()
    example_5_graph_integration()
    example_6_inverse_design()
    example_7_training_with_physics()
    
    print("\n" + "=" * 60)
    print("所有示例完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()
