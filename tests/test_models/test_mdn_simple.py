"""
MDN 验证脚本
"""
import sys
sys.path.insert(0, '.')

import torch
from models.inverse.mdn import MDN, MDNConfig, GaussianMixtureDistribution

def test_basic_functionality():
    """测试基本功能"""
    print("=" * 60)
    print("测试 MDN 基本功能")
    print("=" * 60)
    
    # 创建配置
    config = MDNConfig(
        input_dim=3,
        output_dim=100,
        design_shape=(10, 10),
        n_components=5,
        hidden_dims=[64, 128, 64]
    )
    
    # 创建模型
    mdn = MDN(config)
    print(f"\nMDN 创建成功")
    print(f"  - 输入维度: {config.input_dim}")
    print(f"  - 输出维度: {config.output_dim}")
    print(f"  - 高斯分量数: {config.n_components}")
    print(f"  - 参数数量: {mdn.count_parameters():,}")
    
    # 测试前向传播
    batch_size = 4
    condition = torch.randn(batch_size, config.input_dim)
    
    pi, mu, sigma = mdn(condition)
    print(f"\n前向传播测试:")
    print(f"  - 输入形状: {condition.shape}")
    print(f"  - pi 形状: {pi.shape}")
    print(f"  - mu 形状: {mu.shape}")
    print(f"  - sigma 形状: {sigma.shape}")
    
    # 验证 pi 归一化
    pi_sum = pi.sum(dim=-1)
    print(f"  - pi 和: {pi_sum.tolist()}")
    assert torch.allclose(pi_sum, torch.ones(batch_size), atol=1e-5), "pi 未正确归一化"
    
    # 测试采样
    samples = mdn.sample(condition, n_samples=5)
    print(f"\n采样测试:")
    print(f"  - 样本形状: {samples.shape}")
    assert samples.shape == (batch_size, 5, 10, 10), "采样形状错误"
    
    # 测试众数
    mode = mdn.sample_mode(condition)
    print(f"  - 众数形状: {mode.shape}")
    assert mode.shape == (batch_size, 10, 10), "众数形状错误"
    
    # 测试损失计算
    target = torch.randn(batch_size, 10, 10)
    loss = mdn.compute_loss(condition, target)
    print(f"\n损失计算:")
    print(f"  - 损失值: {loss.item():.4f}")
    assert not torch.isnan(loss), "损失为 NaN"
    
    print("\n所有基本功能测试通过!")


def test_distribution():
    """测试高斯混合分布"""
    print("\n" + "=" * 60)
    print("测试高斯混合分布")
    print("=" * 60)
    
    batch_size = 4
    n_components = 3
    output_dim = 50
    
    # 创建分布参数
    pi = torch.softmax(torch.randn(batch_size, n_components), dim=-1)
    mu = torch.randn(batch_size, n_components, output_dim)
    sigma = torch.exp(torch.randn(batch_size, n_components, output_dim)) * 0.5 + 0.1
    
    # 创建分布
    distribution = GaussianMixtureDistribution(pi, mu, sigma)
    
    # 测试对数概率
    x = torch.randn(batch_size, output_dim)
    log_prob = distribution.log_prob(x)
    print(f"\n对数概率:")
    print(f"  - 形状: {log_prob.shape}")
    print(f"  - 值范围: [{log_prob.min().item():.2f}, {log_prob.max().item():.2f}]")
    
    # 测试采样
    samples = distribution.sample(n_samples=10)
    print(f"\n采样:")
    print(f"  - 形状: {samples.shape}")
    
    # 测试熵
    entropy = distribution.get_entropy()
    print(f"\n熵:")
    print(f"  - 形状: {entropy.shape}")
    print(f"  - 值: {entropy.tolist()}")
    
    print("\n分布测试通过!")


def test_training_step():
    """测试训练步骤"""
    print("\n" + "=" * 60)
    print("测试训练步骤")
    print("=" * 60)
    
    config = MDNConfig(
        input_dim=3,
        output_dim=50,
        design_shape=(5, 10),
        n_components=3,
        hidden_dims=[32, 64]
    )
    
    mdn = MDN(config)
    optimizer = torch.optim.Adam(mdn.parameters(), lr=1e-3)
    
    # 创建模拟数据
    performances = torch.randn(32, 3)
    designs = torch.randn(32, 5, 10)
    
    # 训练几步
    losses = []
    for step in range(5):
        optimizer.zero_grad()
        loss = mdn.compute_loss(performances, designs)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        print(f"  步骤 {step+1}: 损失 = {loss.item():.4f}")
    
    # 检查损失是否有效
    assert all(not (l != l) for l in losses), "存在 NaN 损失"
    
    print("\n训练步骤测试通过!")


def test_numerical_stability():
    """测试数值稳定性"""
    print("\n" + "=" * 60)
    print("测试数值稳定性")
    print("=" * 60)
    
    config = MDNConfig(
        input_dim=3,
        output_dim=50,
        n_components=3,
        log_sigma_min=-5.0,
        log_sigma_max=1.0
    )
    
    mdn = MDN(config)
    
    # 极端输入
    extreme_input = torch.randn(4, 3) * 100
    pi, mu, sigma = mdn(extreme_input)
    
    print(f"\n极端输入测试:")
    print(f"  - sigma 范围: [{sigma.min().item():.6f}, {sigma.max().item():.6f}]")
    print(f"  - pi 范围: [{pi.min().item():.6f}, {pi.max().item():.6f}]")
    
    # 检查无 NaN 或 Inf
    assert not torch.isnan(sigma).any(), "sigma 包含 NaN"
    assert not torch.isinf(sigma).any(), "sigma 包含 Inf"
    assert not torch.isnan(mu).any(), "mu 包含 NaN"
    
    print("\n数值稳定性测试通过!")


if __name__ == '__main__':
    test_basic_functionality()
    test_distribution()
    test_training_step()
    test_numerical_stability()
    
    print("\n" + "=" * 60)
    print("所有 MDN 测试通过!")
    print("=" * 60)
