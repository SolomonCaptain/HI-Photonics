"""
提示词模板模块
"""

from pathlib import Path

PROMPTS_DIR = Path(__file__).parent

# 设备类型关键词
DEVICE_KEYWORDS = {
    "grating_coupler": ["光栅耦合器", "grating coupler", "耦合器", "grating", "coupler"],
    "metagrating": ["超构光栅", "metagrating", "元光栅", "超表面光栅"],
    "wavelength_demux": ["波分复用", "wavelength demux", "解复用", "demultiplexer", "wdm"]
}

# 模型特点描述
MODEL_DESCRIPTIONS = {
    "tnn": "串联神经网络 (TNN): 快速原型设计，一对一映射，适合简单设计空间",
    "mdn": "混合密度网络 (MDN): 处理一对多映射，提供不确定性估计",
    "cgan": "条件生成对抗网络 (CGAN): 生成多样化设计，探索设计空间",
    "pinn": "物理信息神经网络 (PINN): 物理约束保证，数据需求少",
    "hilab": "HiLab (VAE + 贝叶斯优化): 高质量设计，支持多目标优化"
}

# 默认模型配置
DEFAULT_MODEL_CONFIGS = {
    "tnn": {
        "type": "tnn",
        "hidden_dims": [256, 512, 256],
        "activation": "relu"
    },
    "mdn": {
        "type": "mdn",
        "hidden_dims": [256, 512, 256],
        "n_components": 5
    },
    "cgan": {
        "type": "cgan",
        "generator_dims": [256, 512, 256],
        "discriminator_dims": [256, 128, 1],
        "latent_dim": 64
    },
    "pinn": {
        "type": "pinn",
        "hidden_dims": [256, 512, 512, 256],
        "physics_weight": 0.1
    },
    "hilab": {
        "type": "hilab",
        "vae_hidden_dims": [256, 512],
        "latent_dim": 64,
        "bayesian_iterations": 50
    }
}
