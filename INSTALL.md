# HI-Photonics 安装指南

## 系统要求

- Ubuntu 22.04 LTS
- NVIDIA GPU (推荐 RTX 3060 或更高)
- CUDA 12.0+
- 16GB+ RAM
- 50GB+ 磁盘空间

## 快速安装

```bash
# 1. 克隆项目
git clone https://github.com/SolomonCaptain/HI-Photonics.git
cd HI-Photonics

# 2. 运行一键配置脚本
bash scripts/ubuntu/setup.sh

# 3. 重启终端 (首次安装需要)
source ~/.bashrc

# 4. 启动服务
./start.sh
```

## 详细步骤

### 1. 安装 NVIDIA 驱动

```bash
# 添加 NVIDIA PPA
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update

# 安装驱动 (推荐 535 版本)
sudo apt install nvidia-driver-535

# 重启
sudo reboot

# 验证
nvidia-smi
```

### 2. 安装 CUDA Toolkit (可选，PyTorch 自带 CUDA)

```bash
# 如果需要系统级 CUDA
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install cuda-toolkit-12-0
```

### 3. 安装 Anaconda/Miniconda

```bash
# 下载 Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# 安装
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3

# 初始化
~/miniconda3/bin/conda init bash
source ~/.bashrc
```

### 4. 创建 Conda 环境

```bash
# 方式 1: 使用环境文件
conda env create -f environment.yml

# 方式 2: 手动创建
conda create -n hi_photonics python=3.13 -y
conda activate hi_photonics
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu120
pip install -r api/requirements.txt
```

### 5. 安装前端依赖

```bash
# 安装 Node.js (如果未安装)
sudo apt install nodejs npm

# 或使用 nvm 安装最新版本
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
source ~/.bashrc
nvm install --lts

# 安装前端依赖
cd frontend
npm install
cd ..
```

### 6. (可选) 安装 Meep FDTD

```bash
# Meep 用于 FDTD 仿真，安装时间约 30 分钟
conda activate hi_photonics
conda install -c conda-forge pymeep -y
```

## 启动服务

```bash
# 激活环境
conda activate hi_photonics

# 启动所有服务
./start.sh

# 仅启动 API
./scripts/ubuntu/quick_start.sh --api

# 仅启动前端
./scripts/ubuntu/quick_start.sh --ui
```

## 访问地址

- **前端界面**: http://localhost:3000
- **API 服务**: http://localhost:8000
- **API 文档**: http://localhost:8000/docs
- **健康检查**: http://localhost:8000/health

## 验证 GPU

```bash
conda activate hi_photonics
python scripts/ubuntu/check_gpu.py
```

## 常见问题

### CUDA not available

```bash
# 检查驱动
nvidia-smi

# 检查 CUDA
python -c "import torch; print(torch.cuda.is_available())"

# 重新安装 PyTorch
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu120
```

### 端口被占用

```bash
# 查看端口占用
lsof -i :8000
lsof -i :3000

# 杀死进程
kill -9 <PID>
```

### 前端启动失败

```bash
# 清理缓存
cd frontend
rm -rf node_modules package-lock.json
npm install
```

## 目录结构

```
HI-Photonics/
├── scripts/
│   └── ubuntu/
│       ├── setup.sh          # 一键配置脚本
│       ├── quick_start.sh    # 快速启动
│       └── check_gpu.py      # GPU 检查
├── frontend/                 # React 前端
├── api/                      # FastAPI 后端
├── models/                   # 深度学习模型
├── data/                     # 数据处理
├── start.sh                  # 启动脚本
├── stop.sh                   # 停止脚本
└── environment.yml           # Conda 环境配置
```
