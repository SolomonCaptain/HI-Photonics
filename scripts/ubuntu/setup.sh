#!/bin/bash
#
# HI-Photonics Ubuntu 22.04 一键配置脚本
# 支持: Anaconda + Python 3.13 + CUDA
#
# 用法: bash setup.sh [--cuda-version=12.0]
#

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# 配置变量
CUDA_VERSION="${CUDA_VERSION:-12.0}"
CONDA_ENV_NAME="hi_photonics"
PYTHON_VERSION="3.13"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "============================================================"
echo "  HI-Photonics Ubuntu 22.04 Setup Script"
echo "============================================================"
echo ""
echo "Configuration:"
echo "  - Python: ${PYTHON_VERSION}"
echo "  - CUDA: ${CUDA_VERSION}"
echo "  - Conda Env: ${CONDA_ENV_NAME}"
echo "  - Project Root: ${PROJECT_ROOT}"
echo ""

# 检查是否为 root 用户
if [ "$EUID" -eq 0 ]; then
    log_warn "Running as root is not recommended for Conda installation"
fi

# ========================================
# Step 1: 系统依赖安装
# ========================================
log_info "Step 1: Installing system dependencies..."

sudo apt-get update
sudo apt-get install -y \
    build-essential \
    git \
    curl \
    wget \
    ca-certificates \
    gnupg \
    lsb-release \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libhdf5-dev \
    libopenblas-dev \
    nodejs \
    npm

log_success "System dependencies installed"

# ========================================
# Step 2: 安装 Anaconda/Miniconda
# ========================================
log_info "Step 2: Checking Anaconda/Miniconda installation..."

if command -v conda &> /dev/null; then
    log_success "Conda already installed: $(conda --version)"
else
    log_info "Installing Miniconda..."
    
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    MINICONDA_SH="/tmp/miniconda.sh"
    
    wget -q $MINICONDA_URL -O $MINICONDA_SH
    bash $MINICONDA_SH -b -p $HOME/miniconda3
    rm $MINICONDA_SH
    
    # 初始化 conda
    eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
    conda init bash
    
    log_success "Miniconda installed at $HOME/miniconda3"
    log_warn "Please restart your terminal or run: source ~/.bashrc"
fi

# 确保 conda 可用
if ! command -v conda &> /dev/null; then
    eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
fi

# ========================================
# Step 3: 创建 Conda 环境
# ========================================
log_info "Step 3: Creating Conda environment '${CONDA_ENV_NAME}'..."

if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
    log_warn "Environment '${CONDA_ENV_NAME}' already exists"
    read -p "Do you want to remove and recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        conda env remove -n ${CONDA_ENV_NAME} -y
    else
        log_info "Using existing environment"
    fi
fi

if ! conda env list | grep -q "^${CONDA_ENV_NAME} "; then
    conda create -n ${CONDA_ENV_NAME} python=${PYTHON_VERSION} -y
    log_success "Environment '${CONDA_ENV_NAME}' created"
fi

# 激活环境
conda activate ${CONDA_ENV_NAME}
log_success "Activated environment: ${CONDA_ENV_NAME}"

# ========================================
# Step 4: 安装 PyTorch (CUDA)
# ========================================
log_info "Step 4: Installing PyTorch with CUDA ${CUDA_VERSION}..."

# 根据 CUDA 版本选择 PyTorch 版本
case ${CUDA_VERSION} in
    "12.0"|"12.1"|"12.4"|"12.8"|"13.0")
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu${CUDA_VERSION//./}
        ;;
    "11.8")
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        ;;
    *)
        log_warn "Unknown CUDA version ${CUDA_VERSION}, installing CPU version"
        pip install torch torchvision torchaudio
        ;;
esac

# 验证 PyTorch CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

log_success "PyTorch installed"

# ========================================
# Step 5: 安装 Python 依赖
# ========================================
log_info "Step 5: Installing Python dependencies..."

cd ${PROJECT_ROOT}

# 安装后端依赖
pip install --upgrade pip
pip install -r api/requirements.txt

# 安装其他科学计算依赖
pip install \
    numpy \
    scipy \
    matplotlib \
    h5py \
    scikit-learn \
    pandas \
    tqdm \
    tensorboard

# 可选: Meep FDTD (需要较长时间编译)
read -p "Install Meep FDTD simulator? This may take 30+ minutes. (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    log_info "Installing Meep..."
    conda install -c conda-forge pymeep -y
    log_success "Meep installed"
fi

log_success "Python dependencies installed"

# ========================================
# Step 6: 安装前端依赖
# ========================================
log_info "Step 6: Installing frontend dependencies..."

cd ${PROJECT_ROOT}/frontend

# 检查 Node.js 版本
NODE_VERSION=$(node -v 2>/dev/null || echo "v0")
log_info "Node.js version: ${NODE_VERSION}"

if [ ! -d "node_modules" ]; then
    npm install
    log_success "Frontend dependencies installed"
else
    log_warn "node_modules already exists, skipping npm install"
fi

# ========================================
# Step 7: 创建启动脚本
# ========================================
log_info "Step 7: Creating launch scripts..."

# 创建启动脚本
cat > ${PROJECT_ROOT}/start.sh << 'SCRIPT'
#!/bin/bash
#
# HI-Photonics 一键启动脚本
#

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV_NAME="hi_photonics"

# 颜色
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }

# 初始化 conda
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate ${CONDA_ENV_NAME}

# 检查 GPU
log_info "Checking GPU..."
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

# 启动 API 服务器
log_info "Starting API Server..."
cd ${PROJECT_ROOT}/api
python -m uvicorn main:app --host 0.0.0.0 --port 8000 &
API_PID=$!
cd ${PROJECT_ROOT}

# 等待 API 启动
sleep 3

# 检查 API
if curl -s http://localhost:8000/health > /dev/null; then
    log_success "API Server running at http://localhost:8000"
else
    log_info "API Server starting..."
fi

# 启动前端
log_info "Starting Frontend..."
cd ${PROJECT_ROOT}/frontend
npm start &
UI_PID=$!
cd ${PROJECT_ROOT}

echo ""
echo "============================================================"
echo "  HI-Photonics Studio Started"
echo "============================================================"
echo ""
echo "  API Server:  http://localhost:8000"
echo "  API Docs:    http://localhost:8000/docs"
echo "  Frontend:    http://localhost:3000"
echo ""
echo "  Press Ctrl+C to stop all services"
echo "============================================================"

# 保存 PID
echo ${API_PID} > /tmp/hi_photonics_api.pid
echo ${UI_PID} > /tmp/hi_photonics_ui.pid

# 等待中断
trap "echo 'Stopping...'; kill ${API_PID} ${UI_PID} 2>/dev/null; rm -f /tmp/hi_photonics_*.pid; exit 0" INT TERM
wait
SCRIPT

chmod +x ${PROJECT_ROOT}/start.sh

# 创建停止脚本
cat > ${PROJECT_ROOT}/stop.sh << 'SCRIPT'
#!/bin/bash
#
# HI-Photonics 停止脚本
#

echo "Stopping HI-Photonics services..."

# 读取 PID 并停止
if [ -f /tmp/hi_photonics_api.pid ]; then
    kill $(cat /tmp/hi_photonics_api.pid) 2>/dev/null || true
    rm -f /tmp/hi_photonics_api.pid
fi

if [ -f /tmp/hi_photonics_ui.pid ]; then
    kill $(cat /tmp/hi_photonics_ui.pid) 2>/dev/null || true
    rm -f /tmp/hi_photonics_ui.pid
fi

# 额外清理
pkill -f "uvicorn main:app" 2>/dev/null || true
pkill -f "react-scripts start" 2>/dev/null || true

echo "All services stopped"
SCRIPT

chmod +x ${PROJECT_ROOT}/stop.sh

log_success "Launch scripts created"

# ========================================
# Step 8: 验证安装
# ========================================
log_info "Step 8: Verifying installation..."

cd ${PROJECT_ROOT}

# 测试 Python 模块
python -c "
import sys
print(f'Python: {sys.version}')

import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')

if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')

import numpy as np
print(f'NumPy: {np.__version__}')

import scipy
print(f'SciPy: {scipy.__version__}')

print('\nAll core modules loaded successfully!')
"

log_success "Installation verified"

# ========================================
# 完成
# ========================================
echo ""
echo "============================================================"
echo "  ${GREEN}Setup Complete!${NC}"
echo "============================================================"
echo ""
echo "To activate the environment:"
echo "  conda activate ${CONDA_ENV_NAME}"
echo ""
echo "To start HI-Photonics Studio:"
echo "  cd ${PROJECT_ROOT}"
echo "  ./start.sh"
echo ""
echo "To stop all services:"
echo "  ./stop.sh"
echo ""
echo "============================================================"
