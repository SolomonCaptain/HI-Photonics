#!/bin/bash
#
# HI-Photonics 启动脚本
#
# 自动检测平台并启动相应服务
#

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV_NAME="hi_photonics"

# 颜色
GREEN='\033[0;32m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# 检测操作系统
detect_os() {
    case "$(uname -s)" in
        Linux*)  echo "linux" ;;
        Darwin*) echo "macos" ;;
        CYGWIN*) echo "windows" ;;
        MINGW*)  echo "windows" ;;
        *)       echo "unknown" ;;
    esac
}

OS=$(detect_os)

echo ""
echo -e "${CYAN}============================================================${NC}"
echo -e "${CYAN}  HI-Photonics Studio Launcher${NC}"
echo -e "${CYAN}============================================================${NC}"
echo ""
echo "  OS: ${OS}"
echo "  Project: ${PROJECT_ROOT}"
echo ""

# 初始化 conda
init_conda() {
    local conda_path=""
    
    if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
        conda_path=~/miniconda3/etc/profile.d/conda.sh
    elif [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
        conda_path=~/anaconda3/etc/profile.d/conda.sh
    elif [ -f /opt/conda/etc/profile.d/conda.sh ]; then
        conda_path=/opt/conda/etc/profile.d/conda.sh
    fi
    
    if [ -z "$conda_path" ]; then
        echo -e "${RED}[ERROR] Conda not found. Please install Miniconda or Anaconda.${NC}"
        echo ""
        echo "  Download: https://docs.conda.io/en/latest/miniconda.html"
        exit 1
    fi
    
    source "$conda_path"
    
    if ! conda activate ${CONDA_ENV_NAME} 2>/dev/null; then
        echo -e "${YELLOW}[WARN] Environment '${CONDA_ENV_NAME}' not found.${NC}"
        echo ""
        echo "  Creating environment..."
        conda create -n ${CONDA_ENV_NAME} python=3.13 -y
        conda activate ${CONDA_ENV_NAME}
        
        echo ""
        echo "  Installing dependencies..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu120
        pip install -r ${PROJECT_ROOT}/api/requirements.txt
    fi
}

# 创建日志目录
mkdir -p ${PROJECT_ROOT}/logs

# 初始化环境
init_conda

# 检查 GPU
echo ""
echo -e "${BLUE}[INFO]${NC} Checking GPU..."
python ${PROJECT_ROOT}/scripts/ubuntu/check_gpu.py 2>/dev/null || true

# 启动服务
echo ""
echo -e "${BLUE}[INFO]${NC} Starting services..."

# 启动 API
echo -e "${BLUE}[INFO]${NC} Starting API Server..."
cd ${PROJECT_ROOT}/api

# 杀死旧进程
pkill -f "uvicorn main:app" 2>/dev/null || true
sleep 1

nohup python -m uvicorn main:app \
    --host 0.0.0.0 \
    --port 8000 \
    > ${PROJECT_ROOT}/logs/api.log 2>&1 &

API_PID=$!
echo ${API_PID} > /tmp/hi_photonics_api.pid

# 等待 API
sleep 3
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo -e "${GREEN}[OK]${NC} API Server: http://localhost:8000"
else
    echo -e "${YELLOW}[WAIT]${NC} API Server starting..."
fi

# 启动前端
echo -e "${BLUE}[INFO]${NC} Starting Frontend..."
cd ${PROJECT_ROOT}/frontend

if [ ! -d "node_modules" ]; then
    echo -e "${BLUE}[INFO]${NC} Installing frontend dependencies..."
    npm install --silent
fi

# 杀死旧进程
pkill -f "react-scripts start" 2>/dev/null || true
sleep 1

nohup npm start > ${PROJECT_ROOT}/logs/ui.log 2>&1 &
UI_PID=$!
echo ${UI_PID} > /tmp/hi_photonics_ui.pid

cd ${PROJECT_ROOT}

# 完成
echo ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}  HI-Photonics Studio Started${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""
echo -e "  ${CYAN}API Server:${NC}  http://localhost:8000"
echo -e "  ${CYAN}API Docs:${NC}    http://localhost:8000/docs"
echo -e "  ${CYAN}Frontend:${NC}    http://localhost:3000"
echo ""
echo "  Logs:"
echo "    - API: ${PROJECT_ROOT}/logs/api.log"
echo "    - UI:  ${PROJECT_ROOT}/logs/ui.log"
echo ""
echo -e "  ${BLUE}Press Ctrl+C to stop${NC}"
echo ""

# 等待中断
trap "echo 'Stopping...'; pkill -f 'uvicorn main:app' 2>/dev/null; pkill -f 'react-scripts' 2>/dev/null; rm -f /tmp/hi_photonics_*.pid; exit 0" INT TERM

wait