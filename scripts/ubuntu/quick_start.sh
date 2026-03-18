#!/bin/bash
#
# HI-Photonics 快速启动脚本 (Ubuntu 22.04 + GPU)
#
# 用法: 
#   ./quick_start.sh          # 启动所有服务
#   ./quick_start.sh --api    # 仅启动 API
#   ./quick_start.sh --ui     # 仅启动前端
#

set -e

# 配置
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}/..")" && pwd)"
CONDA_ENV_NAME="hi_photonics"
API_PORT=8000
UI_PORT=3000

# 颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# 解析参数
MODE="all"
while [[ $# -gt 0 ]]; do
    case $1 in
        --api) MODE="api" ;;
        --ui) MODE="ui" ;;
        --help) 
            echo "Usage: $0 [--api|--ui|--help]"
            echo "  --api    Start only API server"
            echo "  --ui     Start only frontend"
            exit 0
            ;;
        *) log_error "Unknown option: $1"; exit 1 ;;
    esac
    shift
done

# 初始化 conda
init_conda() {
    if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
        source ~/miniconda3/etc/profile.d/conda.sh
    elif [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
        source ~/anaconda3/etc/profile.d/conda.sh
    else
        log_error "Conda not found. Please run setup.sh first."
        exit 1
    fi
    
    conda activate ${CONDA_ENV_NAME} 2>/dev/null || {
        log_error "Conda environment '${CONDA_ENV_NAME}' not found."
        log_info "Please run: bash scripts/ubuntu/setup.sh"
        exit 1
    }
}

# 检查 GPU
check_gpu() {
    log_info "Checking GPU..."
    
    GPU_INFO=$(python -c "
import torch
if torch.cuda.is_available():
    print(f'{torch.cuda.get_device_name(0)}|{torch.cuda.get_device_properties(0).total_memory // (1024**3)}GB')
else:
    print('CPU')
" 2>/dev/null || echo "ERROR")
    
    if [ "$GPU_INFO" = "ERROR" ]; then
        log_error "PyTorch not properly installed"
        exit 1
    elif [ "$GPU_INFO" = "CPU" ]; then
        log_warn "No GPU detected, running in CPU mode"
    else
        IFS='|' read -r GPU_NAME GPU_MEM <<< "$GPU_INFO"
        log_success "GPU: ${GPU_NAME} (${GPU_MEM})"
    fi
}

# 启动 API
start_api() {
    log_info "Starting API Server on port ${API_PORT}..."
    cd ${PROJECT_ROOT}/api
    
    # 检查端口是否被占用
    if lsof -i:${API_PORT} >/dev/null 2>&1; then
        log_warn "Port ${API_PORT} is in use. Killing existing process..."
        lsof -ti:${API_PORT} | xargs kill -9 2>/dev/null || true
        sleep 1
    fi
    
    # 启动 uvicorn
    nohup python -m uvicorn main:app \
        --host 0.0.0.0 \
        --port ${API_PORT} \
        --reload \
        > ${PROJECT_ROOT}/logs/api.log 2>&1 &
    
    API_PID=$!
    echo ${API_PID} > /tmp/hi_photonics_api.pid
    
    # 等待启动
    for i in {1..10}; do
        if curl -s http://localhost:${API_PORT}/health >/dev/null 2>&1; then
            log_success "API Server running (PID: ${API_PID})"
            log_info "API Docs: http://localhost:${API_PORT}/docs"
            return 0
        fi
        sleep 1
    done
    
    log_error "API Server failed to start. Check logs/api.log"
    return 1
}

# 启动前端
start_ui() {
    log_info "Starting Frontend on port ${UI_PORT}..."
    cd ${PROJECT_ROOT}/frontend
    
    # 检查端口
    if lsof -i:${UI_PORT} >/dev/null 2>&1; then
        log_warn "Port ${UI_PORT} is in use. Killing existing process..."
        lsof -ti:${UI_PORT} | xargs kill -9 2>/dev/null || true
        sleep 1
    fi
    
    # 检查 node_modules
    if [ ! -d "node_modules" ]; then
        log_info "Installing frontend dependencies..."
        npm install --silent
    fi
    
    # 启动 React
    nohup npm start > ${PROJECT_ROOT}/logs/ui.log 2>&1 &
    UI_PID=$!
    echo ${UI_PID} > /tmp/hi_photonics_ui.pid
    
    log_success "Frontend starting (PID: ${UI_PID})"
    log_info "Frontend: http://localhost:${UI_PORT}"
}

# 停止服务
stop_services() {
    log_info "Stopping services..."
    
    if [ -f /tmp/hi_photonics_api.pid ]; then
        kill $(cat /tmp/hi_photonics_api.pid) 2>/dev/null || true
        rm -f /tmp/hi_photonics_api.pid
    fi
    
    if [ -f /tmp/hi_photonics_ui.pid ]; then
        kill $(cat /tmp/hi_photonics_ui.pid) 2>/dev/null || true
        rm -f /tmp/hi_photonics_ui.pid
    fi
    
    pkill -f "uvicorn main:app" 2>/dev/null || true
    pkill -f "react-scripts" 2>/dev/null || true
    
    log_success "All services stopped"
}

# 创建日志目录
mkdir -p ${PROJECT_ROOT}/logs

# 主流程
echo ""
echo -e "${CYAN}============================================================${NC}"
echo -e "${CYAN}  HI-Photonics Studio${NC}"
echo -e "${CYAN}============================================================${NC}"
echo ""

# 设置退出时清理
trap stop_services INT TERM

# 初始化
init_conda
check_gpu

# 根据模式启动
case ${MODE} in
    api)
        start_api
        log_info "API only mode. Press Ctrl+C to stop."
        wait
        ;;
    ui)
        start_ui
        log_info "UI only mode. Press Ctrl+C to stop."
        wait
        ;;
    all)
        start_api
        start_ui
        
        echo ""
        echo -e "${GREEN}============================================================${NC}"
        echo -e "${GREEN}  HI-Photonics Studio Started${NC}"
        echo -e "${GREEN}============================================================${NC}"
        echo ""
        echo -e "  ${CYAN}API Server:${NC}  http://localhost:${API_PORT}"
        echo -e "  ${CYAN}API Docs:${NC}    http://localhost:${API_PORT}/docs"
        echo -e "  ${CYAN}Frontend:${NC}    http://localhost:${UI_PORT}"
        echo ""
        echo -e "  ${YELLOW}Press Ctrl+C to stop all services${NC}"
        echo ""
        
        wait
        ;;
esac
