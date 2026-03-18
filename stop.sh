#!/bin/bash
#
# HI-Photonics 停止脚本
#

echo "Stopping HI-Photonics services..."

# 通过 PID 文件停止
for pid_file in /tmp/hi_photonics_*.pid; do
    if [ -f "$pid_file" ]; then
        pid=$(cat "$pid_file")
        kill $pid 2>/dev/null && echo "  Killed process $pid"
        rm -f "$pid_file"
    fi
done

# 额外清理
pkill -f "uvicorn main:app" 2>/dev/null && echo "  Stopped API server"
pkill -f "react-scripts" 2>/dev/null && echo "  Stopped frontend"

echo "All services stopped."
