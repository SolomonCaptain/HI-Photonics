#!/usr/bin/env python3
"""
GPU 环境验证脚本

检查 CUDA 和 GPU 是否正确配置。
"""

import sys

def check_cuda():
    """检查 CUDA 环境"""
    print("=" * 60)
    print("  GPU Environment Check")
    print("=" * 60)
    
    # 1. Python 版本
    print(f"\n[1] Python: {sys.version}")
    
    # 2. PyTorch
    try:
        import torch
        print(f"[2] PyTorch: {torch.__version__}")
        
        # CUDA 可用性
        cuda_available = torch.cuda.is_available()
        print(f"[3] CUDA Available: {cuda_available}")
        
        if cuda_available:
            # CUDA 版本
            print(f"[4] CUDA Version: {torch.version.cuda}")
            print(f"[5] cuDNN Version: {torch.backends.cudnn.version()}")
            
            # GPU 信息
            num_gpus = torch.cuda.device_count()
            print(f"[6] GPU Count: {num_gpus}")
            
            for i in range(num_gpus):
                props = torch.cuda.get_device_properties(i)
                print(f"\n    GPU {i}: {props.name}")
                print(f"    - Compute Capability: {props.major}.{props.minor}")
                print(f"    - Total Memory: {props.total_memory / 1024**3:.2f} GB")
                print(f"    - Multi-Processors: {props.multi_processor_count}")
            
            # 内存测试
            print("\n[7] GPU Memory Test:")
            x = torch.randn(10000, 10000, device='cuda')
            allocated = torch.cuda.memory_allocated() / 1024**2
            print(f"    - Allocated: {allocated:.2f} MB")
            del x
            torch.cuda.empty_cache()
            print("    - Memory test passed!")
            
        else:
            print("\n[!] CUDA not available. Possible reasons:")
            print("    - No NVIDIA GPU detected")
            print("    - CUDA drivers not installed")
            print("    - PyTorch CPU-only version installed")
            
            # 检查 nvidia-smi
            import subprocess
            try:
                result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
                if result.returncode == 0:
                    print("\n[nvidia-smi output]:")
                    print(result.stdout[:500])
            except FileNotFoundError:
                print("\n[nvidia-smi not found]")
    
    except ImportError:
        print("[!] PyTorch not installed")
        print("    Install with: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu120")
        return False
    
    print("\n" + "=" * 60)
    
    return cuda_available


def test_training():
    """测试 GPU 训练"""
    print("\n[8] Quick Training Test:")
    
    try:
        import torch
        import torch.nn as nn
        
        if not torch.cuda.is_available():
            print("    - Skipped (no CUDA)")
            return
        
        # 简单模型
        model = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        ).cuda()
        
        # 数据
        x = torch.randn(64, 1000).cuda()
        y = torch.randn(64, 10).cuda()
        
        # 训练步骤
        optimizer = torch.optim.Adam(model.parameters())
        criterion = nn.MSELoss()
        
        import time
        start = time.time()
        
        for _ in range(100):
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
        
        elapsed = time.time() - start
        print(f"    - 100 iterations: {elapsed*1000:.1f} ms")
        print(f"    - Throughput: {100/elapsed:.1f} iter/s")
        print("    - Training test passed!")
        
    except Exception as e:
        print(f"    - Test failed: {e}")


if __name__ == "__main__":
    cuda_ok = check_cuda()
    
    if cuda_ok:
        test_training()
    
    print("\n" + "=" * 60)
    if cuda_ok:
        print("  ✓ GPU Environment Ready!")
    else:
        print("  ✗ GPU Environment Not Ready")
        print("\n  Troubleshooting:")
        print("  1. Install NVIDIA drivers: sudo apt install nvidia-driver-535")
        print("  2. Install CUDA toolkit: https://developer.nvidia.com/cuda-downloads")
        print("  3. Reinstall PyTorch with CUDA:")
        print("     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu120")
    print("=" * 60)