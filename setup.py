"""
HI-Photonics 安装配置

光子学逆向设计框架，支持多种深度学习模型和仿真器。
"""

import subprocess
import sys
import os
from pathlib import Path
from setuptools import setup, find_packages
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop

# 读取 README
readme_path = Path(__file__).parent / "README.md"
long_description = ""
if readme_path.exists():
    long_description = readme_path.read_text(encoding="utf-8")

# 读取版本
version = "0.1.0"
try:
    from hi_photonics import __version__
    version = __version__
except ImportError:
    pass


def build_optics():
    """编译 Optics C++ FDTD 仿真器"""
    optics_dir = Path(__file__).parent / "interfaces" / "simulators" / "optics"
    
    if not optics_dir.exists():
        print("Optics directory not found, skipping...")
        return
    
    # 检查是否已安装编译依赖
    try:
        import pybind11
        import skbuild
    except ImportError:
        print("Installing build dependencies...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "pybind11>=2.10", "scikit-build-core>=0.12.2"
        ])
    
    # 编译 optics
    print("Building Optics FDTD simulator...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-e", str(optics_dir)
        ])
        print("Optics FDTD simulator built successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Warning: Failed to build optics: {e}")
        print("The optics simulator will not be available, but other features will work.")


class BuildPyCommand(build_py):
    """自定义 build_py 命令，包含 optics 编译"""
    
    def run(self):
        # 编译 optics
        build_optics()
        # 继续正常构建
        build_py.run(self)


class DevelopCommand(develop):
    """自定义 develop 命令，包含 optics 编译"""
    
    def run(self):
        # 编译 optics
        build_optics()
        # 继续正常开发模式安装
        develop.run(self)


setup(
    name="hi-photonics",
    version=version,
    author="HI-Photonics Team",
    author_email="",
    description="Deep learning-based inverse design framework for photonics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SolomonCaptain/HI-Photonics",
    license="GNU General Public License v3.0",
    
    # 包发现
    packages=find_packages(exclude=["tests", "tests.*", "examples", "docs"]),
    
    # Python 版本要求
    python_requires=">=3.9",
    
    # 核心依赖
    install_requires=[
        "torch>=2.0",
        "numpy>=1.20",
        "scipy>=1.7",
        "matplotlib>=3.5",
        "h5py>=3.0",
        "tqdm>=4.60",
        "pyyaml>=6.0",
        "safetensors>=0.4.0",
    ],
    
    # 可选依赖
    extras_require={
        "api": [
            "fastapi>=0.100",
            "uvicorn>=0.23",
            "pydantic>=2.0",
            "httpx>=0.24",
        ],
        "llm": [
            "httpx>=0.24",
            "python-dotenv>=1.0",
            "qdrant-client>=1.7",
            "numpy>=1.20",
        ],
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "isort>=5.0",
            "mypy>=1.0",
        ],
        "meep": [
            "pymeep>=1.20",
        ],
        "optics": [
            "pybind11>=2.10",
            "scikit-build-core>=0.12.2",
        ],
        "docs": [
            "sphinx>=5.0",
            "sphinx-rtd-theme>=1.0",
            "myst-parser>=0.18",
        ],
    },
    
    # 自定义命令
    cmdclass={
        "build_py": BuildPyCommand,
        "develop": DevelopCommand,
    },
    
    # 入口点
    entry_points={
        "console_scripts": [
            "hi-photonics=hi_photonics.cli:main",
        ],
    },
    
    # 包含数据文件
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.json", "*.md"],
    },
    
    # 分类器
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    
    # 关键词
    keywords=[
        "photonics",
        "inverse design",
        "deep learning",
        "FDTD",
        "nanophotonics",
        "optical devices",
        "neural networks",
        "adjoint method",
    ],
)
