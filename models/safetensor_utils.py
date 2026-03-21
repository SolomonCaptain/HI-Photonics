"""
Safetensor 格式工具模块

提供模型格式转换、加载、验证等功能。
"""

import torch
from pathlib import Path
from datetime import datetime
import json
from typing import Dict, Any, Optional, Union, List

try:
    import safetensors.torch
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False


def check_safetensors_available() -> bool:
    """检查 safetensors 是否可用"""
    return SAFETENSORS_AVAILABLE


def convert_torch_to_safetensors(
    torch_path: Union[str, Path],
    safetensors_path: Optional[Union[str, Path]] = None,
    include_metadata: bool = True
) -> Path:
    """
    将 PyTorch 模型转换为 safetensors 格式
    
    Args:
        torch_path: PyTorch 模型路径
        safetensors_path: safetensors 保存路径（可选，默认自动生成）
        include_metadata: 是否保存元数据
        
    Returns:
        safetensors 文件路径
    """
    if not SAFETENSORS_AVAILABLE:
        raise ImportError("safetensors 库未安装，请运行: pip install safetensors")
    
    torch_path = Path(torch_path)
    
    if safetensors_path is None:
        safetensors_path = torch_path.with_suffix('.safetensors')
    else:
        safetensors_path = Path(safetensors_path)
    
    # 加载 PyTorch 模型
    checkpoint = torch.load(torch_path, map_location='cpu', weights_only=False)
    
    # 提取 state_dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # 保存权重为 safetensors
    safetensors.torch.save_file(state_dict, str(safetensors_path))
    
    # 保存元数据
    if include_metadata:
        metadata = {
            'source_file': str(torch_path),
            'converted_at': datetime.now().isoformat(),
            'pytorch_version': torch.__version__,
        }
        
        # 从 checkpoint 提取额外信息
        if 'config' in checkpoint:
            metadata['config'] = checkpoint['config']
        if 'training_history' in checkpoint:
            metadata['training_history'] = checkpoint['training_history']
        
        metadata_path = safetensors_path.with_suffix('.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
    
    return safetensors_path


def convert_safetensors_to_torch(
    safetensors_path: Union[str, Path],
    torch_path: Optional[Union[str, Path]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Path:
    """
    将 safetensors 格式转换为 PyTorch 格式
    
    Args:
        safetensors_path: safetensors 文件路径
        torch_path: PyTorch 保存路径（可选）
        metadata: 额外元数据（可选）
        
    Returns:
        PyTorch 文件路径
    """
    if not SAFETENSORS_AVAILABLE:
        raise ImportError("safetensors 库未安装，请运行: pip install safetensors")
    
    safetensors_path = Path(safetensors_path)
    
    if torch_path is None:
        torch_path = safetensors_path.with_suffix('.pt')
    else:
        torch_path = Path(torch_path)
    
    # 加载 safetensors 权重
    state_dict = safetensors.torch.load_file(str(safetensors_path))
    
    # 构建 checkpoint
    checkpoint = {
        'model_state_dict': state_dict,
    }
    
    # 合并元数据
    metadata_path = safetensors_path.with_suffix('.json')
    if metadata_path.exists():
        with open(metadata_path, 'r', encoding='utf-8') as f:
            file_metadata = json.load(f)
        checkpoint['config'] = file_metadata.get('config', {})
        checkpoint['training_history'] = file_metadata.get('training_history', {})
    
    if metadata:
        checkpoint.update(metadata)
    
    torch.save(checkpoint, torch_path)
    
    return torch_path


def load_safetensors_metadata(safetensors_path: Union[str, Path]) -> Dict[str, Any]:
    """
    加载 safetensors 模型的元数据
    
    Args:
        safetensors_path: safetensors 文件路径
        
    Returns:
        元数据字典
    """
    safetensors_path = Path(safetensors_path)
    metadata_path = safetensors_path.with_suffix('.json')
    
    if metadata_path.exists():
        with open(metadata_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    return {}


def get_safetensors_info(safetensors_path: Union[str, Path]) -> Dict[str, Any]:
    """
    获取 safetensors 模型的详细信息
    
    Args:
        safetensors_path: safetensors 文件路径
        
    Returns:
        模型信息字典
    """
    if not SAFETENSORS_AVAILABLE:
        raise ImportError("safetensors 库未安装，请运行: pip install safetensors")
    
    safetensors_path = Path(safetensors_path)
    
    # 加载权重
    state_dict = safetensors.torch.load_file(str(safetensors_path))
    
    # 统计参数
    total_params = sum(t.numel() for t in state_dict.values())
    total_size = sum(t.numel() * t.element_size() for t in state_dict.values())
    
    # 获取张量信息
    tensor_info = {}
    for name, tensor in state_dict.items():
        tensor_info[name] = {
            'shape': list(tensor.shape),
            'dtype': str(tensor.dtype),
            'numel': tensor.numel(),
            'size_bytes': tensor.numel() * tensor.element_size()
        }
    
    # 合并元数据
    metadata = load_safetensors_metadata(safetensors_path)
    
    return {
        'path': str(safetensors_path),
        'file_size_bytes': safetensors_path.stat().st_size,
        'total_parameters': total_params,
        'total_size_bytes': total_size,
        'num_tensors': len(state_dict),
        'tensors': tensor_info,
        'metadata': metadata
    }


def batch_convert_to_safetensors(
    directory: Union[str, Path],
    pattern: str = "*.pt",
    delete_original: bool = False
) -> List[Path]:
    """
    批量转换目录中的 PyTorch 模型为 safetensors 格式
    
    Args:
        directory: 模型目录
        pattern: 文件匹配模式
        delete_original: 是否删除原始文件
        
    Returns:
        转换后的文件路径列表
    """
    if not SAFETENSORS_AVAILABLE:
        raise ImportError("safetensors 库未安装，请运行: pip install safetensors")
    
    directory = Path(directory)
    converted_files = []
    
    for torch_file in directory.glob(pattern):
        if torch_file.suffix in ['.pt', '.pth', '.ckpt']:
            try:
                safetensors_path = convert_torch_to_safetensors(torch_file)
                converted_files.append(safetensors_path)
                
                if delete_original:
                    torch_file.unlink()
                    print(f"Converted and deleted: {torch_file}")
                else:
                    print(f"Converted: {torch_file} -> {safetensors_path}")
            except Exception as e:
                print(f"Failed to convert {torch_file}: {e}")
    
    return converted_files


def validate_safetensors_file(safetensors_path: Union[str, Path]) -> Dict[str, Any]:
    """
    验证 safetensors 文件完整性
    
    Args:
        safetensors_path: safetensors 文件路径
        
    Returns:
        验证结果
    """
    if not SAFETENSORS_AVAILABLE:
        raise ImportError("safetensors 库未安装，请运行: pip install safetensors")
    
    safetensors_path = Path(safetensors_path)
    
    result = {
        'valid': False,
        'path': str(safetensors_path),
        'errors': [],
        'warnings': []
    }
    
    # 检查文件存在
    if not safetensors_path.exists():
        result['errors'].append(f"文件不存在: {safetensors_path}")
        return result
    
    # 检查文件扩展名
    if safetensors_path.suffix not in ['.safetensors', '.sft']:
        result['warnings'].append(f"非标准扩展名: {safetensors_path.suffix}")
    
    try:
        # 尝试加载
        state_dict = safetensors.torch.load_file(str(safetensors_path))
        
        # 检查是否为空
        if len(state_dict) == 0:
            result['errors'].append("模型为空")
            return result
        
        # 检查张量有效性
        for name, tensor in state_dict.items():
            if not isinstance(tensor, torch.Tensor):
                result['errors'].append(f"无效张量类型: {name}")
            if tensor.numel() == 0:
                result['warnings'].append(f"空张量: {name}")
        
        # 检查元数据文件
        metadata_path = safetensors_path.with_suffix('.json')
        if not metadata_path.exists():
            result['warnings'].append("缺少元数据文件")
        
        result['valid'] = len(result['errors']) == 0
        result['num_tensors'] = len(state_dict)
        result['total_parameters'] = sum(t.numel() for t in state_dict.values())
        
    except Exception as e:
        result['errors'].append(f"加载失败: {str(e)}")
    
    return result
