import os
import torch
import logging
from typing import Dict, Any, Optional
from model import create_lightweight_model


def get_device(device: Optional[str] = None) -> torch.device:
    """
    获取设备
    
    Args:
        device: 设备名称
        
    Returns:
        设备实例
    """
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_model(
    model: torch.nn.Module,
    path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    loss: Optional[float] = None
):
    """
    保存模型
    
    Args:
        model: 模型实例
        path: 保存路径
        optimizer: 优化器
        epoch: 当前轮数
        loss: 当前损失
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    save_dict = {
        'model_state_dict': model.state_dict(),
    }
    
    if optimizer is not None:
        save_dict['optimizer_state_dict'] = optimizer.state_dict()
    if epoch is not None:
        save_dict['epoch'] = epoch
    if loss is not None:
        save_dict['loss'] = loss
    
    torch.save(save_dict, path)
    print(f"Model saved to {path}")


def load_model(
    path: str,
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    加载模型
    
    Args:
        path: 模型路径
        model: 模型实例
        optimizer: 优化器
        device: 设备
        
    Returns:
        加载的模型信息
    """
    if device is None:
        device = get_device()
    
    checkpoint = torch.load(path, map_location=device)
    
    if model is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Model loaded from {path}")
    return checkpoint


def setup_logging(log_file: Optional[str] = None, log_level: int = logging.INFO):
    """
    设置日志
    
    Args:
        log_file: 日志文件路径
        log_level: 日志级别
    """
    # 配置日志
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    
    # 添加文件处理器
    if log_file is not None:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        logging.getLogger().addHandler(file_handler)


def count_parameters(model: torch.nn.Module) -> int:
    """
    计算模型参数数量
    
    Args:
        model: 模型实例
        
    Returns:
        参数数量
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size(model: torch.nn.Module) -> float:
    """
    获取模型大小（MB）
    
    Args:
        model: 模型实例
        
    Returns:
        模型大小
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb


def create_directory(path: str):
    """
    创建目录
    
    Args:
        path: 目录路径
    """
    os.makedirs(path, exist_ok=True)


def get_file_list(directory: str, extension: Optional[str] = None) -> list:
    """
    获取文件列表
    
    Args:
        directory: 目录路径
        extension: 文件扩展名
        
    Returns:
        文件列表
    """
    file_list = []
    for root, _, files in os.walk(directory):
        for file in files:
            if extension is None or file.endswith(extension):
                file_list.append(os.path.join(root, file))
    return file_list


def print_model_summary(model: torch.nn.Module):
    """
    打印模型摘要
    
    Args:
        model: 模型实例
    """
    print("Model Summary:")
    print("=" * 80)
    print(model)
    print("=" * 80)
    print(f"Total parameters: {count_parameters(model):,}")
    print(f"Model size: {get_model_size(model):.2f} MB")
    print("=" * 80)


def get_latest_model(path: str) -> Optional[str]:
    """
    获取最新的模型文件
    
    Args:
        path: 模型目录
        
    Returns:
        最新模型文件路径
    """
    if not os.path.exists(path):
        return None
    
    model_files = get_file_list(path, ".pt")
    if not model_files:
        return None
    
    # 按修改时间排序
    model_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return model_files[0]


def freeze_layers(model: torch.nn.Module, freeze_percentage: float = 0.5):
    """
    冻结部分层
    
    Args:
        model: 模型实例
        freeze_percentage: 冻结比例
    """
    total_layers = len(list(model.named_parameters()))
    freeze_layers = int(total_layers * freeze_percentage)
    
    for i, (name, param) in enumerate(model.named_parameters()):
        if i < freeze_layers:
            param.requires_grad = False
        else:
            param.requires_grad = True
    
    print(f"Froze {freeze_layers}/{total_layers} layers")
