import json
import os
from typing import Dict, Any, Optional


class TrainingConfig:
    """
    训练配置类
    """
    
    def __init__(
        self,
        # 数据参数
        train_data: str = "./data/train.txt",
        val_data: Optional[str] = None,
        batch_size: int = 32,
        max_length: int = 128,
        
        # 模型参数
        hidden_size: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        intermediate_size: int = 1024,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        
        # 训练参数
        epochs: int = 10,
        learning_rate: float = 5e-5,
        accumulation_steps: int = 1,
        save_dir: str = "./models",
        
        # 其他参数
        device: Optional[str] = None,
        log_interval: int = 100
    ):
        """
        初始化训练配置
        
        Args:
            train_data: 训练数据路径
            val_data: 验证数据路径
            batch_size: 批次大小
            max_length: 最大序列长度
            hidden_size: 隐藏层维度
            num_layers: Transformer层数
            num_heads: 注意力头数
            intermediate_size: 中间层维度
            hidden_dropout_prob: 隐藏层dropout概率
            attention_probs_dropout_prob: 注意力dropout概率
            epochs: 训练轮数
            learning_rate: 学习率
            accumulation_steps: 梯度累积步数
            save_dir: 模型保存目录
            device: 设备
            log_interval: 日志间隔
        """
        # 数据参数
        self.train_data = train_data
        self.val_data = val_data
        self.batch_size = batch_size
        self.max_length = max_length
        
        # 模型参数
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        
        # 训练参数
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.accumulation_steps = accumulation_steps
        self.save_dir = save_dir
        
        # 其他参数
        self.device = device
        self.log_interval = log_interval
    
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典
        
        Returns:
            配置字典
        """
        return {
            "train_data": self.train_data,
            "val_data": self.val_data,
            "batch_size": self.batch_size,
            "max_length": self.max_length,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "intermediate_size": self.intermediate_size,
            "hidden_dropout_prob": self.hidden_dropout_prob,
            "attention_probs_dropout_prob": self.attention_probs_dropout_prob,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "accumulation_steps": self.accumulation_steps,
            "save_dir": self.save_dir,
            "device": self.device,
            "log_interval": self.log_interval
        }
    
    def save(self, path: str):
        """
        保存配置
        
        Args:
            path: 保存路径
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, path: str) -> 'TrainingConfig':
        """
        加载配置
        
        Args:
            path: 配置文件路径
            
        Returns:
            配置实例
        """
        with open(path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls(**config_dict)


def get_default_config() -> TrainingConfig:
    """
    获取默认配置
    
    Returns:
        默认配置实例
    """
    return TrainingConfig()


def get_small_config() -> TrainingConfig:
    """
    获取小模型配置
    
    Returns:
        小模型配置实例
    """
    return TrainingConfig(
        hidden_size=128,
        num_layers=2,
        num_heads=2,
        intermediate_size=512,
        batch_size=64
    )


def get_large_config() -> TrainingConfig:
    """
    获取大模型配置
    
    Returns:
        大模型配置实例
    """
    return TrainingConfig(
        hidden_size=512,
        num_layers=6,
        num_heads=8,
        intermediate_size=2048,
        batch_size=16
    )
