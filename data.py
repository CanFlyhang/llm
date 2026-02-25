import os
import json
from typing import List, Dict, Any
import torch
from torch.utils.data import Dataset, DataLoader


class SimpleTokenizer:
    """
    简单分词器，避免依赖网络下载
    """
    
    def __init__(self):
        """
        初始化分词器
        """
        # 简单的词汇表
        self.vocab = {
            '<pad>': 0,
            '<sos>': 1,
            '<eos>': 2,
            '<unk>': 3
        }
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)
        self.pad_token_id = self.vocab['<pad>']
        self.sos_token_id = self.vocab['<sos>']
        self.eos_token_id = self.vocab['<eos>']
        self.unk_token_id = self.vocab['<unk>']
    
    def _tokenize(self, text: str) -> List[str]:
        """
        分词
        
        Args:
            text: 输入文本
            
        Returns:
            分词结果
        """
        # 简单的按空格分词
        return text.split()
    
    def _encode(self, tokens: List[str]) -> List[int]:
        """
        编码
        
        Args:
            tokens: 分词结果
            
        Returns:
            编码结果
        """
        ids = []
        for token in tokens:
            if token not in self.vocab:
                self.vocab[token] = self.vocab_size
                self.reverse_vocab[self.vocab_size] = token
                self.vocab_size += 1
            ids.append(self.vocab[token])
        return ids
    
    def __call__(self, text: str, max_length: int = 128, padding: str = 'max_length', 
                 truncation: bool = True, return_tensors: str = 'pt') -> Dict[str, Any]:
        """
        分词和编码
        
        Args:
            text: 输入文本
            max_length: 最大长度
            padding: 填充方式
            truncation: 是否截断
            return_tensors: 返回张量类型
            
        Returns:
            编码结果
        """
        # 分词
        tokens = self._tokenize(text)
        # 编码
        ids = self._encode(tokens)
        
        # 添加特殊标记
        ids = [self.sos_token_id] + ids + [self.eos_token_id]
        
        # 截断
        if truncation and len(ids) > max_length:
            ids = ids[:max_length]
        
        # 填充
        attention_mask = [1] * len(ids)
        if padding == 'max_length' and len(ids) < max_length:
            pad_length = max_length - len(ids)
            ids += [self.pad_token_id] * pad_length
            attention_mask += [0] * pad_length
        
        # 转换为张量
        if return_tensors == 'pt':
            ids = torch.tensor([ids])
            attention_mask = torch.tensor([attention_mask])
        
        return {
            'input_ids': ids,
            'attention_mask': attention_mask
        }


class TextDataset(Dataset):
    """
    文本数据集类，用于加载和处理训练数据
    """
    
    def __init__(self, data_path: str, tokenizer_name: str = 'simple', max_length: int = 128):
        """
        初始化数据集
        
        Args:
            data_path: 数据文件路径
            tokenizer_name: 分词器名称
            max_length: 最大序列长度
        """
        self.data = self._load_data(data_path)
        self.tokenizer = SimpleTokenizer()  # 使用简单分词器
        self.max_length = max_length
        
        # 预处理所有数据，构建完整词汇表
        self._build_vocab()
    
    def _load_data(self, data_path: str) -> List[str]:
        """
        加载数据
        
        Args:
            data_path: 数据文件路径
            
        Returns:
            文本数据列表
        """
        data = []
        if os.path.exists(data_path):
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data.append(line)
        return data
    
    def __len__(self) -> int:
        """
        返回数据集长度
        """
        return len(self.data)
    
    def _build_vocab(self):
        """
        构建完整词汇表
        """
        for text in self.data:
            # 分词处理
            tokens = self.tokenizer._tokenize(text)
            # 编码处理（会自动添加新token到词汇表）
            self.tokenizer._encode(tokens)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取单个样本
        
        Args:
            idx: 样本索引
            
        Returns:
            分词后的样本
        """
        text = self.data[idx]
        # 分词处理
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 构建标签（自回归任务，输入偏移一位作为标签）
        input_ids = encoding['input_ids'].squeeze()
        labels = input_ids.clone()
        labels[:-1] = input_ids[1:]
        labels[-1] = self.tokenizer.pad_token_id
        
        return {
            'input_ids': input_ids,
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': labels
        }


def get_data_loader(data_path: str, batch_size: int = 32, shuffle: bool = True, **kwargs) -> DataLoader:
    """
    获取数据加载器
    
    Args:
        data_path: 数据文件路径
        batch_size: 批次大小
        shuffle: 是否打乱数据
        **kwargs: 其他参数
        
    Returns:
        数据加载器
    """
    dataset = TextDataset(data_path, **kwargs)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True
    )


def prepare_dataset(texts: List[str], output_path: str):
    """
    准备数据集
    
    Args:
        texts: 文本列表
        output_path: 输出文件路径
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for text in texts:
            f.write(text + '\n')
