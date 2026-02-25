import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional
from tqdm import tqdm
from transformers import AutoTokenizer


def evaluate_loss(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> float:
    """
    评估模型损失
    
    Args:
        model: 模型实例
        data_loader: 数据加载器
        criterion: 损失函数
        device: 设备
        
    Returns:
        平均损失
    """
    model.eval()
    total_loss = 0.0
    total_steps = 0
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating Loss"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            lm_logits, _ = model(input_ids, attention_mask)
            loss = criterion(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            
            total_loss += loss.item()
            total_steps += 1
    
    return total_loss / total_steps


def generate_text(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_length: int = 100,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.95,
    device: torch.device = None
) -> str:
    """
    生成文本
    
    Args:
        model: 模型实例
        tokenizer: 分词器
        prompt: 提示文本
        max_length: 最大生成长度
        temperature: 采样温度
        top_k: 	top-k采样
        top_p: 	top-p采样
        device: 设备
        
    Returns:
        生成的文本
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.eval()
    model.to(device)
    
    # 编码提示文本
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    # 生成文本
    generated_ids = input_ids.clone()
    
    with torch.no_grad():
        for _ in range(max_length):
            # 前向传播
            lm_logits, _ = model(generated_ids, attention_mask)
            
            # 获取最后一个token的logits
            next_token_logits = lm_logits[:, -1, :] / temperature
            
            # 应用top-k和top-p采样
            if top_k > 0:
                next_token_logits = _top_k_filtering(next_token_logits, top_k=top_k)
            if top_p < 1.0:
                next_token_logits = _top_p_filtering(next_token_logits, top_p=top_p)
            
            # 采样下一个token
            next_token = torch.multinomial(torch.softmax(next_token_logits, dim=-1), num_samples=1)
            
            # 添加到生成序列
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=-1)
            
            # 检查是否生成了结束token
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    # 解码生成的文本
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_text


def _top_k_filtering(logits: torch.Tensor, top_k: int = 50) -> torch.Tensor:
    """
    Top-k过滤
    
    Args:
        logits: 模型输出的logits
        top_k: 保留的token数量
        
    Returns:
        过滤后的logits
    """
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # 获取top-k的logits值
        values, _ = torch.topk(logits, top_k)
        min_values = values[:, -1].unsqueeze(1)
        # 将非top-k的logits设置为负无穷
        logits = torch.where(logits < min_values, torch.full_like(logits, -float('inf')), logits)
    return logits


def _top_p_filtering(logits: torch.Tensor, top_p: float = 0.95) -> torch.Tensor:
    """
    Top-p过滤
    
    Args:
        logits: 模型输出的logits
        top_p: 累积概率阈值
        
    Returns:
        过滤后的logits
    """
    if top_p < 1.0:
        # 排序logits
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        # 计算累积概率
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        # 移除累积概率超过top_p的token
        sorted_indices_to_remove = cumulative_probs > top_p
        # 保留第一个超过top_p的token
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = 0
        # 创建掩码
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[:, indices_to_remove] = -float('inf')
    return logits


def calculate_perplexity(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device = None
) -> float:
    """
    计算困惑度
    
    Args:
        model: 模型实例
        data_loader: 数据加载器
        device: 设备
        
    Returns:
        困惑度
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.eval()
    model.to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Calculating Perplexity"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            lm_logits, _ = model(input_ids, attention_mask)
            loss = criterion(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            
            # 计算有效token数量
            mask = labels.view(-1) != -100
            total_loss += loss[mask].sum().item()
            total_tokens += mask.sum().item()
    
    # 计算困惑度
    perplexity = torch.exp(torch.tensor(total_loss / total_tokens)).item()
    return perplexity


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device = None
) -> Dict[str, float]:
    """
    评估模型
    
    Args:
        model: 模型实例
        data_loader: 数据加载器
        device: 设备
        
    Returns:
        评估结果
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 计算损失
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    loss = evaluate_loss(model, data_loader, criterion, device)
    
    # 计算困惑度
    perplexity = calculate_perplexity(model, data_loader, device)
    
    return {
        'loss': loss,
        'perplexity': perplexity
    }
