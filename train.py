import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional
from tqdm import tqdm


def train_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    accumulation_steps: int = 1
) -> float:
    """
    训练一个epoch
    
    Args:
        model: 模型实例
        data_loader: 数据加载器
        optimizer: 优化器
        criterion: 损失函数
        device: 设备
        accumulation_steps: 梯度累积步数
        
    Returns:
        平均损失
    """
    model.train()
    total_loss = 0.0
    total_steps = 0
    
    # 清空梯度
    optimizer.zero_grad()
    
    for step, batch in enumerate(tqdm(data_loader, desc="Training")):
        # 移动数据到设备
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # 前向传播
        lm_logits, _ = model(input_ids, attention_mask)
        
        # 计算损失
        loss = criterion(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
        loss = loss / accumulation_steps
        
        # 反向传播
        loss.backward()
        
        # 梯度累积
        if (step + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps
        total_steps += 1
    
    # 确保最后一个批次的梯度也被更新
    if total_steps % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()
    
    return total_loss / total_steps


def validate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> float:
    """
    验证模型
    
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
        for batch in tqdm(data_loader, desc="Validation"):
            # 移动数据到设备
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # 前向传播
            lm_logits, _ = model(input_ids, attention_mask)
            
            # 计算损失
            loss = criterion(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            
            total_loss += loss.item()
            total_steps += 1
    
    return total_loss / total_steps


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    epochs: int = 10,
    learning_rate: float = 5e-5,
    batch_size: int = 32,
    accumulation_steps: int = 1,
    save_dir: str = "./models",
    device: Optional[torch.device] = None,
    log_interval: int = 100
) -> Dict[str, Any]:
    """
    训练模型
    
    Args:
        model: 模型实例
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        epochs: 训练轮数
        learning_rate: 学习率
        batch_size: 批次大小
        accumulation_steps: 梯度累积步数
        save_dir: 模型保存目录
        device: 设备
        log_interval: 日志间隔
        
    Returns:
        训练结果
    """
    # 设置设备
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 配置优化器和损失函数
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)  # 忽略填充token
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.1,
        total_iters=epochs
    )
    
    # 训练记录
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print(f"Training on {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        print("-" * 50)
        
        # 训练
        start_time = time.time()
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            accumulation_steps
        )
        train_losses.append(train_loss)
        
        # 验证
        if val_loader is not None:
            val_loss = validate(model, val_loader, criterion, device)
            val_losses.append(val_loss)
            
            # 更新学习率
            scheduler.step()
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model_path = os.path.join(save_dir, f"best_model_epoch_{epoch}.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': val_loss,
                }, model_path)
                print(f"Saved best model to {model_path}")
            
            # 打印日志
            elapsed_time = time.time() - start_time
            print(f"Time: {elapsed_time:.2f}s")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Best Val Loss: {best_val_loss:.4f}")
        else:
            # 没有验证集时也更新学习率
            scheduler.step()
            
            # 保存模型
            model_path = os.path.join(save_dir, f"model_epoch_{epoch}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
            }, model_path)
            print(f"Saved model to {model_path}")
            
            # 打印日志
            elapsed_time = time.time() - start_time
            print(f"Time: {elapsed_time:.2f}s")
            print(f"Train Loss: {train_loss:.4f}")
    
    # 保存最终模型
    final_model_path = os.path.join(save_dir, "final_model.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
    }, final_model_path)
    print(f"Saved final model to {final_model_path}")
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'final_model_path': final_model_path
    }
