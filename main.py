import argparse
import torch
from data import get_data_loader
from model import create_lightweight_model, Config
from train import train


def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description="Train a lightweight language model")
    
    # 数据参数
    parser.add_argument("--train_data", type=str, default="./data/train.txt", help="Path to training data")
    parser.add_argument("--val_data", type=str, default=None, help="Path to validation data")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length")
    
    # 模型参数
    parser.add_argument("--hidden_size", type=int, default=256, help="Hidden size")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--intermediate_size", type=int, default=1024, help="Intermediate size")
    
    # 训练参数
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--save_dir", type=str, default="./models", help="Directory to save models")
    
    # 其他参数
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda or cpu)")
    parser.add_argument("--log_interval", type=int, default=100, help="Log interval")
    
    return parser.parse_args()


def main():
    """
    主函数
    """
    # 解析参数
    args = parse_args()
    
    # 设置设备
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # 先创建数据集实例获取词汇表大小
    from data import TextDataset
    train_dataset = TextDataset(args.train_data, max_length=args.max_length)
    vocab_size = train_dataset.tokenizer.vocab_size
    print(f"Vocabulary size: {vocab_size}")
    
    # 创建模型配置
    config = Config(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        intermediate_size=args.intermediate_size
    )
    
    # 创建模型
    model = create_lightweight_model(config, vocab_size=vocab_size)
    print(f"Model created with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameters")
    
    # 获取数据加载器
    from torch.utils.data import DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True
    )
    
    val_loader = None
    if args.val_data is not None:
        # 使用与训练集相同的分词器
        from data import TextDataset
        val_dataset = TextDataset(args.val_data, max_length=args.max_length)
        # 确保验证集使用与训练集相同的词汇表
        val_dataset.tokenizer = train_dataset.tokenizer
        from torch.utils.data import DataLoader
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=True
        )
    
    # 开始训练
    print("\nStarting training...")
    results = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        accumulation_steps=args.accumulation_steps,
        save_dir=args.save_dir,
        device=device,
        log_interval=args.log_interval
    )
    
    # 打印训练结果
    print("\nTraining completed!")
    print(f"Best validation loss: {results.get('best_val_loss', 'N/A'):.4f}")
    print(f"Final model saved to: {results['final_model_path']}")


if __name__ == "__main__":
    main()
