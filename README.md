# 轻量化语言模型训练架构 (LightweightLM)

<div align="center">

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.7%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Stars](https://img.shields.io/github/stars/CanFlyhang/llm.svg?style=social)

</div>

## 📚 项目简介

LightweightLM 是一个高效、灵活的轻量化语言模型训练架构，基于 Transformer 架构设计，专为资源受限环境打造。

- **参数量少**：相比传统大模型，参数量大幅减少，适合在普通 GPU 或甚至 CPU 上训练
- **推理速度快**：优化的模型结构，实现更快的推理速度
- **功能完整**：包含从数据处理到模型训练、评估和部署的完整流程
- **易于扩展**：模块化设计，支持多种模型配置和训练策略

## 🚀 快速开始

### 环境依赖

```bash
# 安装核心依赖
pip install torch transformers tqdm
```

### 准备数据

在 `data/` 目录下创建 `train.txt` 文件，每行一条文本数据：

```bash
# 示例数据格式
这是第一条训练数据
这是第二条训练数据
这是第三条训练数据
```

### 训练模型

```bash
# 使用默认配置训练
python main.py

# 自定义配置训练
python main.py --hidden_size 128 --num_layers 2 --num_heads 2 --batch_size 16

# 在CPU上训练（适合内存受限环境）
python main.py --device cpu --batch_size 8
```

### 生成文本

```python
from evaluate import generate_text
from model import create_lightweight_model
from utils import get_device
import torch

# 加载模型
model = create_lightweight_model()
model.load_state_dict(torch.load("./models/best_model.pt")['model_state_dict'])
model.to(get_device())
model.eval()

# 生成文本
prompt = "人工智能的未来"
generated = generate_text(model, prompt, max_length=100, temperature=0.7)
print(f"生成结果: {generated}")
```

## 📁 项目结构

```
.
├── data/             # 数据集目录
│   └── train.txt     # 训练数据
├── models/           # 模型保存目录
├── data.py           # 数据处理模块
├── model.py          # 模型定义模块
├── train.py          # 训练模块
├── evaluate.py       # 评估模块
├── config.py         # 配置模块
├── utils.py          # 工具模块
├── main.py           # 主训练脚本
└── README.md         # 项目说明
```

## 🎯 核心功能

### 模型特性

- **轻量化设计**：基于小型 Transformer 架构，参数量少，适合资源受限环境
- **完整训练流程**：包含数据处理、模型训练、评估和保存功能
- **灵活配置**：支持多种模型和训练参数配置
- **文本生成**：内置文本生成功能，支持温度采样、top-k 和 top-p 采样
- **性能评估**：支持困惑度等指标评估

### 训练策略

- **自监督学习**：使用自回归语言建模任务
- **AdamW 优化器**：带有权重衰减的 Adam 优化器
- **学习率调度**：线性学习率衰减
- **梯度累积**：支持多步梯度累积，模拟更大批次
- **模型保存**：自动保存最佳验证模型

## 📊 模型参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| hidden_size | 256 | 隐藏层维度 |
| num_layers | 4 | Transformer 层数 |
| num_heads | 4 | 注意力头数 |
| intermediate_size | 1024 | 前馈网络中间层维度 |
| max_position_embeddings | 512 | 最大位置编码 |
| hidden_dropout_prob | 0.1 | 隐藏层 dropout 概率 |
| attention_probs_dropout_prob | 0.1 | 注意力 dropout 概率 |

## 💡 使用场景

- **移动设备**：资源受限的移动设备上的文本处理
- **边缘计算**：边缘设备上的实时推理
- **嵌入式系统**：嵌入式系统中的语言理解
- **实时应用**：需要快速响应的实时文本生成应用
- **教育研究**：学习语言模型原理和实现
- **原型开发**：快速验证语言模型相关想法

## 🔧 高级配置

### 命令行参数

```bash
# 基本训练参数
python main.py --train_data ./data/train.txt --epochs 10 --batch_size 32

# 模型架构参数
python main.py --hidden_size 128 --num_layers 2 --num_heads 2

# 优化器参数
python main.py --learning_rate 1e-4 --weight_decay 0.01

# 生成参数
python main.py --temperature 0.7 --top_k 50 --top_p 0.9
```

### 内存优化

对于内存受限的环境，可以尝试以下配置：

```bash
# 小批量训练
python main.py --batch_size 4 --gradient_accumulation_steps 8

# 更小的模型
python main.py --hidden_size 64 --num_layers 1 --num_heads 1

# 混合精度训练（需要 apex 库）
python main.py --fp16
```

## 📈 性能评估

训练完成后，可以通过以下指标评估模型性能：

- **损失值**：训练和验证过程中的交叉熵损失
- **困惑度**：语言模型预测能力的衡量指标，值越低越好
- **文本生成质量**：通过人工评估生成文本的连贯性和合理性

```python
from evaluate import calculate_perplexity
from data import get_data_loader

# 计算困惑度
data_loader = get_data_loader("./data/val.txt", batch_size=16)
perplexity = calculate_perplexity(model, data_loader)
print(f"困惑度: {perplexity:.2f}")
```

## 🤝 贡献指南

我们欢迎社区贡献！如果你有兴趣参与项目，请遵循以下步骤：

1. **Fork 项目**：在 GitHub 上 fork 本项目到你的账号
2. **创建分支**：从 `main` 分支创建一个新的功能分支
3. **提交更改**：实现你的功能或修复，并提交更改
4. **创建 PR**：创建一个 Pull Request，描述你的更改内容

### 开发规范

- 代码风格：遵循 PEP 8 规范
- 提交信息：使用清晰、简洁的提交信息
- 测试：为新功能添加适当的测试
- 文档：更新相关文档

## 📄 许可证

本项目采用 MIT 许可证，详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

本项目参考了以下资源：

- [Transformer 原始论文](https://arxiv.org/abs/1706.03762)：Attention Is All You Need
- [Hugging Face Transformers 库](https://github.com/huggingface/transformers)
- [PyTorch 官方文档](https://pytorch.org/docs/stable/)

## 📞 联系方式

如有问题或建议，欢迎通过以下方式联系：

- **GitHub Issues**：在 [项目页面](https://github.com/CanFlyhang/llm/issues) 提交 Issue
- **Email**：2153208034@qq.com

---

<div align="center">

⭐️ 如果你觉得这个项目有用，请给它点个星！

</div>
