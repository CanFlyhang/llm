import torch
import torch.nn as nn
from typing import Optional, Tuple


class Config:
    """
    模型配置类
    """
    
    def __init__(self,
                 vocab_size: int = 30522,  # BERT词汇表大小
                 hidden_size: int = 256,     # 隐藏层维度
                 num_hidden_layers: int = 4,  # Transformer层数
                 num_attention_heads: int = 4,  # 注意力头数
                 intermediate_size: int = 1024,  # 中间层维度
                 hidden_act: str = "gelu",  # 激活函数
                 hidden_dropout_prob: float = 0.1,  # 隐藏层 dropout
                 attention_probs_dropout_prob: float = 0.1,  # 注意力 dropout
                 max_position_embeddings: int = 512,  # 最大位置编码
                 initializer_range: float = 0.02):  # 初始化范围
        """
        初始化模型配置
        
        Args:
            vocab_size: 词汇表大小
            hidden_size: 隐藏层维度
            num_hidden_layers: Transformer层数
            num_attention_heads: 注意力头数
            intermediate_size: 中间层维度
            hidden_act: 激活函数
            hidden_dropout_prob: 隐藏层dropout概率
            attention_probs_dropout_prob: 注意力dropout概率
            max_position_embeddings: 最大位置编码
            initializer_range: 初始化范围
        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range


class Attention(nn.Module):
    """
    注意力机制模块
    """
    
    def __init__(self, config: Config):
        """
        初始化注意力模块
        
        Args:
            config: 模型配置
        """
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # 线性变换层
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        
        # Dropout
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
    
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """
        调整张量维度以适应注意力计算
        
        Args:
            x: 输入张量 [batch_size, seq_length, hidden_size]
            
        Returns:
            调整后的张量 [batch_size, num_heads, seq_length, head_size]
        """
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            hidden_states: 隐藏状态 [batch_size, seq_length, hidden_size]
            attention_mask: 注意力掩码 [batch_size, seq_length]
            
        Returns:
            注意力输出 [batch_size, seq_length, hidden_size]
        """
        # 线性变换
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        
        # 调整维度
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        
        # 计算注意力分数
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / torch.sqrt(torch.tensor(self.attention_head_size, dtype=torch.float32))
        
        # 应用注意力掩码
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype)
            attention_mask = (1.0 - attention_mask) * -10000.0
            attention_scores = attention_scores + attention_mask
        
        # 计算注意力权重
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        
        # 计算注意力输出
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        return context_layer


class TransformerLayer(nn.Module):
    """
    Transformer层
    """
    
    def __init__(self, config: Config):
        """
        初始化Transformer层
        
        Args:
            config: 模型配置
        """
        super().__init__()
        self.attention = Attention(config)
        self.attention_output = nn.Linear(config.hidden_size, config.hidden_size)
        self.attention_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # 前馈网络
        self.intermediate = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = nn.GELU()
        self.output = nn.Linear(config.intermediate_size, config.hidden_size)
        self.output_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.output_dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            hidden_states: 隐藏状态 [batch_size, seq_length, hidden_size]
            attention_mask: 注意力掩码 [batch_size, seq_length]
            
        Returns:
            层输出 [batch_size, seq_length, hidden_size]
        """
        # 注意力子层
        attention_output = self.attention(hidden_states, attention_mask)
        attention_output = self.attention_output(attention_output)
        attention_output = self.attention_dropout(attention_output)
        attention_output = self.attention_norm(hidden_states + attention_output)
        
        # 前馈子层
        intermediate_output = self.intermediate(attention_output)
        intermediate_output = self.intermediate_act_fn(intermediate_output)
        layer_output = self.output(intermediate_output)
        layer_output = self.output_dropout(layer_output)
        layer_output = self.output_norm(attention_output + layer_output)
        
        return layer_output


class LightweightLM(nn.Module):
    """
    轻量化语言模型
    """
    
    def __init__(self, config: Config):
        """
        初始化模型
        
        Args:
            config: 模型配置
        """
        super().__init__()
        self.config = config
        
        # 嵌入层
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(2, config.hidden_size)
        
        self.embeddings_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.embeddings_dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Transformer层
        self.transformer_layers = nn.ModuleList(
            [TransformerLayer(config) for _ in range(config.num_hidden_layers)]
        )
        
        # 输出层
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """
        初始化权重
        
        Args:
            module: 模型模块
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, 
                token_type_ids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播
        
        Args:
            input_ids: 输入ID [batch_size, seq_length]
            attention_mask: 注意力掩码 [batch_size, seq_length]
            token_type_ids: token类型ID [batch_size, seq_length]
            
        Returns:
            模型输出和隐藏状态
        """
        batch_size, seq_length = input_ids.shape
        
        # 生成位置IDs，确保不超过max_position_embeddings
        position_ids = torch.arange(min(seq_length, self.config.max_position_embeddings), 
                                   dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # 如果序列长度超过max_position_embeddings，截断或重复位置编码
        if seq_length > self.config.max_position_embeddings:
            position_ids = position_ids.repeat(1, (seq_length + self.config.max_position_embeddings - 1) // self.config.max_position_embeddings)
            position_ids = position_ids[:, :seq_length]
        
        # 嵌入层
        word_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        
        if token_type_ids is None:
            token_type_ids = torch.zeros(batch_size, seq_length, dtype=torch.long, device=input_ids.device)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        
        embeddings = word_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.embeddings_norm(embeddings)
        embeddings = self.embeddings_dropout(embeddings)
        
        # Transformer层
        hidden_states = embeddings
        for layer in self.transformer_layers:
            hidden_states = layer(hidden_states, attention_mask)
        
        # 语言模型输出
        lm_logits = self.lm_head(hidden_states)
        
        return lm_logits, hidden_states


def create_lightweight_model(config: Optional[Config] = None, vocab_size: int = 30522) -> LightweightLM:
    """
    创建轻量化语言模型
    
    Args:
        config: 模型配置，如果为None则使用默认配置
        vocab_size: 词汇表大小
        
    Returns:
        轻量化语言模型实例
    """
    if config is None:
        config = Config()
    # 更新词汇表大小
    config.vocab_size = vocab_size
    return LightweightLM(config)
