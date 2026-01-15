# ViT结构化剪枝实现指南

## 一、ViT的可剪枝结构单元

### 1. Attention Head Pruning (剪枝attention head)
最常见和有效的ViT结构化剪枝方式

### 2. MLP Neuron/Channel Pruning (剪枝MLP神经元)
类似CNN的通道剪枝

### 3. Transformer Block Pruning (剪枝整个block)
直接删除某些transformer层

### 4. Token Pruning (剪枝token)
减少序列长度

### 5. Embedding Dimension Pruning (降低维度)
减少hidden dimension

---

## 二、Attention Head Pruning 详细实现

### ViT Attention结构回顾

```
输入: x [B, N, D]  (batch, num_tokens, embed_dim)

QKV生成:
qkv = Linear(x)  → [B, N, 3*D]
q, k, v = split(qkv)  → 各 [B, N, D]

多头分割:
q = reshape(q, [B, N, num_heads, head_dim])
k = reshape(k, [B, N, num_heads, head_dim])
v = reshape(v, [B, N, num_heads, head_dim])

Attention计算:
attn = softmax(q @ k.T / sqrt(head_dim))  [B, num_heads, N, N]
output = attn @ v  [B, num_heads, N, head_dim]

输出投影:
output = reshape(output, [B, N, D])
output = Linear(output)  [B, N, D]
```

### Head Pruning策略

#### 方法1: 重要性评估 + 直接删除

```python
def compute_head_importance(model, dataloader):
    """
    计算每个attention head的重要性
    """
    head_importance = {}
    
    for batch in dataloader:
        outputs = model(batch, output_attentions=True)
        attentions = outputs.attentions  # 每层的attention weights
        
        for layer_idx, attn in enumerate(attentions):
            # attn shape: [B, num_heads, N, N]
            
            # 方法1: 基于attention权重的方差
            importance = attn.var(dim=[0, 2, 3])  # [num_heads]
            
            # 方法2: 基于梯度
            loss = outputs.loss
            loss.backward()
            grad = attn.grad.abs().mean(dim=[0, 2, 3])  # [num_heads]
            
            # 方法3: 基于Taylor展开
            importance = (attn * attn.grad).abs().mean(dim=[0, 2, 3])
            
            if layer_idx not in head_importance:
                head_importance[layer_idx] = []
            head_importance[layer_idx].append(importance)
    
    # 平均所有batch
    for layer_idx in head_importance:
        head_importance[layer_idx] = torch.stack(
            head_importance[layer_idx]
        ).mean(0)
    
    return head_importance

def prune_heads(model, head_importance, prune_ratio=0.5):
    """
    根据重要性剪枝head
    """
    for layer_idx, block in enumerate(model.blocks):
        importance = head_importance[layer_idx]
        num_heads = len(importance)
        num_to_prune = int(num_heads * prune_ratio)
        
        # 选择重要性最低的heads
        _, indices = importance.sort()
        heads_to_prune = indices[:num_to_prune].tolist()
        
        # 剪枝这些heads
        prune_attention_heads(block.attn, heads_to_prune)
```

#### 方法2: Mask-based Soft Pruning

```python
class MaskedMultiheadAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        
        # Head mask (可学习的或固定的)
        self.head_mask = nn.Parameter(
            torch.ones(num_heads), 
            requires_grad=True
        )
    
    def forward(self, x):
        B, N, C = x.shape
        
        # QKV生成
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # [B, num_heads, N, head_dim]
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = attn.softmax(dim=-1)
        
        # 应用head mask
        attn = attn * self.head_mask.view(1, -1, 1, 1)
        
        # 输出
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        
        return x
    
    def get_active_heads(self, threshold=0.1):
        """返回激活的head索引"""
        return (self.head_mask > threshold).nonzero().squeeze()
    
    def prune_heads(self, threshold=0.1):
        """永久删除不重要的heads"""
        active_heads = self.get_active_heads(threshold)
        # 重构qkv和proj矩阵，只保留active heads
        # ... (具体实现见下文)
```

---

## 三、具体实现：Hard Head Pruning

### 核心思路
删除不重要的head，重构QKV矩阵和投影矩阵

### 关键步骤

```python
def prune_attention_heads_hard(attn_module, heads_to_prune):
    """
    硬剪枝：物理删除attention heads
    
    Args:
        attn_module: MultiheadAttention模块
        heads_to_prune: 要删除的head索引列表 [0, 2, 5]
    """
    num_heads = attn_module.num_heads
    head_dim = attn_module.head_dim
    embed_dim = num_heads * head_dim
    
    # 1. 确定要保留的heads
    all_heads = set(range(num_heads))
    heads_to_keep = sorted(list(all_heads - set(heads_to_prune)))
    
    if len(heads_to_keep) == 0:
        raise ValueError("Cannot prune all heads!")
    
    # 2. 提取qkv权重和偏置
    qkv_weight = attn_module.qkv.weight.data  # [3*embed_dim, embed_dim]
    qkv_bias = attn_module.qkv.bias.data      # [3*embed_dim]
    
    # 重塑为 [3, num_heads, head_dim, embed_dim]
    qkv_weight = qkv_weight.view(3, num_heads, head_dim, embed_dim)
    qkv_bias = qkv_bias.view(3, num_heads, head_dim)
    
    # 3. 选择要保留的heads
    qkv_weight = qkv_weight[:, heads_to_keep, :, :]
    qkv_bias = qkv_bias[:, heads_to_keep, :]
    
    # 4. 重塑回线性层格式
    new_num_heads = len(heads_to_keep)
    new_embed_dim = new_num_heads * head_dim
    
    qkv_weight = qkv_weight.reshape(3 * new_embed_dim, embed_dim)
    qkv_bias = qkv_bias.reshape(3 * new_embed_dim)
    
    # 5. 创建新的qkv层
    new_qkv = nn.Linear(embed_dim, 3 * new_embed_dim, bias=True)
    new_qkv.weight.data = qkv_weight
    new_qkv.bias.data = qkv_bias
    attn_module.qkv = new_qkv
    
    # 6. 处理输出投影层
    proj_weight = attn_module.proj.weight.data  # [embed_dim, embed_dim]
    proj_bias = attn_module.proj.bias.data      # [embed_dim]
    
    # 重塑为 [embed_dim, num_heads, head_dim]
    proj_weight = proj_weight.view(embed_dim, num_heads, head_dim)
    
    # 选择要保留的heads
    proj_weight = proj_weight[:, heads_to_keep, :].reshape(
        embed_dim, new_embed_dim
    )
    
    # 创建新的投影层
    new_proj = nn.Linear(new_embed_dim, embed_dim, bias=True)
    new_proj.weight.data = proj_weight
    new_proj.bias.data = proj_bias
    attn_module.proj = new_proj
    
    # 7. 更新模块属性
    attn_module.num_heads = new_num_heads
    
    print(f"Pruned {len(heads_to_prune)} heads: {heads_to_prune}")
    print(f"New configuration: {new_num_heads} heads")
```

---

## 四、MLP Neuron Pruning

### 实现方法

```python
def prune_mlp_neurons(mlp_module, neuron_importance, prune_ratio=0.5):
    """
    剪枝MLP中的神经元（类似CNN通道剪枝）
    
    MLP结构:
    x → fc1 [dim, hidden_dim] → GELU → fc2 [hidden_dim, dim] → x
    """
    hidden_dim = mlp_module.fc1.out_features
    num_to_prune = int(hidden_dim * prune_ratio)
    
    # 选择要保留的神经元
    _, indices = neuron_importance.sort(descending=True)
    neurons_to_keep = indices[num_to_prune:].sort()[0]
    
    # 1. 剪枝fc1的输出维度
    fc1_weight = mlp_module.fc1.weight.data[neurons_to_keep, :]
    fc1_bias = mlp_module.fc1.bias.data[neurons_to_keep]
    
    new_hidden_dim = len(neurons_to_keep)
    new_fc1 = nn.Linear(mlp_module.fc1.in_features, new_hidden_dim)
    new_fc1.weight.data = fc1_weight
    new_fc1.bias.data = fc1_bias
    
    # 2. 剪枝fc2的输入维度
    fc2_weight = mlp_module.fc2.weight.data[:, neurons_to_keep]
    fc2_bias = mlp_module.fc2.bias.data
    
    new_fc2 = nn.Linear(new_hidden_dim, mlp_module.fc2.out_features)
    new_fc2.weight.data = fc2_weight
    new_fc2.bias.data = fc2_bias
    
    # 3. 替换
    mlp_module.fc1 = new_fc1
    mlp_module.fc2 = new_fc2
    
    print(f"MLP pruned: {hidden_dim} → {new_hidden_dim} neurons")

def compute_mlp_neuron_importance(mlp_module, dataloader):
    """计算MLP神经元重要性"""
    importance = torch.zeros(mlp_module.fc1.out_features)
    
    for batch in dataloader:
        # 前向传播
        x = batch
        hidden = mlp_module.fc1(x)  # [B, N, hidden_dim]
        
        # 方法1: 基于激活值的L1范数
        imp = hidden.abs().mean(dim=[0, 1])  # [hidden_dim]
        
        # 方法2: 基于梯度
        loss.backward()
        grad = mlp_module.fc1.weight.grad.abs().sum(dim=1)  # [hidden_dim]
        
        importance += imp
    
    return importance / len(dataloader)
```

---

## 五、完整的ViT结构化剪枝Pipeline

```python
class StructuredViTPruner:
    def __init__(self, model, dataloader):
        self.model = model
        self.dataloader = dataloader
        self.head_importance = {}
        self.mlp_importance = {}
    
    def compute_importance(self):
        """步骤1: 计算所有结构单元的重要性"""
        print("Computing importance scores...")
        
        for layer_idx, block in enumerate(self.model.blocks):
            # Attention heads
            head_imp = self._compute_head_importance(
                block.attn, layer_idx
            )
            self.head_importance[layer_idx] = head_imp
            
            # MLP neurons
            mlp_imp = self._compute_mlp_importance(
                block.mlp, layer_idx
            )
            self.mlp_importance[layer_idx] = mlp_imp
    
    def prune(self, head_prune_ratio=0.3, mlp_prune_ratio=0.3):
        """步骤2: 执行剪枝"""
        print(f"Pruning {head_prune_ratio*100}% heads and "
              f"{mlp_prune_ratio*100}% MLP neurons...")
        
        for layer_idx, block in enumerate(self.model.blocks):
            # 剪枝heads
            head_imp = self.head_importance[layer_idx]
            num_heads = len(head_imp)
            num_to_prune = int(num_heads * head_prune_ratio)
            
            _, indices = head_imp.sort()
            heads_to_prune = indices[:num_to_prune].tolist()
            
            prune_attention_heads_hard(block.attn, heads_to_prune)
            
            # 剪枝MLP neurons
            mlp_imp = self.mlp_importance[layer_idx]
            prune_mlp_neurons(block.mlp, mlp_imp, mlp_prune_ratio)
    
    def fine_tune(self, epochs=10):
        """步骤3: 微调剪枝后的模型"""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        
        for epoch in range(epochs):
            for batch in self.dataloader:
                loss = self.model(batch).loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

# 使用示例
pruner = StructuredViTPruner(model, train_loader)
pruner.compute_importance()
pruner.prune(head_prune_ratio=0.5, mlp_prune_ratio=0.3)
pruner.fine_tune(epochs=10)
```

---

## 六、与当前实现的对比

### 当前实现 (非结构化)
```python
# vit_pruning_utils.py
prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=px,
)
```
- ✅ 简单易实现
- ❌ 0散布在矩阵中
- ❌ 无法硬件加速
- ❌ 不减少计算量

### 结构化剪枝 (提议的实现)
```python
# Head-level pruning
prune_attention_heads_hard(attn, heads_to_prune)
prune_mlp_neurons(mlp, neurons_to_prune)
```
- ✅ 真正减少参数量
- ✅ 减少计算量
- ✅ 硬件友好
- ✅ 可实际部署
- ⚠️ 实现复杂度高
- ⚠️ 可能影响精度

---

## 七、实验建议

### 剪枝率设置
- Head pruning: 30-50% (通常可以容忍)
- MLP pruning: 20-40%
- Block pruning: 10-20% (最激进)

### 重要性度量对比
1. **Gradient-based**: 计算成本高，但准确
2. **Activation-based**: 快速，但可能不够准确
3. **Taylor expansion**: 折中方案

### 迭代剪枝 vs 一次剪枝
- 迭代剪枝: 每次剪5-10%，重复多轮
- 一次剪枝: 直接剪到目标，需要更多微调

