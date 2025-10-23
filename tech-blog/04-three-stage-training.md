# NanoChat深入解析(4)：三阶段训练策略详解

## 前言

在现代大语言模型训练中，多阶段渐进式训练已经成为标准做法。NanoChat采用了一个精简而有效的三阶段训练策略：基础预训练(Base Pretraining) → 中间训练(Mid Training) → 指令微调(Supervised Fine-Tuning)。这种设计既保证了模型的基础语言能力，又赋予其对话和任务执行的专长。让我们深入解析这个训练流水线的每个环节。

## 训练流程概览

```bash
# speedrun.sh 中的三阶段训练流程

# 1. 分词器训练
python -m scripts.tok_train --max_chars=2000000000

# 2. 基础预训练 (Base Pretraining)
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=20

# 3. 中间训练 (Mid Training)
torchrun --standalone --nproc_per_node=8 -m scripts.mid_train

# 4. 指令微调 (Supervised Fine-Tuning)
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft

# 5. 可选：强化学习 (Optional RL)
torchrun --standalone --nproc_per_node=8 -m scripts.chat_rl
```

## 阶段一：基础预训练 (Base Pretraining)

### 目标与定位
基础预训练的目标是让模型掌握基本的语言理解能力，包括语法、语义、世界知识等。这个阶段决定了模型的"智商"基础。

### 数据规模计算
NanoChat采用Chinchilla定律来计算训练数据量：

```bash
# d20模型配置参数
depth = 20  # 20层Transformer
model_dim = depth * 64 = 1280  # 模型维度
num_heads = 10  # 注意力头数

# 参数量估算
params ≈ 561M (约5.6亿参数)

# 根据Chinchilla定律：tokens = 20 × params
target_tokens = 561e6 × 20 = 11.2B tokens

# 考虑分词器压缩比 (4.8 chars/token)
target_chars = 11.2B × 4.8 ≈ 54B chars

# 数据分片计算 (每分片250M chars)
needed_shards = 54B / 250M ≈ 216 shards
# 向上取整到240分片，留有余量
```

### 训练配置解析

```python
# base_train.py 关键配置
depth = 20                          # 模型层数
max_seq_len = 2048                  # 最大序列长度
device_batch_size = 32              # 单设备批次大小
total_batch_size = 524288           # 总批次大小(按token计算)

# 优化器配置
embedding_lr = 0.2                  # 嵌入层学习率
unembedding_lr = 0.004              # 输出层学习率
matrix_lr = 0.02                    # Transformer矩阵学习率
weight_decay = 0.0                  # 权重衰减

# 训练终止条件
target_param_data_ratio = 20        # Chinchilla比率
```

### 梯度累积策略

```python
# 计算梯度累积步数
tokens_per_fwdbwd = device_batch_size * max_seq_len  # 32 * 2048 = 65,536
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size  # 65,536 * 8 = 524,288
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd  # 524,288 / 524,288 = 1

# 在这个配置下，不需要梯度累积，因为单次前向传播已经达到目标批次大小
```

### 训练循环实现

```python
def train_loop():
    model.train()
    for step in range(num_iterations):
        # 1. 数据加载
        with timer("data"):
            x, y = data_loader.next_batch()  # 加载下一个批次

        # 2. 前向传播
        with timer("fwd"):
            with autocast_ctx:
                loss = model(x, targets=y)
                loss = loss / grad_accum_steps

        # 3. 反向传播
        with timer("bwd"):
            loss.backward()

        # 4. 梯度更新
        if (step + 1) % grad_accum_steps == 0:
            # 梯度裁剪
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            # 优化器更新
            for optimizer in optimizers:
                optimizer.step()
                optimizer.zero_grad()

        # 5. 评估与日志
        if step % eval_every == 0:
            evaluate_model()
        if step % sample_every == 0:
            sample_from_model()
```

### 评估体系

基础预训练阶段的评估包括：

1. **验证损失**：计算Bits Per Character (BPC)
2. **CORE指标**：综合语言理解能力评估
3. **采样质量**：生成文本的质量检查

```python
def evaluate_bpb(model, data_loader, eval_tokens):
    """计算验证集的BPC (Bits Per Character)"""
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        while total_tokens < eval_tokens:
            x, y = data_loader.next_batch()
            loss = model(x, targets=y)
            total_loss += loss.item() * x.numel()
            total_tokens += x.numel()

    avg_loss = total_loss / total_tokens
    bpc = avg_loss / math.log(2)  # 转换为bits
    return bpc
```

## 阶段二：中间训练 (Mid Training)

### 目标与定位
中间训练的目标是让模型适应对话格式，学习特殊token的用法，包括：
- 对话格式的理解
- 特殊token (`<|user_start|>`, `<|assistant_end|>`等)的含义
- 工具调用格式 (Python代码块)
- 多选题格式

### 数据准备

```bash
# 下载合成身份对话数据 (2.3MB)
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl \
  https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
```

这些身份对话数据的作用是给模型赋予个性，让模型知道自己是谁。

### 训练配置

```python
# mid_train.py 配置特点
source = "base"                    # 从基础模型加载
num_epochs = 1                     # 较少的训练轮数
device_batch_size = 8              # 较小的批次 (避免OOM)
target_examples_per_step = 32      # 每步处理样本数

# 学习率配置 (更保守)
unembedding_lr = 0.001             # 降低学习率
embedding_lr = 0.05
matrix_lr = 0.005
```

### 数据格式示例

中间训练使用特殊格式化的对话数据：

```json
{
  "messages": [
    {"role": "user", "content": "Hello, who are you?"},
    {"role": "assistant", "content": "I'm NanoChat, a helpful AI assistant."}
  ]
}
```

训练时会被分词器渲染为：
```
<|bos|><|user_start|>Hello, who are you?<|user_end|><|assistant_start|>I'm NanoChat, a helpful AI assistant.<|assistant_end|>
```

### 特殊Token学习

```python
def render_conversation_with_tools(conversation):
    """
    支持工具调用的对话渲染
    """
    for message in conversation["messages"]:
        if message["role"] == "assistant":
            if isinstance(message["content"], list):
                for part in message["content"]:
                    if part["type"] == "python":
                        # 渲染Python代码块
                        tokens += tokenizer.encode("<|python_start|>")
                        tokens += tokenizer.encode(part["text"])
                        tokens += tokenizer.encode("<|python_end|>")
                    elif part["type"] == "python_output":
                        # 渲染Python输出
                        tokens += tokenizer.encode("<|output_start|>")
                        tokens += tokenizer.encode(part["text"])
                        tokens += tokenizer.encode("<|output_end|>")
```

## 阶段三：指令微调 (Supervised Fine-Tuning)

### 目标与定位
SFT阶段的目标是让模型掌握遵循指令的能力，提升在具体任务上的表现。这个阶段决定了模型的"情商"和实用性。

### 任务组合

```python
# chat_sft.py 中的任务组合
task_mixture = TaskMixture(
    tasks=[
        SmolTalk(weight=1.0),      # 对话任务
        ARC(weight=0.3),           # 科学推理
        GSM8K(weight=0.3),         # 数学推理
        CustomJSON(weight=0.1),    # 自定义任务
    ]
)
```

### 训练策略

```python
# SFT关键配置
source = "mid"                     # 从中间模型加载
device_batch_size = 4              # 更小的批次 (序列长度变化大)
num_epochs = 1                     # 完整遍历数据集
target_examples_per_step = 32      # 每步样本数

# 学习率进一步降低
unembedding_lr = 0.004             # 保持相对较高 (输出层重要)
embedding_lr = 0.2                 # 较高 (嵌入层需要适应新格式)
matrix_lr = 0.02                   # 较低 (Transformer微调)
```

### 数据加载策略

```python
def sft_data_loader():
    """SFT阶段的数据加载，支持多任务混合"""
    while True:
        # 1. 随机选择任务
        task = task_mixture.sample_task()

        # 2. 从任务中获取样本
        conversation = task.get_sample()

        # 3. 渲染为token序列
        ids, mask = tokenizer.render_conversation(conversation)

        # 4. 填充/截断到统一长度
        if len(ids) > max_seq_len:
            ids = ids[:max_seq_len]
            mask = mask[:max_seq_len]
        else:
            # 填充
            padding_len = max_seq_len - len(ids)
            ids.extend([pad_token_id] * padding_len)
            mask.extend([0] * padding_len)

        yield torch.tensor(ids), torch.tensor(mask)
```

### 损失计算

```python
def compute_sft_loss(logits, targets, mask):
    """
    SFT损失计算，只计算助手回复的损失
    """
    # logits: (B, T, vocab_size)
    # targets: (B, T)
    # mask: (B, T) - 1表示需要训练的位置

    # 重塑为二维
    B, T, V = logits.shape
    logits_flat = logits.view(B * T, V)
    targets_flat = targets.view(B * T)
    mask_flat = mask.view(B * T)

    # 只计算mask=1的位置
    active_logits = logits_flat[mask_flat == 1]
    active_targets = targets_flat[mask_flat == 1]

    # 计算交叉熵损失
    loss = F.cross_entropy(active_logits, active_targets)
    return loss
```

### 评估指标

SFT阶段评估多个维度的性能：

```python
def evaluate_chat_model():
    """评估对话模型的多个指标"""
    results = {}

    # 1. ARC-Challenge (科学推理)
    arc_results = evaluate_arc(model, tokenizer)
    results["ARC-Challenge"] = arc_results["accuracy"]

    # 2. GSM8K (数学推理)
    gsm8k_results = evaluate_gsm8k(model, tokenizer)
    results["GSM8K"] = gsm8k_results["accuracy"]

    # 3. MMLU (综合知识)
    mmlu_results = evaluate_mmlu(model, tokenizer)
    results["MMLU"] = mmlu_results["accuracy"]

    # 4. HumanEval (代码生成)
    humaneval_results = evaluate_humaneval(model, tokenizer)
    results["HumanEval"] = humaneval_results["pass@1"]

    return results
```

## 可选阶段：强化学习 (Optional RL)

### 目标与定位
RL阶段主要针对特定任务进行优化，如GSM8K数学推理。使用PPO算法进一步优化模型。

### 奖励模型设计

```python
class MathRewardModel:
    """数学推理奖励模型"""

    def compute_reward(self, question, answer, ground_truth):
        # 1. 答案正确性检查
        if self.is_correct(answer, ground_truth):
            base_reward = 1.0
        else:
            base_reward = 0.0

        # 2. 推理过程质量
        reasoning_quality = self.evaluate_reasoning_quality(answer)

        # 3. 最终奖励
        reward = base_reward + 0.1 * reasoning_quality
        return reward
```

### PPO训练循环

```python
def ppo_train_step():
    """PPO训练步骤"""
    # 1. 生成响应
    with torch.no_grad():
        responses = model.generate(questions, max_new_tokens=256)

    # 2. 计算奖励
    rewards = reward_model(questions, responses, ground_truths)

    # 3. 计算优势函数
    advantages = compute_advantages(rewards, values)

    # 4. PPO更新
    for epoch in range(ppo_epochs):
        # 4.1 计算新旧策略比率
        ratio = new_policy.log_prob(actions) / old_policy.log_prob(actions)

        # 4.2 PPO裁剪目标
        surrogate1 = ratio * advantages
        surrogate2 = torch.clamp(ratio, 1-epsilon, 1+epsilon) * advantages
        policy_loss = -torch.min(surrogate1, surrogate2).mean()

        # 4.3 价值函数损失
        value_loss = F.mse_loss(new_values, returns)

        # 4.4 熵正则化
        entropy_loss = -new_policy.entropy().mean()

        # 4.5 总损失
        loss = policy_loss + c1 * value_loss - c2 * entropy_loss

        # 4.6 反向传播
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

## 训练监控与可视化

### WandB集成

```python
# 训练过程中的监控
def log_training_metrics(step, loss, learning_rates, eval_results):
    wandb.log({
        "train/loss": loss,
        "train/lr_embedding": learning_rates[0],
        "train/lr_matrix": learning_rates[1],
        "eval/ARC": eval_results.get("ARC-Challenge", 0),
        "eval/GSM8K": eval_results.get("GSM8K", 0),
        "eval/MMLU": eval_results.get("MMLU", 0),
    }, step=step)
```

### 报告生成

```python
def generate_training_report():
    """生成训练报告"""
    report = {
        "model_config": {
            "depth": depth,
            "model_dim": model_dim,
            "num_heads": num_heads,
            "vocab_size": vocab_size,
        },
        "training_stats": {
            "total_steps": total_steps,
            "final_loss": final_loss,
            "training_time": training_time,
        },
        "evaluation_results": final_eval_results,
    }

    # 保存为markdown
    with open("report.md", "w") as f:
        f.write(format_report_as_markdown(report))
```

## 性能优化技术

### 1. 内存优化
```python
# 梯度检查点
model.gradient_checkpointing_enable()

# 混合精度训练
with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
    loss = model(inputs, targets=targets)

# 可扩展内存段
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
```

### 2. 计算优化
```python
# 编译优化 (可选)
model = torch.compile(model, dynamic=True)

# 数据加载优化
data_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
)
```

### 3. 分布式训练
```python
# 分布式数据并行
torchrun --standalone --nproc_per_node=8 -m scripts.base_train

# 梯度同步
dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
loss_tensor /= dist.get_world_size()
```

## 训练时间与成本分析

### d20模型 (5.6亿参数)
- **数据量**: 54B字符 (240个数据分片)
- **训练时间**: ~4小时 (8×H100)
- **计算成本**: ~$96 (8×$3/GPU/小时×4小时)
- **总成本**: ~$100 (包含其他开销)

### 性能对比

| 模型 | 参数量 | 训练数据 | 训练成本 | CORE分数 |
|------|--------|----------|----------|----------|
| GPT-2 (2019) | 1.5B | ~40B tokens | ~$10,000+ | 0.195 |
| NanoChat d20 | 0.56B | 11B tokens | ~$100 | 0.2219 |

## 设计权衡分析

### 1. 训练阶段划分
**优势**：
- 渐进式能力提升
- 每个阶段目标明确
- 便于调试和优化

**代价**：
- 增加训练复杂度
- 需要多个数据集
- 训练时间延长

### 2. 模型规模选择
**d20 vs d32的权衡**：
- d20: 快速训练，低成本，适合原型
- d32: 更强性能，更高成本，适合产品

### 3. 数据策略
**数据质量 vs 数据量**：
- 优先高质量数据
- 平衡不同任务类型
- 合理的数据混合比例

## 常见问题与解决方案

### 1. 内存不足 (OOM)
```python
# 解决方案
device_batch_size = 16  # 减少批次大小
max_seq_len = 1024      # 减少序列长度
model.gradient_checkpointing_enable()  # 启用梯度检查点
```

### 2. 训练不稳定
```python
# 解决方案
learning_rate = 0.001   # 降低学习率
grad_clip = 1.0        # 启用梯度裁剪
warmup_steps = 1000    # 添加学习率预热
```

### 3. 收敛缓慢
```python
# 解决方案
total_batch_size = 1048576  # 增加批次大小
num_epochs = 2              # 增加训练轮数
task_mixture_weights = {...}  # 调整任务权重
```

## 最佳实践建议

### 1. 开发阶段
- 从小模型开始 (d4, d6)
- 使用少量数据快速迭代
- 重点验证训练流程的正确性

### 2. 生产阶段
- 根据预算选择合适的模型规模
- 充分利用分布式训练
- 仔细监控训练过程

### 3. 调优阶段
- 实验不同的学习率组合
- 调整任务混合比例
- 优化数据预处理流程

## 总结

NanoChat的三阶段训练策略体现了现代LLM训练的核心思想：

1. **基础能力**：通过大规模预训练获得语言理解能力
2. **格式适应**：通过中间训练学习对话格式和特殊token
3. **任务优化**：通过指令微调掌握具体任务技能

这种渐进式训练方法既保证了模型的基础能力，又确保了在特定任务上的表现。通过精心设计的数据、合理的超参数选择和有效的优化策略，NanoChat在有限预算下实现了令人印象深刻的性能。

## 下一步

在下一篇文章中，我们将深入分析NanoChat的优化器实现，了解AdamW和Muon优化器的设计理念和性能特点。

---

**第五篇文章预告**：《NanoChat深入解析(5)：自定义优化器的性能优化》将详细解析优化器的设计原理和实现细节。