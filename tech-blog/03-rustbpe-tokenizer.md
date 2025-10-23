# NanoChat深入解析(3)：Rust实现的BPE分词器

## 前言

在NanoChat项目中，分词器的实现是一个特别有趣的技术选择。虽然有多种现成的分词器解决方案，但NanoChat选择了用Rust从零实现BPE(Byte Pair Encoding)算法，然后用Python封装。这个选择背后有什么考量？让我们深入解析这个高性能分词器的设计与实现。

## 分词器的两种实现方案

NanoChat提供了两种分词器实现：

### 1. HuggingFace Tokenizer方案
```python
class HuggingFaceTokenizer:
    """基于HuggingFace Tokenizer的轻量包装"""

    @classmethod
    def train_from_iterator(cls, text_iterator, vocab_size):
        tokenizer = HFTokenizer(BPE(
            byte_fallback=True,
            unk_token=None,
            fuse_unk=False,
        ))
        # 配置GPT-4风格的预分词器
        gpt4_split_regex = Regex(SPLIT_PATTERN)
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(pattern=gpt4_split_regex, behavior="isolated", invert=False),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
        ])
        # 训练逻辑...
```

### 2. RustBPE + Tiktoken方案（NanoChat的选择）
```python
class RustBPETokenizer:
    """基于rustbpe训练 + tiktoken推理的高效分词器"""

    @classmethod
    def train_from_iterator(cls, text_iterator, vocab_size):
        # 1) 使用rustbpe进行训练
        tokenizer = rustbpe.Tokenizer()
        vocab_size_no_special = vocab_size - len(SPECIAL_TOKENS)
        tokenizer.train_from_iterator(text_iterator, vocab_size_no_special, pattern=SPLIT_PATTERN)

        # 2) 构建tiktoken编码器用于推理
        pattern = tokenizer.get_pattern()
        mergeable_ranks_list = tokenizer.get_mergeable_ranks()
        mergeable_ranks = {bytes(k): v for k, v in mergeable_ranks_list}
        tokens_offset = len(mergeable_ranks)
        special_tokens = {name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)}

        enc = tiktoken.Encoding(
            name="rustbpe",
            pat_str=pattern,
            mergeable_ranks=mergeable_ranks,
            special_tokens=special_tokens,
        )
        return cls(enc, "<|bos|>")
```

## 为什么选择Rust实现？

### 性能优势
- **训练速度**：Rust的零成本抽象和内存安全特性，使得BPE训练速度比Python快数倍
- **内存效率**：精确的内存控制，避免Python的GIL限制
- **并行处理**：内置的并发安全，充分利用多核CPU

### 工程考量
- **依赖最小化**：避免引入庞大的HuggingFace依赖
- **可控性**：完全掌控算法实现细节，便于定制优化
- **学习价值**：从零实现有助于深入理解BPE算法

## BPE算法核心原理

### 算法流程
BPE算法是一种数据压缩算法，被广泛应用于自然语言处理中：

1. **初始化**：将文本拆分为字符级别的词汇表
2. **统计频率**：计算所有相邻字符对的出现频率
3. **合并操作**：选择频率最高的字符对，合并为新的token
4. **迭代更新**：重复步骤2-3，直到达到目标词汇表大小

### Python伪代码示例
```python
def bpe_train(corpus, vocab_size):
    # 1. 初始化：字符级词汇表
    vocab = set(''.join(corpus))
    vocab = list(vocab)

    # 2. 统计字符对频率
    def get_pair_frequency(words):
        pairs = {}
        for word in words:
            for i in range(len(word) - 1):
                pair = (word[i], word[i+1])
                pairs[pair] = pairs.get(pair, 0) + 1
        return pairs

    # 3. 迭代合并
    while len(vocab) < vocab_size:
        pairs = get_pair_frequency(words)
        if not pairs:
            break

        # 选择最高频的字符对
        best_pair = max(pairs, key=pairs.get)

        # 合并操作
        merge_words(words, best_pair)

        # 添加到词汇表
        vocab.append(''.join(best_pair))

    return vocab, merges
```

## Rust实现的技术细节

### 核心数据结构

```rust
// 词表示
#[derive(Clone, Debug)]
struct Word {
    ids: Vec<u32>,
}

impl Word {
    // 获取所有相邻token对
    #[inline]
    fn pairs<'a>(&'a self) -> impl Iterator<Item = Pair> + 'a {
        self.ids.windows(2).map(|w| (w[0], w[1]))
    }

    // 合并指定的token对
    fn merge_pair(&mut self, pair: Pair, new_id: u32) -> Vec<(Pair, i32)> {
        let (a, b) = pair;
        let n = self.ids.len();
        if n < 2 {
            return Vec::new();
        }

        let mut out: Vec<u32> = Vec::with_capacity(n);
        let mut deltas: Vec<(Pair, i32)> = Vec::with_capacity(6);

        let mut i = 0;
        while i < n {
            if i + 1 < n && self.ids[i] == a && self.ids[i + 1] == b {
                // 处理合并的边界效应
                let left = out.last().copied();
                let right = if i + 2 < n { Some(self.ids[i + 2]) } else { None };

                // 记录旧对子的移除
                if let Some(x) = left {
                    deltas.push(((x, a), -1));
                    deltas.push(((x, new_id), 1));
                }
                deltas.push(((a, b), -1));
                if let Some(y) = right {
                    deltas.push(((b, y), -1));
                    deltas.push(((new_id, y), 1));
                }

                // 写入合并后的token
                out.push(new_id);
                i += 2; // 跳过已合并的两个token
            } else {
                out.push(self.ids[i]);
                i += 1;
            }
        }

        self.ids = out;
        deltas
    }
}
```

### 高效的并行处理

```rust
// 并行统计token对频率
#[inline]
fn count_pairs_parallel(
    words: &[Word],
    counts: &[i32],
) -> (AHashMap<Pair, i32>, AHashMap<Pair, AHashSet<usize>>) {
    words
        .par_iter()  // 并行迭代
        .enumerate()
        .map(|(i, w)| {
            let mut local_pc: AHashMap<Pair, i32> = AHashMap::new();
            let mut local_wtu: AHashMap<Pair, AHashSet<usize>> = AHashMap::new();
            if w.ids.len() >= 2 && counts[i] != 0 {
                for (a, b) in w.pairs() {
                    *local_pc.entry((a, b)).or_default() += counts[i];
                    local_wtu.entry((a, b)).or_default().insert(i);
                }
            }
            (local_pc, local_wtu)
        })
        .reduce(
            || (AHashMap::new(), AHashMap::new()),
            |(mut acc_pc, mut acc_wtu), (pc, wtu)| {
                // 合并并行结果
                for (k, v) in pc {
                    *acc_pc.entry(k).or_default() += v;
                }
                for (k, s) in wtu {
                    acc_wtu.entry(k).or_default().extend(s);
                }
                (acc_pc, acc_wtu)
            }
        )
}
```

### 优先队列管理

```rust
#[derive(Debug, Eq)]
struct MergeJob {
    pair: Pair,
    count: u64,
    pos: AHashSet<usize>,  // 该token对可能出现的词位置
}

impl Ord for MergeJob {
    fn cmp(&self, other: &Self) -> Ordering {
        // 按频率最大堆排序，频率相同时按token对字典序
        if self.count != other.count {
            self.count.cmp(&other.count)
        } else {
            other.pair.cmp(&self.pair)  // 升序排列确保确定性
        }
    }
}
```

## Python封装层设计

### PyO3集成
Rust代码通过PyO3库暴露给Python：

```rust
use pyo3::prelude::*;

#[pyclass]
pub struct Tokenizer {
    pub merges: StdHashMap<Pair, u32>,
    pub pattern: String,
    compiled_pattern: Regex,
}

#[pymethods]
impl Tokenizer {
    #[new]
    fn new() -> Self {
        Self {
            merges: StdHashMap::new(),
            pattern: GPT4_PATTERN.to_string(),
            compiled_pattern: Regex::new(GPT4_PATTERN).unwrap(),
        }
    }

    fn train_from_iterator(
        &mut self,
        text_iterator: &PyAny,
        vocab_size: u32,
        pattern: String,
    ) -> PyResult<()> {
        // 实现训练逻辑...
        Ok(())
    }

    fn get_pattern(&self) -> PyResult<String> {
        Ok(self.pattern.clone())
    }

    fn get_mergeable_ranks(&self) -> PyResult<Vec<(Vec<u8>, u32)>> {
        // 返回可合并的token等级
        Ok(self.merges.iter()
            .map(|(&(a, b), &new_id)| {
                let mut token = self.decode_token(a);
                token.extend(self.decode_token(b));
                (token, new_id)
            })
            .collect())
    }
}
```

## GPT-4风格的正则表达式分词

### 分词模式
```python
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
```

这个正则表达式体现了GPT-4的分词策略：

1. **缩写和词缀**：`'(?i:[sdmt]|ll|ve|re)` - 处理英文缩写
2. **字母序列**：`[^\r\n\p{L}\p{N}]?+\p{L}+` - 字母词元
3. **数字序列**：`\p{N}{1,2}` - 1-2位数字（NanoChat的优化）
4. **非字母数字**：` ?[^\s\p{L}\p{N}]++[\r\n]*` - 标点符号
5. **换行处理**：`\s*[\r\n]` - 换行符
6. **空格处理**：`\s+(?!\S)|\s+` - 各种空格情况

### 设计考量
NanoChat将数字匹配从`\p{N}{1,3}`改为`\p{N}{1,2}`，这是一个有趣的优化选择：

```python
# NOTE: 这个选择偏离了GPT-4，使用\p{N}{1,2}而不是\p{N}{1,3}
# 我这样做是因为不想在小词汇表上"浪费"太多token给数字
# 我还没有验证这是否真的是个好主意，TODO
```

**原因分析**：
- **词汇表效率**：小模型下，数字token占用过多空间
- **泛化能力**：2位数字覆盖了大部分常见数字场景
- **压缩效果**：长数字序列可以分解为多个2位数字的组合

## 特殊Token处理

### 对话专用Token
```python
SPECIAL_TOKENS = [
    "<|bos|>",           # 文档开始分隔符
    "<|user_start|>",    # 用户消息开始
    "<|user_end|>",      # 用户消息结束
    "<|assistant_start|>", # 助手消息开始
    "<|assistant_end|>",   # 助手消息结束
    "<|python_start|>",    # Python代码开始
    "<|python_end|>",      # Python代码结束
    "<|output_start|>",    # Python输出开始
    "<|output_end|>",      # Python输出结束
]
```

这些特殊Token支持复杂的对话格式化，包括工具调用场景。

### 对话渲染机制
```python
def render_conversation(self, conversation, max_tokens=2048):
    """
    将对话渲染为token序列
    返回：
    - ids: token id列表
    - mask: 掩码列表，1表示需要训练的助手token
    """
    ids, mask = [], []

    # 处理系统消息（如果有）
    if conversation["messages"][0]["role"] == "system":
        conversation = copy.deepcopy(conversation)
        messages = conversation["messages"]
        messages[1]["content"] = messages[0]["content"] + "\n\n" + messages[1]["content"]
        messages = messages[1:]
    else:
        messages = conversation["messages"]

    # 添加BOS token
    add_tokens(self.get_bos_token_id(), 0)

    # 渲染对话
    for i, message in enumerate(messages):
        if message["role"] == "user":
            value_ids = self.encode(content)
            add_tokens(user_start, 0)
            add_tokens(value_ids, 0)
            add_tokens(user_end, 0)
        elif message["role"] == "assistant":
            add_tokens(assistant_start, 0)
            # 处理助手消息（可能包含工具调用）
            if isinstance(content, str):
                value_ids = self.encode(content)
                add_tokens(value_ids, 1)  # 助手回复需要训练
            elif isinstance(content, list):
                for part in content:
                    if part["type"] == "text":
                        value_ids = self.encode(part["text"])
                        add_tokens(value_ids, 1)
                    elif part["type"] == "python":
                        add_tokens(python_start, 1)
                        value_ids = self.encode(part["text"])
                        add_tokens(value_ids, 1)
                        add_tokens(python_end, 1)
                    elif part["type"] == "python_output":
                        # Python输出不需要训练
                        add_tokens(output_start, 0)
                        value_ids = self.encode(part["text"])
                        add_tokens(value_ids, 0)
                        add_tokens(output_end, 0)
            add_tokens(assistant_end, 1)

    return ids[:max_tokens], mask[:max_tokens]
```

## 性能优化策略

### 1. 缓存机制
```python
@lru_cache(maxsize=32)
def encode_special(self, text):
    return self.enc.encode_single_token(text)
```

### 2. 批处理优化
```python
def encode(self, text, prepend=None, append=None, num_threads=8):
    if isinstance(text, list):
        # 批量编码，支持多线程
        ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
        # 批量添加特殊token
        if prepend is not None:
            for ids_row in ids:
                ids_row.insert(0, prepend_id)
```

### 3. 内存管理
- **Rust零拷贝**：避免不必要的内存分配
- **预分配容量**：根据预估大小预分配内存
- **紧凑字符串**：使用`CompactString`减少小字符串的内存占用

## 训练流程分析

### 1. 数据准备
```python
def train_tokenizer(vocab_size=50304):
    from nanochat.dataset import create_data_iterator
    from nanochat.tokenizer import RustBPETokenizer

    # 创建数据迭代器
    text_iterator = create_data_iterator(num_shards=100)

    # 训练分词器
    tokenizer = RustBPETokenizer.train_from_iterator(text_iterator, vocab_size)

    # 保存分词器
    tokenizer.save("tokenizer")
```

### 2. 训练监控
```rust
// 在Rust实现中添加进度监控
fn train_from_iterator(&mut self, text_iterator: &PyAny, vocab_size: u32, pattern: String) -> PyResult<()> {
    let mut current_vocab_size = 256; // 初始词汇表（字节级别）

    while current_vocab_size < vocab_size {
        // 统计token对频率
        let (pair_counts, word_to_update) = count_pairs_parallel(&words, &counts);

        if pair_counts.is_empty() {
            break;
        }

        // 选择最高频的token对
        let best_pair = pair_counts.iter()
            .max_by_key(|(_, &count)| count)
            .map(|(&pair, _)| pair)
            .unwrap();

        // 执行合并
        let new_token_id = current_vocab_size;
        for &word_idx in &word_to_update[&best_pair] {
            let deltas = words[word_idx].merge_pair(best_pair, new_token_id);
            // 更新频率统计...
        }

        self.merges.insert(best_pair, new_token_id);
        current_vocab_size += 1;

        // 输出进度
        if current_vocab_size % 1000 == 0 {
            println!("Trained vocab size: {}", current_vocab_size);
        }
    }

    Ok(())
}
```

## 技术权衡分析

### 1. Rust vs Python实现

| 方面 | Rust实现 | Python实现 |
|------|----------|------------|
| 训练速度 | 快（5-10倍） | 慢 |
| 内存效率 | 高 | 中等 |
| 开发复杂度 | 高 | 低 |
| 调试便利性 | 中等 | 高 |
| 依赖管理 | 简单 | 复杂 |

### 2. 自研 vs HuggingFace

**自研优势**：
- 完全控制算法细节
- 最小化依赖
- 学习价值高
- 便于定制优化

**HuggingFace优势**：
- 成熟稳定
- 功能丰富
- 社区支持
- 标准兼容

### 3. 性能优化的得与失

**得**：
- 训练速度显著提升
- 内存使用更加高效
- 支持大规模数据训练

**失**：
- 增加了编译复杂度
- 需要Rust开发环境
- 调试门槛较高

## 使用示例

### 训练自定义分词器
```python
from nanochat.tokenizer import RustBPETokenizer

# 准备训练数据
def text_data():
    with open("large_corpus.txt", "r") as f:
        for line in f:
            yield line.strip()

# 训练分词器
tokenizer = RustBPETokenizer.train_from_iterator(
    text_data(),
    vocab_size=50304
)

# 保存分词器
tokenizer.save("my_tokenizer")
```

### 推理使用
```python
# 加载分词器
tokenizer = RustBPETokenizer.from_directory("my_tokenizer")

# 编码文本
text = "Hello, world! How are you?"
tokens = tokenizer.encode(text)
print(f"Tokens: {tokens}")
print(f"Decoded: {tokenizer.decode(tokens)}")

# 编码对话
conversation = {
    "messages": [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."}
    ]
}
ids, mask = tokenizer.render_conversation(conversation)
```

## 调试与可视化

```python
def visualize_tokenization(self, ids, mask):
    """可视化tokenization结果，便于调试"""
    RED = '\033[91m'
    GREEN = '\033[92m'
    RESET = '\033[0m'
    tokens = []
    for i, (token_id, mask_val) in enumerate(zip(ids, mask)):
        token_str = self.decode([token_id])
        color = GREEN if mask_val == 1 else RED
        tokens.append(f"{color}{token_str}{RESET}")
    return '|'.join(tokens)

# 使用示例
conversation = {...}
ids, mask = tokenizer.render_conversation(conversation)
visualization = tokenizer.visualize_tokenization(ids, mask)
print(visualization)
```

## 总结

NanoChat的RustBPE分词器体现了以下工程哲学：

1. **性能优先**：选择Rust实现关键的训练算法
2. **实用主义**：推理时使用成熟的tiktoken库
3. **可维护性**：清晰的Python封装层
4. **扩展性**：支持复杂的对话格式化需求

这种设计在保持高性能的同时，也确保了代码的可读性和可维护性。对于需要在有限资源下训练自定义分词器的场景，这是一个很好的参考实现。

## 下一步

在下一篇文章中，我们将深入分析NanoChat的三阶段训练流水线，了解如何从基础预训练到指令微调的完整过程。

---

**第四篇文章预告**：《NanoChat深入解析(4)：三阶段训练策略详解》将详细解析预训练、中间训练和指令微调的每个环节。