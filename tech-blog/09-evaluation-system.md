# NanoChat深入解析(9)：全方位模型评估体系

## 前言

在大语言模型开发中，全面的评估体系是衡量模型性能和指导改进方向的关键。NanoChat实现了多层次的评估框架，包括基础语言能力评估、任务特定评估、对话质量评估等。本文将深入解析NanoChat的评估体系设计，了解如何科学地评估模型表现。

## 评估体系架构

NanoChat的评估体系分为多个层次：

```
基础评估 ← 任务评估 ← 对话评估 ← 生产评估
    ↓         ↓         ↓         ↓
  CORE     ARC/MMLU   ChatCORE   用户体验
  BPC      GSM8K      采样质量   响应时间
  困惑度    HumanEval  工具使用   资源消耗
```

### 评估类型
1. **基础评估**：语言模型基础能力
2. **任务评估**：特定任务的性能表现
3. **对话评估**：聊天交互的质量评估
4. **系统评估**：整体性能和资源使用

## CORE基准评估

### CORE指标简介

CORE (Comprehensive Overall Reading Evaluation) 是DCLM论文中提出的综合评估指标，用于衡量语言模型的整体理解能力。

### 提示词模板系统

```python
from jinja2 import Template

def render_prompts_mc(item, continuation_delimiter, fewshot_examples=None):
    """
    为多选题渲染完整提示词
    """
    template_str = """
{%- for example in fewshot_examples -%}
{{ example.query }}{{ continuation_delimiter }}{{ example.choices[example.gold] }}

{% endfor -%}
{{ item.query }}{{ continuation_delimiter }}{{ choice }}""".strip()

    template = Template(template_str)
    fewshot_examples = fewshot_examples or []
    context = {
        'fewshot_examples': fewshot_examples,
        'continuation_delimiter': continuation_delimiter,
        'item': item
    }

    # 为每个选项生成提示词
    prompts = [
        template.render(choice=choice, **context)
        for choice in item['choices']
    ]
    return prompts

def render_prompts_schema(item, continuation_delimiter, fewshot_examples=None):
    """
    为schema问题渲染提示词
    """
    template_str = """
{%- for example in fewshot_examples -%}
{{ example.context_options[example.gold] }}{{ continuation_delimiter }}{{ example.continuation }}

{% endfor -%}
{{ context }}{{ continuation_delimiter }}{{ item.continuation }}""".strip()

    template = Template(template_str)
    context = {
        'fewshot_examples': fewshot_examples,
        'continuation_delimiter': continuation_delimiter,
        'item': item
    }

    prompts = [
        template.render(context=context_option, **context)
        for context_option in item['context_options']
    ]
    return prompts

def render_prompts_lm(item, continuation_delimiter, fewshot_examples=None):
    """
    为语言建模任务渲染提示词
    注意：在模板中手动处理上下文，去除尾部空白字符
    """
    template_str = """
{%- for example in fewshot_examples -%}
{{ example.context | trim }}{{ continuation_delimiter }}{{ example.continuation }}

{% endfor -%}
{{ item.context | trim }}{{ continuation_delimiter }}{% if include_continuation %}{{ item.continuation }}{% endif %}""".strip()

    template = Template(template_str)
    context = {
        'fewshot_examples': fewshot_examples,
        'continuation_delimiter': continuation_delimiter,
        'item': item
    }

    # 返回两个提示词：不包含和包含延续内容
    prompt_without = template.render(include_continuation=False, **context)
    prompt_with = template.render(include_continuation=True, **context)

    # 去除尾部空白字符
    prompt_without = prompt_without.strip()
    return [prompt_without, prompt_with]
```

### 通用长度查找算法

```python
def find_common_length(token_sequences, direction='left'):
    """
    在token序列中查找公共前缀或后缀的长度
    - direction: 'left' 表示前缀，'right' 表示后缀
    """
    min_len = min(len(seq) for seq in token_sequences)
    indices = {
        'left': range(min_len),
        'right': range(-1, -min_len-1, -1)
    }[direction]

    # 查找第一个不同位置
    for i, idx in enumerate(indices):
        token = token_sequences[0][idx]
        if not all(seq[idx] == token for seq in token_sequences):
            return i

    return min_len  # 所有token都相同
```

### 任务评估执行

```python
def evaluate_task(model, tokenizer, data, device, task_meta):
    """
    评估单个任务的性能
    """
    task_type = task_meta['task_type']
    num_fewshot = task_meta['num_fewshot']
    continuation_delimiter = task_meta.get('continuation_delimiter', ' ')

    correct = 0
    total = 0

    for item in data:
        # 准备少样本示例
        fewshot_examples = item.get('fewshot_examples', [])[:num_fewshot]

        if task_type == 'multiple_choice':
            # 多选题评估
            prompts = render_prompts_mc(item, continuation_delimiter, fewshot_examples)
            scores = []

            for prompt in prompts:
                # 计算每个选项的对数概率
                tokens = tokenizer.encode(prompt)
                input_ids = torch.tensor([tokens], dtype=torch.long, device=device)

                with torch.no_grad():
                    logits = model(input_ids)
                    log_probs = F.log_softmax(logits[:, -1, :], dim=-1)

                    # 计算选项部分的对数似然
                    choice_tokens = tokenizer.encode(item['choice'])  # 需要适配
                    choice_logprob = 0
                    for i, token in enumerate(choice_tokens):
                        if i == 0:
                            choice_logprob += log_probs[0, token].item()
                        else:
                            # 后续token需要重新计算
                            # 这里简化处理
                            pass

                    scores.append(choice_logprob)

            # 选择得分最高的选项
            predicted = scores.index(max(scores))
            if predicted == item['gold']:
                correct += 1

        elif task_type == 'schema':
            # Schema任务评估
            prompts = render_prompts_schema(item, continuation_delimiter, fewshot_examples)
            # 类似多选题的处理逻辑

        elif task_type == 'language_modeling':
            # 语言建模任务评估
            prompts = render_prompts_lm(item, continuation_delimiter, fewshot_examples)
            prompt_without, prompt_with = prompts

            # 计算延续部分的对数似然
            # 实现细节...

        total += 1

    return correct / total if total > 0 else 0.0
```

## 基础模型评估框架

### 评估流程设计

```python
def evaluate_model(model, tokenizer, device, max_per_task=-1):
    """
    评估基础模型在CORE基准上的表现
    - max_per_task: 每个任务的最大示例数（-1表示不限制）
    """
    # 加载配置和任务元数据
    base_dir = get_base_dir()
    eval_bundle_dir = os.path.join(base_dir, "eval_bundle")
    config_path = os.path.join(eval_bundle_dir, "core.yaml")
    data_base_path = os.path.join(eval_bundle_dir, "eval_data")
    eval_meta_data = os.path.join(eval_bundle_dir, "eval_meta_data.csv")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    tasks = config['icl_tasks']
    eval_metadata = pd.read_csv(eval_meta_data)

    # 评估每个任务
    results = {}
    centered_results = {}

    for task in tasks:
        start_time = time.time()
        label = task['label']

        task_meta = {
            'task_type': task['icl_task_type'],
            'dataset_uri': task['dataset_uri'],
            'num_fewshot': task['num_fewshot'][0],
            'continuation_delimiter': task.get('continuation_delimiter', ' ')
        }

        print0(f"Evaluating: {label} ({task_meta['num_fewshot']}-shot, type: {task_meta['task_type']})... ", end='')

        # 加载任务数据
        data_path = os.path.join(data_base_path, task_meta['dataset_uri'])
        with open(data_path, 'r') as f:
            data = [json.loads(line.strip()) for line in f]

        # 随机打乱数据（支持子集调试）
        shuffle_rng = random.Random(1337)
        shuffle_rng.shuffle(data)
        if max_per_task > 0:
            data = data[:max_per_task]

        # 运行任务评估
        accuracy = evaluate_task(model, tokenizer, data, device, task_meta)

        results[label] = accuracy

        # 计算中心化结果（相对于随机基线）
        row = eval_metadata[eval_metadata["Eval Task"] == label]
        random_baseline = row["Random baseline"].values[0]
        centered_result = (accuracy - 0.01 * random_baseline) / (1.0 - 0.01 * random_baseline)
        centered_results[label] = centered_result

        elapsed = time.time() - start_time
        print0(f"{accuracy:.4f} (centered: {centered_result:.4f}, {elapsed:.1f}s)")

    # 计算总体CORE分数
    core_score = sum(centered_results.values()) / len(centered_results)

    return {
        'core_score': core_score,
        'task_results': results,
        'centered_results': centered_results
    }
```

### 配置文件系统

```yaml
# core.yaml 示例配置
icl_tasks:
  - label: "hellaswag"
    icl_task_type: "multiple_choice"
    dataset_uri: "hellaswag.jsonl"
    num_fewshot: [10]
    continuation_delimiter: ""

  - label: "piqa"
    icl_task_type: "multiple_choice"
    dataset_uri: "piqa.jsonl"
    num_fewshot: [10]
    continuation_delimiter: ""

  - label: "arc_easy"
    icl_task_type: "multiple_choice"
    dataset_uri: "arc_easy.jsonl"
    num_fewshot: [10]
    continuation_delimiter: "\n"

  - label: "arc_challenge"
    icl_task_type: "multiple_choice"
    dataset_uri: "arc_challenge.jsonl"
    num_fewshot: [10]
    continuation_delimiter: "\n"

  # 更多任务...
```

## 任务特定评估

### ARC推理评估

```python
class ARCEvaluator:
    """ARC (AI2 Reasoning Challenge) 评估器"""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def evaluate(self, dataset_path, max_examples=None):
        """
        评估ARC数据集

        Args:
            dataset_path: 数据集路径
            max_examples: 最大评估示例数
        """
        # 加载数据
        examples = self.load_arc_data(dataset_path)
        if max_examples:
            examples = examples[:max_examples]

        correct = 0
        total = 0

        for example in examples:
            # 构建提示词
            prompt = self.build_arc_prompt(example)

            # 生成回答
            prediction = self.predict_answer(prompt)

            # 评估正确性
            if self.is_correct(prediction, example['answerKey']):
                correct += 1

            total += 1

        accuracy = correct / total if total > 0 else 0.0
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total
        }

    def build_arc_prompt(self, example):
        """构建ARC问题提示词"""
        question = example['question']
        choices = example['choices']['text']
        labels = example['choices']['label']

        prompt = f"Question: {question}\n\n"
        for i, (choice, label) in enumerate(zip(choices, labels)):
            prompt += f"{label}. {choice}\n"
        prompt += "\nAnswer: "

        return prompt

    def predict_answer(self, prompt):
        """预测答案"""
        tokens = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([tokens], dtype=torch.long)

        with torch.no_grad():
            logits = self.model(input_ids)
            predicted_id = torch.argmax(logits[:, -1, :], dim=-1).item()

        # 将预测的token映射到选项标签
        predicted_token = self.tokenizer.decode([predicted_id])
        return self.map_token_to_label(predicted_token)

    def map_token_to_label(self, token):
        """将token映射到ARC选项标签"""
        # 简化实现：检查token是否包含选项字母
        for label in ['A', 'B', 'C', 'D']:
            if label.lower() in token.lower():
                return label
        return token.strip()[:1]  # 默认取第一个字符
```

### GSM8K数学推理评估

```python
class GSM8KEvaluator:
    """GSM8K数学推理评估器"""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def evaluate(self, dataset_path, max_examples=None):
        """评估GSM8K数据集"""
        examples = self.load_gsm8k_data(dataset_path)
        if max_examples:
            examples = examples[:max_examples]

        correct = 0
        total = 0

        for example in examples:
            # 构建数学问题提示
            prompt = self.build_math_prompt(example['question'])

            # 生成解题过程
            solution = self.generate_solution(prompt)

            # 提取答案
            predicted_answer = self.extract_answer(solution)

            # 评估正确性
            if self.is_numeric_correct(predicted_answer, example['answer']):
                correct += 1

            total += 1

        accuracy = correct / total if total > 0 else 0.0
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total
        }

    def build_math_prompt(self, question):
        """构建数学问题提示词"""
        prompt = f"""Solve the following math problem step by step. Show your work and give the final answer.

Problem: {question}

Solution:"""
        return prompt

    def generate_solution(self, prompt, max_tokens=512):
        """生成解题过程"""
        tokens = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([tokens], dtype=torch.long)

        generated_tokens = []
        with torch.no_grad():
            for _ in range(max_tokens):
                logits = self.model(input_ids)
                next_token = torch.argmax(logits[:, -1, :], dim=-1)
                generated_tokens.append(next_token.item())

                # 检查是否生成结束标记
                if next_token.item() == self.tokenizer.encode_special("<|end|>"):
                    break

                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

        return self.tokenizer.decode(generated_tokens)

    def extract_answer(self, solution):
        """从解题过程中提取最终答案"""
        import re

        # 查找 "The answer is" 或类似的模式
        patterns = [
            r"The answer is ([\d,\.]+)",
            r"Answer: ([\d,\.]+)",
            r"= ([\d,\.]+)",
            r"(\d+\.?\d*)$"
        ]

        for pattern in patterns:
            match = re.search(pattern, solution, re.IGNORECASE)
            if match:
                answer = match.group(1).replace(',', '')
                try:
                    return float(answer)
                except ValueError:
                    continue

        return None

    def is_numeric_correct(self, predicted, expected):
        """检查数值答案是否正确"""
        try:
            pred_num = float(str(predicted).replace(',', ''))
            exp_num = float(str(expected).replace(',', ''))
            return abs(pred_num - exp_num) < 1e-6
        except (ValueError, TypeError):
            return False
```

## 对话质量评估

### ChatCORE评估

```python
class ChatCOREEvaluator:
    """对话CORE评估器"""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def evaluate_conversation_quality(self, conversations):
        """评估对话质量"""
        metrics = {
            'coherence': 0.0,
            'relevance': 0.0,
            'helpfulness': 0.0,
            'safety': 0.0,
            'overall': 0.0
        }

        for conversation in conversations:
            # 评估连贯性
            coherence_score = self.evaluate_coherence(conversation)
            metrics['coherence'] += coherence_score

            # 评估相关性
            relevance_score = self.evaluate_relevance(conversation)
            metrics['relevance'] += relevance_score

            # 评估有用性
            helpfulness_score = self.evaluate_helpfulness(conversation)
            metrics['helpfulness'] += helpfulness_score

            # 评估安全性
            safety_score = self.evaluate_safety(conversation)
            metrics['safety'] += safety_score

        # 计算平均分
        num_conversations = len(conversations)
        for key in metrics:
            if key != 'overall':
                metrics[key] /= num_conversations

        # 计算总体分数
        metrics['overall'] = (
            metrics['coherence'] * 0.3 +
            metrics['relevance'] * 0.3 +
            metrics['helpfulness'] * 0.25 +
            metrics['safety'] * 0.15
        )

        return metrics

    def evaluate_coherence(self, conversation):
        """评估对话连贯性"""
        messages = conversation['messages']
        coherence_scores = []

        for i in range(1, len(messages)):
            if messages[i]['role'] == 'assistant':
                # 计算与上下文的连贯性
                context = self.build_context(messages[:i])
                response = messages[i]['content']

                # 使用模型评估连贯性
                coherence_prompt = f"""
Context: {context}
Response: {response}

Rate the coherence of this response on a scale of 1-10, where 10 is perfectly coherent and 1 is completely incoherent.

Coherence score:"""

                score = self.get_model_rating(coherence_prompt)
                coherence_scores.append(score / 10.0)

        return sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0.0

    def evaluate_relevance(self, conversation):
        """评估回复相关性"""
        # 类似连贯性评估的实现
        pass

    def evaluate_helpfulness(self, conversation):
        """评估回复有用性"""
        pass

    def evaluate_safety(self, conversation):
        """评估回复安全性"""
        pass
```

### 采样质量评估

```python
class SamplingQualityEvaluator:
    """采样质量评估器"""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def evaluate_sampling_quality(self, prompts, num_samples=5):
        """评估不同采样参数下的生成质量"""
        results = {}

        for temperature in [0.1, 0.5, 1.0, 1.5]:
            for top_k in [1, 10, 50, 100]:
                config_key = f"temp_{temperature}_topk_{top_k}"
                results[config_key] = self.evaluate_config(
                    prompts, temperature, top_k, num_samples
                )

        return results

    def evaluate_config(self, prompts, temperature, top_k, num_samples):
        """评估特定配置下的采样质量"""
        diversity_scores = []
        coherence_scores = []
        repetition_scores = []

        for prompt in prompts:
            samples = []
            for _ in range(num_samples):
                sample = self.model.generate(
                    self.tokenizer.encode(prompt),
                    temperature=temperature,
                    top_k=top_k,
                    max_tokens=100
                )
                samples.append(self.tokenizer.decode(sample))

            # 计算多样性
            diversity = self.calculate_diversity(samples)
            diversity_scores.append(diversity)

            # 计算连贯性
            coherence = self.calculate_coherence(samples, prompt)
            coherence_scores.append(coherence)

            # 计算重复率
            repetition = self.calculate_repetition(samples)
            repetition_scores.append(repetition)

        return {
            'diversity': sum(diversity_scores) / len(diversity_scores),
            'coherence': sum(coherence_scores) / len(coherence_scores),
            'repetition': sum(repetition_scores) / len(repetition_scores)
        }

    def calculate_diversity(self, samples):
        """计算生成样本的多样性"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        if len(samples) <= 1:
            return 0.0

        # 计算TF-IDF向量
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(samples)

        # 计算相似度矩阵
        similarities = cosine_similarity(tfidf_matrix)

        # 多样性 = 1 - 平均相似度
        avg_similarity = (similarities.sum() - len(samples)) / (len(samples) * (len(samples) - 1))
        diversity = 1 - avg_similarity

        return diversity

    def calculate_repetition(self, samples):
        """计算重复率"""
        total_repetitions = 0
        total_tokens = 0

        for sample in samples:
            tokens = sample.split()
            total_tokens += len(tokens)

            # 计算重复的n-gram
            n_grams = set()
            repetitions = 0

            for i in range(len(tokens) - 1):
                bigram = f"{tokens[i]} {tokens[i+1]}"
                if bigram in n_grams:
                    repetitions += 1
                else:
                    n_grams.add(bigram)

            total_repetitions += repetitions

        repetition_rate = total_repetitions / total_tokens if total_tokens > 0 else 0.0
        return 1 - repetition_rate  # 返回非重复率
```

## 系统性能评估

### 推理性能评估

```python
class InferencePerformanceEvaluator:
    """推理性能评估器"""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def evaluate_inference_performance(self, test_cases):
        """评估推理性能"""
        results = {
            'latency': [],
            'throughput': [],
            'memory_usage': [],
            'gpu_utilization': []
        }

        for test_case in test_cases:
            # 测试不同输入长度下的性能
            input_length = test_case['input_length']
            output_length = test_case['output_length']

            # 预热
            self.warmup_model()

            # 测量延迟
            latency = self.measure_latency(input_length, output_length)
            results['latency'].append(latency)

            # 测量吞吐量
            throughput = self.measure_throughput(input_length, output_length)
            results['throughput'].append(throughput)

            # 测量内存使用
            memory_usage = self.measure_memory_usage(input_length, output_length)
            results['memory_usage'].append(memory_usage)

            # 测量GPU利用率
            gpu_util = self.measure_gpu_utilization(input_length, output_length)
            results['gpu_utilization'].append(gpu_util)

        return {
            'avg_latency': sum(results['latency']) / len(results['latency']),
            'avg_throughput': sum(results['throughput']) / len(results['throughput']),
            'peak_memory': max(results['memory_usage']),
            'avg_gpu_util': sum(results['gpu_utilization']) / len(results['gpu_utilization'])
        }

    def measure_latency(self, input_length, output_length):
        """测量推理延迟"""
        import time

        # 生成测试输入
        test_input = torch.randint(
            0, self.tokenizer.get_vocab_size(),
            (1, input_length)
        )

        # 预热
        for _ in range(3):
            with torch.no_grad():
                _ = self.model(test_input)

        # 测量延迟
        torch.cuda.synchronize()
        start_time = time.time()

        with torch.no_grad():
            for _ in range(output_length):
                logits = self.model(test_input)
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                test_input = torch.cat([test_input, next_token], dim=1)

        torch.cuda.synchronize()
        end_time = time.time()

        return end_time - start_time

    def measure_throughput(self, input_length, output_length, batch_size=8):
        """测量推理吞吐量"""
        import time

        # 生成批次输入
        test_input = torch.randint(
            0, self.tokenizer.get_vocab_size(),
            (batch_size, input_length)
        )

        total_tokens = 0
        start_time = time.time()

        with torch.no_grad():
            for _ in range(output_length):
                logits = self.model(test_input)
                next_tokens = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                test_input = torch.cat([test_input, next_tokens], dim=1)
                total_tokens += batch_size

        end_time = time.time()
        elapsed_time = end_time - start_time

        return total_tokens / elapsed_time  # tokens per second

    def measure_memory_usage(self, input_length, output_length):
        """测量内存使用"""
        if not torch.cuda.is_available():
            return 0.0

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        # 生成测试输入
        test_input = torch.randint(
            0, self.tokenizer.get_vocab_size(),
            (1, input_length),
            device='cuda'
        )

        with torch.no_grad():
            for _ in range(output_length):
                logits = self.model(test_input)
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                test_input = torch.cat([test_input, next_token], dim=1)

        peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
        return peak_memory
```

## 报告生成系统

### 系统信息收集

```python
def get_system_info():
    """收集系统信息"""
    info = {}

    # 基本系统信息
    info['hostname'] = socket.gethostname()
    info['platform'] = platform.system()
    info['python_version'] = platform.python_version()
    info['torch_version'] = torch.__version__

    # CPU和内存信息
    info['cpu_count'] = psutil.cpu_count(logical=False)
    info['cpu_count_logical'] = psutil.cpu_count(logical=True)
    info['memory_gb'] = psutil.virtual_memory().total / (1024**3)

    # 用户和环境信息
    info['user'] = os.environ.get('USER', 'unknown')
    info['cwd'] = os.getcwd()

    return info

def get_gpu_info():
    """收集GPU信息"""
    if not torch.cuda.is_available():
        return {"available": False}

    num_devices = torch.cuda.device_count()
    info = {
        "available": True,
        "count": num_devices,
        "names": [],
        "memory_gb": [],
        "compute_capability": []
    }

    for i in range(num_devices):
        props = torch.cuda.get_device_properties(i)
        info["names"].append(props.name)
        info["memory_gb"].append(props.total_memory / (1024**3))
        info["compute_capability"].append(f"{props.major}.{props.minor}")

    # CUDA版本信息
    info["cuda_version"] = torch.version.cuda or "unknown"
    info["cudnn_version"] = torch.backends.cudnn.version() or "unknown"

    return info

def get_git_info():
    """收集Git信息"""
    info = {}
    info['commit'] = run_command("git rev-parse --short HEAD") or "unknown"
    info['branch'] = run_command("git rev-parse --abbrev-ref HEAD") or "unknown"

    # 检查是否有未提交的更改
    status = run_command("git status --porcelain")
    info['dirty'] = bool(status) if status is not None else False

    # 获取最新提交信息
    info['message'] = run_command("git log -1 --pretty=%B") or ""
    info['message'] = info['message'].split('\n')[0][:80]  # 第一行，截断到80字符

    return info
```

### 评估报告生成

```python
class EvaluationReportGenerator:
    """评估报告生成器"""

    def __init__(self):
        self.system_info = get_system_info()
        self.gpu_info = get_gpu_info()
        self.git_info = get_git_info()

    def generate_comprehensive_report(self, evaluation_results):
        """生成综合评估报告"""
        report_sections = []

        # 1. 执行摘要
        report_sections.append(self.generate_executive_summary(evaluation_results))

        # 2. 系统信息
        report_sections.append(self.generate_system_info_section())

        # 3. 基础评估结果
        if 'base_evaluation' in evaluation_results:
            report_sections.append(
                self.generate_base_evaluation_section(evaluation_results['base_evaluation'])
            )

        # 4. 任务评估结果
        if 'task_evaluations' in evaluation_results:
            report_sections.append(
                self.generate_task_evaluation_section(evaluation_results['task_evaluations'])
            )

        # 5. 对话评估结果
        if 'chat_evaluation' in evaluation_results:
            report_sections.append(
                self.generate_chat_evaluation_section(evaluation_results['chat_evaluation'])
            )

        # 6. 性能评估结果
        if 'performance_evaluation' in evaluation_results:
            report_sections.append(
                self.generate_performance_section(evaluation_results['performance_evaluation'])
            )

        # 7. 结论和建议
        report_sections.append(self.generate_conclusions(evaluation_results))

        # 组合完整报告
        full_report = "\n\n".join(report_sections)
        return full_report

    def generate_executive_summary(self, results):
        """生成执行摘要"""
        summary = "## 执行摘要\n\n"

        # 总体性能概述
        if 'base_evaluation' in results and 'core_score' in results['base_evaluation']:
            core_score = results['base_evaluation']['core_score']
            summary += f"**CORE分数**: {core_score:.4f}\n\n"

        # 关键指标
        summary += "### 关键性能指标\n\n"
        summary += "| 指标 | 数值 | 说明 |\n"
        summary += "|------|------|------|\n"

        if 'task_evaluations' in results:
            task_results = results['task_evaluations']
            for task_name, task_result in task_results.items():
                if 'accuracy' in task_result:
                    summary += f"| {task_name} | {task_result['accuracy']:.4f} | 准确率 |\n"

        # 总体评估
        summary += "\n### 总体评估\n\n"
        summary += self._generate_overall_assessment(results)

        return summary

    def generate_system_info_section(self):
        """生成系统信息部分"""
        section = "## 系统信息\n\n"

        # 硬件信息
        section += "### 硬件配置\n\n"
        section += f"- **CPU**: {self.system_info['cpu_count']} 核心 ({self.system_info['cpu_count_logical']} 逻辑核心)\n"
        section += f"- **内存**: {self.system_info['memory_gb']:.1f} GB\n"

        if self.gpu_info['available']:
            section += f"- **GPU**: {self.gpu_info['count']} 个设备\n"
            for i, gpu_name in enumerate(self.gpu_info['names']):
                memory_gb = self.gpu_info['memory_gb'][i]
                compute_cap = self.gpu_info['compute_capability'][i]
                section += f"  - GPU {i}: {gpu_name} ({memory_gb:.1f} GB, Compute {compute_cap})\n"

        # 软件环境
        section += "\n### 软件环境\n\n"
        section += f"- **操作系统**: {self.system_info['platform']}\n"
        section += f"- **Python版本**: {self.system_info['python_version']}\n"
        section += f"- **PyTorch版本**: {self.system_info['torch_version']}\n"

        if self.gpu_info['available']:
            section += f"- **CUDA版本**: {self.gpu_info['cuda_version']}\n"
            section += f"- **cuDNN版本**: {self.gpu_info['cudnn_version']}\n"

        # Git信息
        section += "\n### 代码版本\n\n"
        section += f"- **提交哈希**: {self.git_info['commit']}\n"
        section += f"- **分支**: {self.git_info['branch']}\n"
        section += f"- **是否有未提交更改**: {'是' if self.git_info['dirty'] else '否'}\n"
        section += f"- **最新提交信息**: {self.git_info['message']}\n"

        return section

    def generate_base_evaluation_section(self, base_results):
        """生成基础评估部分"""
        section = "## 基础模型评估\n\n"

        # CORE分数
        core_score = base_results['core_score']
        section += f"### CORE综合评分: {core_score:.4f}\n\n"

        # 任务详细结果
        section += "### 各任务详细结果\n\n"
        section += "| 任务 | 准确率 | 中心化分数 |\n"
        section += "|------|--------|------------|\n"

        task_results = base_results['task_results']
        centered_results = base_results['centered_results']

        for task_name in sorted(task_results.keys()):
            accuracy = task_results[task_name]
            centered = centered_results[task_name]
            section += f"| {task_name} | {accuracy:.4f} | {centered:.4f} |\n"

        # 性能分析
        section += "\n### 性能分析\n\n"
        section += self._analyze_performance(task_results)

        return section

    def _analyze_performance(self, task_results):
        """分析性能表现"""
        analysis = ""

        # 计算平均分数
        avg_score = sum(task_results.values()) / len(task_results)
        analysis += f"- **平均准确率**: {avg_score:.4f}\n"

        # 找出表现最好和最差的任务
        best_task = max(task_results.items(), key=lambda x: x[1])
        worst_task = min(task_results.items(), key=lambda x: x[1])

        analysis += f"- **表现最好任务**: {best_task[0]} ({best_task[1]:.4f})\n"
        analysis += f"- **表现最差任务**: {worst_task[0]} ({worst_task[1]:.4f})\n"

        # 性能分类
        high_performing = [name for name, score in task_results.items() if score > 0.8]
        medium_performing = [name for name, score in task_results.items() if 0.5 <= score <= 0.8]
        low_performing = [name for name, score in task_results.items() if score < 0.5]

        if high_performing:
            analysis += f"- **高表现任务** (>0.8): {', '.join(high_performing)}\n"
        if medium_performing:
            analysis += f"- **中等表现任务** (0.5-0.8): {', '.join(medium_performing)}\n"
        if low_performing:
            analysis += f"- **低表现任务** (<0.5): {', '.join(low_performing)}\n"

        return analysis

    def save_report(self, report, filename="evaluation_report.md"):
        """保存报告到文件"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"评估报告已保存到: {filename}")
```

## 使用示例

### 完整评估流程

```python
def run_full_evaluation(model, tokenizer, device):
    """运行完整评估流程"""

    # 1. 基础评估
    print("开始基础模型评估...")
    base_results = evaluate_model(model, tokenizer, device, max_per_task=100)

    # 2. 任务评估
    print("开始任务特定评估...")
    task_results = {}

    # ARC评估
    arc_evaluator = ARCEvaluator(model, tokenizer)
    task_results['ARC'] = arc_evaluator.evaluate("arc_challenge.jsonl", max_examples=50)

    # GSM8K评估
    gsm8k_evaluator = GSM8KEvaluator(model, tokenizer)
    task_results['GSM8K'] = gsm8k_evaluator.evaluate("gsm8k.jsonl", max_examples=50)

    # 3. 对话评估
    print("开始对话质量评估...")
    chat_evaluator = ChatCOREEvaluator(model, tokenizer)
    # 加载对话测试数据...
    chat_results = chat_evaluator.evaluate_conversation_quality(test_conversations)

    # 4. 性能评估
    print("开始性能评估...")
    perf_evaluator = InferencePerformanceEvaluator(model, tokenizer)
    perf_results = perf_evaluator.evaluate_inference_performance(perf_test_cases)

    # 5. 生成报告
    print("生成评估报告...")
    report_generator = EvaluationReportGenerator()

    all_results = {
        'base_evaluation': base_results,
        'task_evaluations': task_results,
        'chat_evaluation': chat_results,
        'performance_evaluation': perf_results
    }

    report = report_generator.generate_comprehensive_report(all_results)
    report_generator.save_report(report)

    return all_results
```

## 总结

NanoChat的评估体系体现了现代AI模型评估的几个重要原则：

1. **多维度评估**：从基础能力到任务性能的全方位评估
2. **标准化基准**：使用广泛认可的评估基准和数据集
3. **自动化流程**：自动化的评估流程和报告生成
4. **性能监控**：实时性能跟踪和资源使用监控
5. **可重复性**：标准化的评估流程确保结果可重复

这种全面的评估体系为模型改进提供了科学依据，也为用户提供了透明的性能信息。

## 下一步

在下一篇文章中，我们将深入分析NanoChat的生产环境部署策略，了解如何将模型投入实际使用。

---

**第十篇文章预告**：《NanoChat深入解析(10)：生产环境部署最佳实践》将详细解析容器化部署、负载均衡和监控运维的实现细节。