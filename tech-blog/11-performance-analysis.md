# NanoChat深入解析(11)：性能瓶颈分析与优化方案

## 前言

在AI模型的生产部署中，性能优化是一个永恒的话题。NanoChat作为一个小型但功能完整的ChatGPT克隆，其性能瓶颈分析和优化策略对理解大模型服务的性能工程具有重要价值。本文将深入分析NanoChat的性能瓶颈，并提供实用的优化方案。

## 性能分析框架

### 性能指标体系

```
整体性能
├── 响应时间延迟
│   ├── 首字节时间 (TTFT)
│   ├── 每个token延迟
│   └── 总体响应时间
├── 吞吐量指标
│   ├── 每秒请求数 (RPS)
│   ├── 每秒token数 (TPS)
│   └── 并发处理能力
├── 资源利用率
│   ├── GPU利用率
│   ├── 内存使用率
│   ├── CPU使用率
│   └── 网络带宽
└── 系统健康
    ├── 错误率
    ├── 超时率
    └── 可用性
```

### 性能监控工具

```python
# performance_profiler.py
import time
import torch
import psutil
import numpy as np
from contextlib import contextmanager
from collections import defaultdict, deque

class NanoChatProfiler:
    """NanoChat性能分析器"""

    def __init__(self):
        self.metrics = {
            'latency': deque(maxlen=1000),
            'throughput': deque(maxlen=100),
            'memory': deque(maxlen=100),
            'gpu_utilization': deque(maxlen=100),
            'errors': defaultdict(int)
        }

    @contextmanager
    def profile_inference(self, model, tokenizer, prompt):
        """分析推理性能"""
        start_time = time.perf_counter()
        start_memory = psutil.Process().memory_info().rss
        gpu_memory_start = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        # 执行推理
        try:
            tokens = tokenizer.encode(prompt)
            input_ids = torch.tensor([tokens], dtype=torch.long)

            if torch.cuda.is_available():
                input_ids = input_ids.cuda()

            with torch.no_grad():
                output = model(input_ids)

        except Exception as e:
            self.metrics['errors']['inference_error'] += 1
            raise e

        # 计算指标
        end_time = time.perf_counter()
        end_memory = psutil.Process().memory_info().rss
        gpu_memory_end = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        latency = (end_time - start_time) * 1000  # ms
        memory_delta = end_memory - start_memory
        gpu_memory_delta = gpu_memory_end - gpu_memory_start
        tokens_generated = len(output[0]) if isinstance(output, list) else output.shape[-1]
        tokens_per_second = tokens_generated / (end_time - start_time)

        # 记录指标
        self.metrics['latency'].append(latency)
        self.metrics['throughput'].append(tokens_per_second)
        self.metrics['memory'].append(memory_delta / 1024**2)  # MB
        if torch.cuda.is_available():
            self.metrics['gpu_utilization'].append(gpu_memory_delta / 1024**2)  # MB

        return {
            'latency_ms': latency,
            'tokens_per_second': tokens_per_second,
            'memory_mb': memory_delta / 1024**2,
            'gpu_memory_mb': gpu_memory_delta / 1024**2
        }
```

## 推理性能瓶颈分析

### 1. 计算瓶颈

#### GPU利用率分析

```python
def analyze_gpu_bottlenecks(profiler_data):
    """分析GPU相关瓶颈"""

    bottleneck_analysis = {
        'kernel_launch_overhead': 0,
        'memory_bandwidth': 0,
        'compute_utilization': 0,
        'cache_misses': 0
    }

    # GPU利用率模式分析
    gpu_util_history = list(profiler_data.metrics['gpu_utilization'])

    # 检测GPU利用率不足
    avg_gpu_util = np.mean(gpu_util_history)
    if avg_gpu_util < 50:
        bottleneck_analysis['compute_utilization'] = 1
        bottleneck_analysis['recommendation'] = "GPU利用率不足，考虑增加批大小或模型并行"

    # 检测内存带宽瓶颈
    memory_trend = np.array(gpu_util_history[-100:])  # 最近100个样本
    memory_growth_rate = np.polyfit(range(len(memory_trend)), memory_trend)[0]

    if memory_growth_rate > 0.1:  # 持续增长
        bottleneck_analysis['memory_bandwidth'] = 1
        bottleneck_analysis['recommendation'] = "内存使用持续增长，检查内存泄漏或减小批次大小"

    return bottleneck_analysis

# 使用示例
profiler = NanoChatProfiler()
with profiler.profile_inference(model, tokenizer, "Hello world"):
    result = profiler.get_performance_summary()
    bottleneck_info = analyze_gpu_bottlenecks(profiler)

print(f"GPU利用率: {result['gpu_utilization']:.1f}%")
print(f"瓶颈分析: {bottleneck_info['recommendation']}")
```

#### 内存使用分析

```python
def analyze_memory_patterns(profiler_data):
    """分析内存使用模式"""

    memory_history = list(profiler_data.metrics['memory'])

    # 内存使用模式检测
    memory_stats = {
        'baseline_mb': np.percentile(memory_history[:10], 50),  # 前10个的中位数作为基线
        'peak_mb': max(memory_history),
        'avg_mb': np.mean(memory_history),
        'variance_mb': np.var(memory_history),
        'growth_trend': None
    }

    # 检测内存增长趋势
    if len(memory_history) > 50:
        x = np.arange(len(memory_history))
        coeffs = np.polyfit(x, memory_history, 1)
        memory_stats['growth_trend'] = coeffs[0]  # 斜率

    # 瓶颈识别
    memory_bottlenecks = {
        'memory_leak': memory_stats['growth_trend'] > 1.0,  # 持续增长
        'high_variance': memory_stats['variance_mb'] > 1000,  # 高方差
        'memory_spike': memory_stats['peak_mb'] > memory_stats['baseline_mb'] * 3,
        'insufficient_memory': memory_stats['peak_mb'] > psutil.virtual_memory().total * 0.8 / 1024**2
    }

    recommendations = []

    if memory_bottlenecks['memory_leak']:
        recommendations.append("检测到内存泄漏，检查循环引用和大对象")

    if memory_bottlenecks['high_variance']:
        recommendations.append("内存使用波动较大，考虑实现内存池或垃圾回收")

    if memory_bottlenecks['memory_spike']:
        recommendations.append("检测到内存峰值，考虑分批处理或限制并发")

    return {
        'memory_stats': memory_stats,
        'bottlenecks': memory_bottlenecks,
        'recommendations': recommendations
    }
```

### 2. I/O瓶颈分析

#### 网络延迟分析

```python
import asyncio
import time
from collections import defaultdict

class NetworkAnalyzer:
    """网络I/O分析器"""

    def __init__(self):
        self.request_times = defaultdict(list)
        self.error_rates = defaultdict(int)

    async def analyze_request(self, endpoint, payload_size):
        """分析网络请求性能"""

        start_time = time.perf_counter()

        try:
            # 模拟网络请求
            response = await self.make_request(endpoint, payload_size)

            end_time = time.perf_counter()
            latency = (end_time - start_time) * 1000  # ms

            # 记录指标
            self.request_times[endpoint].append(latency)

            return {
                'success': True,
                'latency_ms': latency,
                'response_size': len(response),
                'throughput_mbps': (len(response) * 8) / (latency / 1000) / 1024**2
            }

        except Exception as e:
            self.error_rates[endpoint] += 1
            return {
                'success': False,
                'error': str(e),
                'latency_ms': None
            }

    def get_network_stats(self):
        """获取网络统计信息"""
        stats = {}

        for endpoint, times in self.request_times.items():
            if times:
                stats[endpoint] = {
                    'avg_latency_ms': np.mean(times),
                    'p95_latency_ms': np.percentile(times, 95),
                    'p99_latency_ms': np.percentile(times, 99),
                    'max_latency_ms': max(times),
                    'request_count': len(times)
                }

                # 性能分级
                avg_latency = stats[endpoint]['avg_latency_ms']
                if avg_latency < 100:
                    stats[endpoint]['performance_grade'] = 'A'
                elif avg_latency < 200:
                    stats[endpoint]['performance_grade'] = 'B'
                elif avg_latency < 500:
                    stats[endpoint]['performance_grade'] = 'C'
                else:
                    stats[endpoint]['performance_grade'] = 'D'

        return stats

# 使用示例
analyzer = NetworkAnalyzer()

# 分析不同端点的网络性能
endpoints = ['/chat/completions', '/health', '/metrics']
for endpoint in endpoints:
    for _ in range(10):  # 模拟10个请求
        await analyzer.analyze_request(endpoint, 1024)  # 1KB payload

network_stats = analyzer.get_network_stats()
print("网络性能分析结果:", network_stats)
```

#### 存储I/O分析

```python
import aiofiles
import time
from pathlib import Path

class StorageAnalyzer:
    """存储I/O分析器"""

    def __init__(self):
        self.io_stats = {
            'read_latency': [],
            'write_latency': [],
            'throughput_mbps': [],
            'queue_depth': []
        }

    async def analyze_file_io(self, file_path, operation='read', chunk_size=8192):
        """分析文件I/O性能"""

        try:
            start_time = time.perf_counter()

            async with aiofiles.open(file_path, 'rb' if operation == 'read' else 'wb') as f:
                total_bytes = 0

                while True:
                    if operation == 'read':
                        chunk = await f.read(chunk_size)
                        if not chunk:
                            break
                    else:
                        total_bytes += len(chunk)

                elif operation == 'write':
                    data = b'x' * chunk_size  # 测试数据
                    await f.write(data)
                    total_bytes += len(data)

                    # 检查队列深度
                    queue_depth = getattr(f, '_raw_fileobj', None)
                    if queue_depth:
                        import fcntl
                        import os
                        fl = fcntl.fcntl(f, fcntl.F_GETFL)
                        if fl & os.O_NONBLOCK:
                            # 非阻塞I/O，检查内核队列
                            import array
                            arg = array.array('I', [0])
                            # 在Linux上检查队列深度
                            try:
                                import ioctl
                                if hasattr(ioctl, 'FIONREAD'):
                                    bytes_ready = ioctl.ioctl(f, ioctl.FIONREAD, arg, False)
                                    self.io_stats['queue_depth'].append(bytes_ready[0])
                            except:
                                pass

            end_time = time.perf_counter()
            operation_time = (end_time - start_time) * 1000  # ms

            throughput_mbps = (total_bytes * 8) / (operation_time / 1000) / 1024**2

            self.io_stats[f'{operation}_latency'].append(operation_time)
            self.io_stats['throughput_mbps'].append(throughput_mbps)

            return {
                'operation': operation,
                'latency_ms': operation_time,
                'throughput_mbps': throughput_mbps,
                'total_bytes': total_bytes
            }

        except Exception as e:
            return {
                'operation': operation,
                'error': str(e),
                'latency_ms': None,
                'throughput_mbps': None
            }

    def get_io_analysis(self):
        """获取I/O分析结果"""
        analysis = {}

        for operation in ['read', 'write']:
            if f'{operation}_latency' in self.io_stats:
                latencies = self.io_stats[f'{operation}_latency']
                throughputs = self.io_stats['throughput_mbps']

                if latencies and throughputs:
                    analysis[operation] = {
                        'avg_latency_ms': np.mean(latencies),
                        'p95_latency_ms': np.percentile(latencies, 95),
                        'avg_throughput_mbps': np.mean(throughputs),
                        'max_throughput_mbps': max(throughputs)
                    }

                    # 瓶颈识别
                    avg_latency = analysis[operation]['avg_latency_ms']
                    avg_throughput = analysis[operation]['avg_throughput_mbps']

                    if avg_latency > 100:  # 高延迟
                        analysis[operation]['bottleneck'] = 'high_latency'
                    elif avg_throughput < 10:  # 低吞吐量
                        analysis[operation]['bottleneck'] = 'low_throughput'
                    else:
                        analysis[operation]['bottleneck'] = 'normal'

        return analysis

# 使用示例
analyzer = StorageAnalyzer()

# 分析文件读取性能
await analyzer.analyze_file_io('/path/to/large/model.bin', 'read')

# 分析文件写入性能
await analyzer.analyze_file_io('/path/to/output.bin', 'write')

io_analysis = analyzer.get_io_analysis()
print("存储I/O分析结果:", io_analysis)
```

## 优化方案

### 1. 推理优化

#### 模型量化

```python
# quantization.py
import torch
from torch.ao.quantization import quantize_dynamic_jit, QConfigMapping

class ModelQuantizer:
    """模型量化工具"""

    def __init__(self, model_path, output_path):
        self.model_path = model_path
        self.output_path = output_path

    def quantize_model(self):
        """动态量化模型"""

        print("加载原始模型...")
        model = torch.load(self.model_path, map_location='cpu')

        print("执行动态量化...")
        quantized_model = quantize_dynamic_jit(
            model,
            {
                torch.nn.Linear: QConfigMapping(
                    dtype=torch.qint8,
                    per_channel=True,
                    group_size=64,
                ),
                torch.nn.Conv1d: QConfigMapping(
                    dtype=torch.qint8,
                    per_channel=True,
                    group_size=64,
                ),
                torch.nn.Conv2d: QConfigMapping(
                    dtype=torch.qint8,
                    per_channel=True,
                    group_size=64,
                )
            },
            example_inputs=torch.randn(1, 1, 1024)
        )

        print(f"量化完成，模型大小: {self.get_model_size(model)} -> {self.get_model_size(quantized_model)}")

        # 保存量化模型
        torch.save(quantized_model.state_dict(), self.output_path)

        return quantized_model

    def compare_performance(self, original_model, quantized_model, test_input):
        """比较原始模型和量化模型的性能"""

        # 原始模型推理
        with torch.no_grad():
            original_output = original_model(test_input.cuda())

        # 量化模型推理
        with torch.no_grad():
            quantized_output = quantized_model(test_input.cuda())

        # 性能对比
        def measure_inference(model, input_tensor, name):
            torch.cuda.synchronize()
            start_time = time.perf_counter()

            for _ in range(100):
                with torch.no_grad():
                    output = model(input_tensor)

            torch.cuda.synchronize()
            end_time = time.perf_counter()

            avg_time = (end_time - start_time) / 100 * 1000  # ms

            return {
                'model': name,
                'avg_time_ms': avg_time,
                'throughput': 1.0 / avg_time * 1000
            }

        original_perf = measure_inference(original_model, test_input, "original")
        quantized_perf = measure_inference(quantized_model, test_input, "quantized")

        speedup = original_perf['avg_time_ms'] / quantized_perf['avg_time_ms']

        print(f"性能对比:")
        print(f"原始模型: {original_perf['avg_time_ms']:.2f}ms")
        print(f"量化模型: {quantized_perf['avg_time_ms']:.2f}ms")
        print(f"加速比: {speedup:.2f}x")

        return {
            'original_performance': original_perf,
            'quantized_performance': quantized_perf,
            'speedup': speedup
        }

    def get_model_size(self, model):
        """获取模型大小(MB)"""
        torch.save(model.state_dict(), '/tmp/temp_model.pt')
        return os.path.getsize('/tmp/temp_model.pt') / 1024**2

# 使用示例
quantizer = ModelQuantizer('models/nanochat_d20.pt', 'models/nanochat_d20_quantized.pt')
original_model = torch.load('models/nanochat_d20.pt')
quantized_model = quantizer.quantize_model()

# 性能对比
test_input = torch.randint(0, 50304, (1, 512))  # 模拟输入
performance_comparison = quantizer.compare_performance(original_model, quantized_model, test_input)
```

#### KV缓存优化

```python
# kv_cache_optimizer.py
import torch
from typing import Dict, List, Optional

class OptimizedKVCache:
    """优化的KV缓存实现"""

    def __init__(self, num_layers, num_heads, head_dim, seq_len):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = seq_len

        # 使用更紧凑的数据类型
        self.k_cache = torch.zeros(
            (num_layers, seq_len, num_heads, head_dim),
            dtype=torch.float16  # 使用FP16节省内存
        )
        self.v_cache = torch.zeros_like(self.k_cache)

        # 缓存位置跟踪
        self.pos = 0

        # 预分配的缓冲区，避免动态分配
        self.buffer_size = 1024
        self.k_buffer = torch.zeros(
            (num_layers, self.buffer_size, num_heads, head_dim),
            dtype=torch.float16
        )
        self.v_buffer = torch.zeros_like(self.k_buffer)

    def insert(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor):
        """优化的KV插入"""
        batch_size, seq_len, num_heads, head_dim = k.shape
        t_start = self.pos
        t_end = t_start + seq_len

        # 检查是否需要扩展缓存
        if t_end > self.max_seq_len:
            self._expand_cache(t_end)

        # 高效的KV更新
        self.k_cache[layer_idx, t_start:t_end] = k
        self.v_cache[layer_idx, t_start:t_end] = v

        self.pos = t_end

        # 返回当前视图
        k_view = self.k_cache[layer_idx, :t_end]
        v_view = self.v_cache[layer_idx, :t_end]

        return k_view, v_view

    def _expand_cache(self, new_size: int):
        """扩展缓存大小"""
        old_size = self.max_seq_len

        # 计算新的大小（2的幂次）
        self.max_seq_len = 2 ** ((new_size - 1).bit_length())

        # 重新分配缓存
        new_k_cache = torch.zeros(
            (self.num_layers, self.max_seq_len, self.num_heads, self.head_dim),
            dtype=torch.float16
        )
        new_v_cache = torch.zeros_like(new_k_cache)

        # 复制现有数据
        min_size = min(old_size, new_size)
        new_k_cache[:, :min_size] = self.k_cache[:, :min_size]
        new_v_cache[:, :min_size] = self.v_cache[:, :min_size]

        # 更新引用
        self.k_cache = new_k_cache
        self.v_cache = new_v_cache

        print(f"缓存扩展: {old_size} -> {self.max_seq_len}")

# 性能测试
def benchmark_kv_cache():
    """KV缓存性能基准测试"""

    cache_configs = [
        ('标准缓存', lambda: StandardKVCache(20, 16, 128, 2048)),
        ('优化缓存', lambda: OptimizedKVCache(20, 16, 128, 2048)),
    ]

    test_config = {
        'num_layers': 20,
        'seq_len': 2048,
        'num_heads': 16,
        'head_dim': 128,
        'num_iterations': 1000
    }

    results = {}

    for config_name, cache_factory in cache_configs:
        cache = cache_factory(**test_config)

        # 模拟KV操作
        start_time = time.perf_counter()

        for i in range(test_config['num_iterations']):
            layer_idx = i % test_config['num_layers']

            # 模拟KV张量
            k = torch.randn(1, 1, 16, 128)  # (batch, seq, heads, dim)
            v = torch.randn_like(k)

            # 执行插入操作
            cache.insert(layer_idx, k, v)

        end_time = time.perf_counter()

        avg_time = (end_time - start_time) / test_config['num_iterations'] * 1000

        results[config_name] = {
            'avg_time_ms': avg_time,
            'throughput_ops_per_sec': 1000 / avg_time,
            'memory_usage_mb': cache.k_cache.numel() * 2 / 1024**2  # FP16
        }

    print("KV缓存性能基准测试结果:")
    for config_name, result in results.items():
        print(f"{config_name}:")
        print(f"  平均时间: {result['avg_time_ms']:.4f}ms")
        print(f"  吞吐量: {result['throughput_ops_per_sec']:.2f} ops/s")
        print(f"  内存使用: {result['memory_usage_mb']:.2f}MB")
        print()

    return results
```

### 2. 内存优化

#### 内存池管理

```python
# memory_pool.py
import torch
from typing import Any, Dict, Optional
from collections import defaultdict
import threading
import weakref

class OptimizedMemoryPool:
    """优化的内存池"""

    def __init__(self):
        self.pools = defaultdict(dict)
        self.pool_lock = threading.Lock()
        self.active_tensors = defaultdict(int)
        self.cache_hits = 0
        self.cache_misses = 0

    def get_tensor(self, shape: tuple, dtype: torch.dtype, device: str) -> torch.Tensor:
        """获取指定形状和类型的tensor"""
        key = (shape, dtype, device)

        with self.pool_lock:
            if key in self.pools and self.pools[key]:
                # 复用现有tensor
                for i, tensor in enumerate(self.pools[key]):
                    if tensor.data_ptr() == 0:  # 未被使用
                        self.pools[key][i] = tensor
                        tensor.data_ptr() = 1  # 标记为已使用
                        self.active_tensors[key] += 1
                        self.cache_hits += 1
                        return tensor

            # 没有可用的tensor，创建新的
            new_tensor = torch.zeros(shape, dtype=dtype, device=device)
            if key not in self.pools:
                self.pools[key] = []
            self.pools[key].append(new_tensor)
            new_tensor.data_ptr() = 1
            self.active_tensors[key] += 1
            self.cache_misses += 1
            return new_tensor

    def return_tensor(self, tensor: torch.Tensor):
        """归还tensor到内存池"""
        tensor.data_ptr() = 0  # 标记为未使用
        key = self._get_tensor_key(tensor)

        with self.pool_lock:
            self.active_tensors[key] -= 1

    def _get_tensor_key(self, tensor: torch.Tensor) -> tuple:
        """生成tensor键"""
        return (tuple(tensor.shape), tensor.dtype, str(tensor.device))

    def get_stats(self) -> Dict[str, float]:
        """获取内存池统计"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0

        return {
            'hit_rate': hit_rate,
            'total_requests': total_requests,
            'active_tensors': sum(self.active_tensors.values()),
            'pools_count': len(self.pools)
        }

# 使用示例
memory_pool = OptimizedMemoryPool()

# 在推理循环中使用内存池
for batch in data_loader:
    inputs = memory_pool.get_tensor((32, 1024), torch.long, "cuda")
    targets = memory_pool.get_tensor((32, 1024), torch.long, "cuda")

    # 使用tensor进行推理
    with torch.no_grad():
        output = model(inputs)
        loss = compute_loss(output, targets)

    # 归还tensor
    memory_pool.return_tensor(inputs)
    memory_pool.return_tensor(targets)
    memory_pool.return_tensor(output)

# 获取内存池统计
stats = memory_pool.get_stats()
print(f"内存池命中率: {stats['hit_rate']:.2%}")
print(f"活跃tensor数: {stats['active_tensors']}")
```

### 3. I/O优化

#### 批量处理优化

```python
# batch_processor.py
import asyncio
import torch
from typing import List, Callable
import time

class BatchProcessor:
    """批量处理优化器"""

    def __init__(self, batch_size: int = 32, max_queue_size: int = 1000):
        self.batch_size = batch_size
        self.max_queue_size = max_queue_size
        self.processing_queue = asyncio.Queue(maxsize=max_queue_size)
        self.is_processing = False

    async def add_request(self, request_data):
        """添加请求到处理队列"""
        await self.processing_queue.put(request_data)

    async def process_batch(self) -> List[Any]:
        """处理一个批量的请求"""
        if self.processing_queue.empty():
            return []

        batch = []

        # 收集批量请求
        for _ in range(min(self.batch_size, self.processing_queue.qsize())):
            try:
                request = asyncio.wait_for(
                    self.processing_queue.get(),
                    timeout=0.001  # 1ms超时，避免阻塞
                )
                batch.append(request)
            except asyncio.TimeoutError:
                break

        if not batch:
            return []

        self.is_processing = True

        try:
            # 批量处理
            results = await self._execute_batch(batch)
            return results

        finally:
            self.is_processing = False

    async def _execute_batch(self, batch: List[Any]) -> List[Any]:
        """执行批量推理"""

        # 准备批量输入
        batch_inputs = []
        max_length = max(len(req['input']) if 'input' in req else 0 for req in batch)

        for req in batch:
            # 填充到相同长度
            input_data = req['input'] + [0] * (max_length - len(req['input']))
            batch_inputs.append(input_data)

        # 批量编码
        batch_tokens = [tokenizer.encode(input_data) for input_data in batch_inputs]
        batch_tensor = torch.tensor(batch_tokens, dtype=torch.long)

        if torch.cuda.is_available():
            batch_tensor = batch_tensor.cuda()

        # 批量推理
        with torch.no_grad():
            batch_outputs = model(batch_tensor)

        # 处理批量结果
        results = []
        for i, (req, output_tokens) in enumerate(zip(batch, batch_outputs)):
            # 截断到原始长度
            output = output_tokens[:len(req['input'])] if 'input' in req else []
            results.append(output)

        return results

    async def start_processing(self):
        """启动批量处理循环"""
        while True:
            results = await self.process_batch()
            if results:
                # 处理结果
                for result in results:
                    await self._handle_result(result)

# 性能对比测试
async def benchmark_batch_sizes():
    """测试不同批量大小的性能"""

    batch_sizes = [1, 4, 8, 16, 32, 64, 128]
    test_requests = [f"test request {i}" for i in range(1000)]

    results = {}

    for batch_size in batch_sizes:
        processor = BatchProcessor(batch_size=batch_size)

        # 添加所有请求
        for request in test_requests:
            await processor.add_request({'input': request})

        # 计时处理
        start_time = time.perf_counter()

        processed_count = 0
        while processed_count < len(test_requests):
            batch_results = await processor.process_batch()
            processed_count += len(batch_results)

        end_time = time.perf_counter()
        total_time = end_time - start_time
        throughput = len(test_requests) / total_time

        results[batch_size] = {
            'throughput_req_per_sec': throughput,
            'total_time_sec': total_time,
            'avg_batch_time_ms': total_time * 1000 / (len(test_requests) // batch_size)
        }

    print("批量大小性能对比:")
    for batch_size, result in results.items():
        print(f"批量大小 {batch_size}: {result['throughput_req_per_sec']:.2f} req/s")

    return results
```

## 实时监控与告警

### 智能告警系统

```python
# alerting_system.py
import asyncio
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Callable
import json

class SmartAlerter:
    """智能告警系统"""

    def __init__(self, config_path: str):
        self.config = self.load_config(config_path)
        self.alert_handlers = {
            'email': self._send_email_alert,
            'slack': self._send_slack_alert,
            'webhook': self._send_webhook_alert
        }

        # 告警状态跟踪
        self.alert_states = {}
        self.last_alert_times = {}

    def load_config(self, config_path: str) -> Dict:
        """加载告警配置"""
        with open(config_path, 'r') as f:
            return json.load(f)

    async def check_metric(self, metric_name: str, current_value: float):
        """检查指标并触发告警"""

        if metric_name not in self.config['thresholds']:
            return

        threshold = self.config['thresholds'][metric_name]

        # 检查告警条件
        should_alert = False
        alert_reason = ""

        if current_value > threshold['critical']:
            should_alert = True
            alert_reason = f"{metric_name} 严重告警: {current_value:.2f} > {threshold['critical']}"
            alert_level = 'critical'
        elif current_value > threshold['warning']:
            should_alert = True
            alert_reason = f"{metric_name} 警告: {current_value:.2f} > {threshold['warning']}"
            alert_level = 'warning'

        # 检查告警间隔
        if should_alert:
            current_time = time.time()
            last_alert_time = self.last_alert_times.get(metric_name, 0)

            # 在冷却期内不重复告警
            cooldown = self.config['cooldown'].get(metric_name, 300)  # 默认5分钟
            if current_time - last_alert_time < cooldown:
                return

            self.last_alert_times[metric_name] = current_time

            # 发送告警
            await self._send_alert(metric_name, current_value, alert_reason, alert_level)

    async def _send_alert(self, metric_name: str, value: float, reason: str, level: str):
        """发送告警"""

        alert_data = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'metric': metric_name,
            'value': value,
            'reason': reason,
            'level': level,
            'service': 'NanoChat'
        }

        # 并行发送多种告警方式
        if 'channels' in self.config:
            tasks = []

            for channel in self.config['channels']:
                if channel in self.alert_handlers:
                    task = self.alert_handlers[channel](alert_data)
                    tasks.append(task)

            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

    def _send_email_alert(self, alert_data: Dict) -> bool:
        """发送邮件告警"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.config['email']['from']
            msg['To'] = ', '.join(self.config['email']['to'])
            msg['Subject'] = f"[{alert_data['level'].upper()}] NanoChat告警"

            body = f"""
            服务: {alert_data['service']}
            时间: {alert_data['timestamp']}
            指标: {alert_data['metric']}
            当前值: {alert_data['value']}
            告警级别: {alert_data['level']}
            原因: {alert_data['reason']}
            """

            msg.attach(MIMEText(body, 'plain'))

            server = smtplib.SMTP(
                self.config['email']['smtp_server'],
                self.config['email']['smtp_port']
            )

            server.starttls()
            server.login(
                self.config['email']['username'],
                self.config['email']['password']
            )

            server.send_message(msg)
            server.quit()
            return True

        except Exception as e:
            print(f"邮件发送失败: {e}")
            return False

    def _send_slack_alert(self, alert_data: Dict) -> bool:
        """发送Slack告警"""
        import aiohttp

        try:
            webhook_url = self.config['slack']['webhook_url']

            slack_message = {
                'text': f"🚨 *{alert_data['level'].upper()}* NanoChat告警\n"
                        f"*指标*: {alert_data['metric']}\n"
                        f"*当前值*: {alert_data['value']}\n"
                        f"*原因*: {alert_data['reason']}",
                'username': 'NanoChat-Monitor'
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=slack_message) as response:
                    return response.status == 200

        except Exception as e:
            print(f"Slack告警发送失败: {e}")
            return False

# 使用示例
alerter = SmartAlerter('config/monitoring_config.json')

# 模拟监控指标检查
async def monitor_system():
    """系统监控主循环"""
    while True:
        # 检查GPU使用率
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024**2  # GB
            gpu_util = gpu_memory / 16.0  # 假设16GB GPU内存
            await alerter.check_metric('gpu_utilization', gpu_util * 100)

        # 检查错误率
        error_rate = 0.05  # 模拟5%错误率
        await alerter.check_metric('error_rate', error_rate * 100)

        # 检查响应时间
        response_time = 350  # 模拟350ms平均响应时间
        await alerter.check_metric('response_time', response_time)

        # 等待下一次检查
        await asyncio.sleep(30)  # 每30秒检查一次

# 启动监控系统
asyncio.run(monitor_system())
```

## 总结

NanoChat的性能优化策略涵盖了AI模型服务的各个层面：

1. **推理优化**：模型量化、KV缓存优化、批量处理
2. **内存优化**：内存池管理、垃圾回收、内存监控
3. **I/O优化**：网络I/O、存储I/O、数据库连接优化
4. **监控告警**：实时性能监控、智能告警、自动化响应

通过这些优化策略，NanoChat能够在有限的硬件资源下实现更好的性能表现，为用户提供更流畅的服务体验。

## 下一步

在最后一篇文章中，我们将总结NanoChat项目的技术亮点、学习价值和未来发展方向。

---

**第十二篇文章预告**：《NanoChat深入解析(12)：大语言模型开发学习路径》将完整总结项目经验和技术要点。