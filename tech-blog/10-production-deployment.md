# NanoChat深入解析(10)：生产环境部署最佳实践

## 前言

将AI模型从开发环境部署到生产环境是一个复杂的过程，需要考虑性能、可靠性、安全性等多个方面。NanoChat作为一个小型但功能完整的ChatGPT克隆，其部署策略对理解大模型服务的生产部署具有很好的参考价值。本文将深入解析NanoChat的生产环境部署最佳实践。

## 部署架构设计

### 单机部署架构

```
┌─────────────────────────────────────────────────────────────┐
│                        单机部署架构                          │
├─────────────────────────────────────────────────────────────┤
│  Nginx (反向代理)                                           │
│  ├── SSL终止                                                │
│  ├── 静态文件服务                                            │
│  └── 负载均衡 (多进程)                                        │
├─────────────────────────────────────────────────────────────┤
│  FastAPI应用实例 × N                                         │
│  ├── Worker Pool (多GPU)                                    │
│  ├── 请求路由                                               │
│  └── 响应流处理                                              │
├─────────────────────────────────────────────────────────────┤
│  GPU推理引擎                                                 │
│  ├── 模型加载                                               │
│  ├── KV缓存                                                 │
│  └── 推理执行                                               │
├─────────────────────────────────────────────────────────────┤
│  监控与日志                                                  │
│  ├── Prometheus + Grafana                                   │
│  ├── 日志收集                                               │
│  └── 健康检查                                               │
└─────────────────────────────────────────────────────────────┘
```

### 分布式部署架构

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   负载均衡器      │    │   API网关       │    │   CDN/边缘节点   │
│   (HAProxy)      │    │   (Kong/Envoy)  │    │   (CloudFlare)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
    ┌─────────────────────────────────────────────────────────────────┐
    │                    微服务集群                                  │
    ├─────────────────────────────────────────────────────────────────┤
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐           │
    │  │  Web服务    │  │  推理服务    │  │  管理服务    │           │
    │  │  (FastAPI)  │  │  (vLLM/TGI) │  │  (Admin)    │           │
    │  └─────────────┘  └─────────────┘  └─────────────┘           │
    ├─────────────────────────────────────────────────────────────────┤
    │                      数据层                                    │
    ├─────────────────────────────────────────────────────────────────┤
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐           │
    │  │  缓存层     │  │  消息队列    │  │  数据库      │           │
    │  │  (Redis)    │  │  (RabbitMQ)  │  │  (PostgreSQL)│           │
    │  └─────────────┘  └─────────────┘  └─────────────┘           │
    ├─────────────────────────────────────────────────────────────────┤
    │                    监控层                                      │
    ├─────────────────────────────────────────────────────────────────┤
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐           │
    │  │  指标收集    │  │  日志聚合    │  │  告警系统    │           │
    │  │ (Prometheus)│  │ (ELK Stack) │  │ (AlertMgr)  │           │
    │  └─────────────┘  └─────────────┘  └─────────────┘           │
    └─────────────────────────────────────────────────────────────────┘
```

## 容器化部署

### Dockerfile优化

```dockerfile
# 多阶段构建优化镜像大小
FROM python:3.10-slim as base

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# 安装Python依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 生产阶段
FROM base as production

# 创建非root用户
RUN useradd -m -u 1000 nanochat

# 复制应用代码
COPY --chown=nanochat:nanochat . .

# 设置权限
RUN chmod +x scripts/*.py

# 切换到非root用户
USER nanochat

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["python", "-m", "scripts.chat_web", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose配置

```yaml
# docker-compose.yml
version: '3.8'

services:
  nanochat:
    build:
      context: .
      dockerfile: Dockerfile
      target: production
    ports:
      - "8000:8000"
    environment:
      - PYTHONPATH=/app
      - CUDA_VISIBLE_DEVICES=0
    volumes:
      - ./models:/app/models:ro
      - ./logs:/app/logs
      - ./cache:/app/.cache
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
      - ./static:/usr/share/nginx/html:ro
    depends_on:
      - nanochat
    restart: unless-stopped

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=200h'
      - '--web.enable-lifecycle'
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards:ro
      - ./grafana/datasources:/etc/grafana/provisioning/datasources:ro
    depends_on:
      - prometheus
    restart: unless-stopped

volumes:
  redis_data:
  prometheus_data:
  grafana_data:

networks:
  default:
    driver: bridge
```

### Kubernetes部署

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nanochat
  labels:
    app: nanochat
spec:
  replicas: 2
  selector:
    matchLabels:
      app: nanochat
  template:
    metadata:
      labels:
        app: nanochat
    spec:
      containers:
      - name: nanochat
        image: nanochat:latest
        ports:
        - containerPort: 8000
        env:
        - name: CUDA_VISIBLE_DEVICES
          value: "0"
        resources:
          requests:
            nvidia.com/gpu: 1
            memory: "8Gi"
            cpu: "4"
          limits:
            nvidia.com/gpu: 1
            memory: "16Gi"
            cpu: "8"
        volumeMounts:
        - name: model-volume
          mountPath: /app/models
          readOnly: true
        - name: cache-volume
          mountPath: /app/.cache
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
      volumes:
      - name: model-volume
        persistentVolumeClaim:
          claimName: model-pvc
      - name: cache-volume
        emptyDir:
          sizeLimit: 10Gi
      nodeSelector:
        accelerator: nvidia-tesla-v100  # 选择GPU节点

---
apiVersion: v1
kind: Service
metadata:
  name: nanochat-service
spec:
  selector:
    app: nanochat
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer

---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: nanochat-ingress
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - nanochat.example.com
    secretName: nanochat-tls
  rules:
  - host: nanochat.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: nanochat-service
            port:
              number: 80
```

## 负载均衡策略

### Nginx配置

```nginx
# nginx.conf
user nginx;
worker_processes auto;
error_log /var/log/nginx/error.log warn;
pid /var/run/nginx.pid;

events {
    worker_connections 1024;
    use epoll;
    multi_accept on;
}

http {
    include /etc/nginx/mime.types;
    default_type application/octet-stream;

    # 日志格式
    log_format main '$remote_addr - $remote_user [$time_local] "$request" '
                    '$status $body_bytes_sent "$http_referer" '
                    '"$http_user_agent" "$http_x_forwarded_for" '
                    'rt=$request_time uct="$upstream_connect_time" '
                    'uht="$upstream_header_time" urt="$upstream_response_time"';

    access_log /var/log/nginx/access.log main;

    # 基础配置
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    types_hash_max_size 2048;

    # Gzip压缩
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types
        text/plain
        text/css
        text/xml
        text/javascript
        application/javascript
        application/xml+rss
        application/json;

    # 上游服务器配置
    upstream nanochat_backend {
        least_conn;
        server nanochat:8000 max_fails=3 fail_timeout=30s;
        # 可以添加多个实例
        # server nanochat-2:8000 max_fails=3 fail_timeout=30s;

        # 连接池配置
        keepalive 32;
    }

    # 限流配置
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_conn_zone $binary_remote_addr zone=conn_limit_per_ip:10m;

    server {
        listen 80;
        server_name nanochat.example.com;

        # 重定向到HTTPS
        return 301 https://$server_name$request_uri;
    }

    server {
        listen 443 ssl http2;
        server_name nanochat.example.com;

        # SSL配置
        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384;
        ssl_prefer_server_ciphers off;
        ssl_session_cache shared:SSL:10m;
        ssl_session_timeout 10m;

        # 安全头
        add_header X-Frame-Options DENY;
        add_header X-Content-Type-Options nosniff;
        add_header X-XSS-Protection "1; mode=block";
        add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;

        # 连接限制
        limit_conn conn_limit_per_ip 10;

        # 静态文件服务
        location /static/ {
            alias /usr/share/nginx/html/;
            expires 1y;
            add_header Cache-Control "public, immutable";
        }

        # API代理
        location / {
            # 限流
            limit_req zone=api burst=20 nodelay;

            # 代理配置
            proxy_pass http://nanochat_backend;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection 'upgrade';
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_cache_bypass $http_upgrade;

            # 超时配置
            proxy_connect_timeout 30s;
            proxy_send_timeout 60s;
            proxy_read_timeout 60s;

            # 缓冲配置
            proxy_buffering on;
            proxy_buffer_size 4k;
            proxy_buffers 8 4k;
        }

        # 健康检查
        location /health {
            access_log off;
            proxy_pass http://nanochat_backend/health;
        }

        # 监控端点（仅内部访问）
        location /metrics {
            allow 127.0.0.1;
            allow 10.0.0.0/8;
            deny all;
            proxy_pass http://nanochat_backend/stats;
        }
    }
}
```

### HAProxy配置

```haproxy
# haproxy.cfg
global
    daemon
    maxconn 4096
    log stdout format raw local0

defaults
    mode http
    timeout connect 10s
    timeout client 60s
    timeout server 60s
    timeout http-request 10s
    timeout http-keep-alive 2s
    option httplog
    option dontlognull

# 统计页面
listen stats
    bind *:8404
    stats enable
    stats uri /stats
    stats refresh 30s

# 前端
frontend nanochat_frontend
    bind *:80
    bind *:443 ssl crt /etc/ssl/certs/nanochat.pem
    redirect scheme https if !{ ssl_fc }

    # ACL规则
    acl is_websocket hdr(upgrade) -i websocket
    acl is_api path_beg /api/

    # 路由规则
    use_backend nanochat_api if is_api
    use_backend nanochat_websocket if is_websocket
    default_backend nanochat_web

# 后端API
backend nanochat_api
    balance roundrobin
    option httpchk GET /health
    server api1 nanochat1:8000 check
    server api2 nanochat2:8000 check

    # 限流配置
    http-request deny status 429 if { src_conn_cnt ge 100 }

# 后端WebSocket
backend nanochat_websocket
    balance roundrobin
    server ws1 nanochat1:8000 check
    server ws2 nanochat2:8000 check

# 后端Web
backend nanochat_web
    balance roundrobin
    server web1 nanochat1:8000 check
    server web2 nanochat2:8000 check
```

## 监控与日志

### Prometheus指标收集

```python
# metrics.py
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time
import psutil
import torch

# 定义指标
REQUEST_COUNT = Counter(
    'nanochat_requests_total',
    'Total number of requests',
    ['method', 'endpoint', 'status']
)

REQUEST_DURATION = Histogram(
    'nanochat_request_duration_seconds',
    'Request duration in seconds',
    ['method', 'endpoint']
)

ACTIVE_CONNECTIONS = Gauge(
    'nanochat_active_connections',
    'Number of active connections'
)

GPU_MEMORY_USAGE = Gauge(
    'nanochat_gpu_memory_bytes',
    'GPU memory usage in bytes',
    ['gpu_id']
)

GPU_UTILIZATION = Gauge(
    'nanochat_gpu_utilization_percent',
    'GPU utilization percentage',
    ['gpu_id']
)

MODEL_LOAD_TIME = Histogram(
    'nanochat_model_load_duration_seconds',
    'Model loading time in seconds'
)

class MetricsCollector:
    """指标收集器"""

    def __init__(self):
        self.start_metrics_server()

    def start_metrics_server(self):
        """启动指标服务器"""
        start_http_server(8001)

    def record_request(self, method, endpoint, status, duration):
        """记录请求指标"""
        REQUEST_COUNT.labels(
            method=method,
            endpoint=endpoint,
            status=status
        ).inc()

        REQUEST_DURATION.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)

    def update_connection_count(self, count):
        """更新连接数"""
        ACTIVE_CONNECTIONS.set(count)

    def update_gpu_metrics(self):
        """更新GPU指标"""
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                # GPU内存使用
                memory_used = torch.cuda.memory_allocated(i)
                GPU_MEMORY_USAGE.labels(gpu_id=i).set(memory_used)

                # GPU利用率 (需要nvidia-ml-py)
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    GPU_UTILIZATION.labels(gpu_id=i).set(util.gpu)
                except ImportError:
                    pass

    def record_model_load_time(self, duration):
        """记录模型加载时间"""
        MODEL_LOAD_TIME.observe(duration)
```

### 日志管理

```python
# logging_config.py
import logging
import logging.config
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    """JSON格式日志格式化器"""

    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }

        # 添加异常信息
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)

        # 添加额外字段
        if hasattr(record, 'request_id'):
            log_entry['request_id'] = record.request_id

        if hasattr(record, 'user_id'):
            log_entry['user_id'] = record.user_id

        if hasattr(record, 'duration'):
            log_entry['duration'] = record.duration

        return json.dumps(log_entry)

# 日志配置
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'json': {
            '()': JSONFormatter,
        },
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'standard',
            'stream': 'ext://sys.stdout'
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'DEBUG',
            'formatter': 'json',
            'filename': '/app/logs/nanochat.log',
            'maxBytes': 100*1024*1024,  # 100MB
            'backupCount': 5
        },
        'error_file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'ERROR',
            'formatter': 'json',
            'filename': '/app/logs/nanochat-error.log',
            'maxBytes': 50*1024*1024,  # 50MB
            'backupCount': 3
        }
    },
    'loggers': {
        '': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': False
        },
        'nanochat': {
            'handlers': ['console', 'file', 'error_file'],
            'level': 'DEBUG',
            'propagate': False
        }
    }
}

def setup_logging():
    """设置日志配置"""
    logging.config.dictConfig(LOGGING_CONFIG)
```

### Grafana仪表板

```json
{
  "dashboard": {
    "title": "NanoChat监控仪表板",
    "panels": [
      {
        "title": "请求速率",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(nanochat_requests_total[5m])",
            "legendFormat": "{{method}} {{endpoint}}"
          }
        ]
      },
      {
        "title": "请求延迟",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.50, nanochat_request_duration_seconds_bucket)",
            "legendFormat": "P50"
          },
          {
            "expr": "histogram_quantile(0.95, nanochat_request_duration_seconds_bucket)",
            "legendFormat": "P95"
          },
          {
            "expr": "histogram_quantile(0.99, nanochat_request_duration_seconds_bucket)",
            "legendFormat": "P99"
          }
        ]
      },
      {
        "title": "GPU使用情况",
        "type": "graph",
        "targets": [
          {
            "expr": "nanochat_gpu_utilization_percent",
            "legendFormat": "GPU {{gpu_id}} 利用率"
          },
          {
            "expr": "nanochat_gpu_memory_bytes / 1024^3",
            "legendFormat": "GPU {{gpu_id}} 内存(GB)"
          }
        ]
      },
      {
        "title": "活跃连接数",
        "type": "singlestat",
        "targets": [
          {
            "expr": "nanochat_active_connections"
          }
        ]
      }
    ]
  }
}
```

## 安全配置

### 应用安全

```python
# security.py
from fastapi import HTTPException, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from datetime import datetime, timedelta
import bcrypt
import secrets

security = HTTPBearer()

class SecurityManager:
    """安全管理器"""

    def __init__(self, secret_key: str):
        self.secret_key = secret_key
        self.algorithm = "HS256"
        self.access_token_expire_minutes = 30

    def create_access_token(self, data: dict):
        """创建访问令牌"""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt

    def verify_token(self, token: str):
        """验证令牌"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.PyJWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="无效的认证令牌",
                headers={"WWW-Authenticate": "Bearer"},
            )

    def hash_password(self, password: str) -> str:
        """哈希密码"""
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')

    def verify_password(self, password: str, hashed: str) -> bool:
        """验证密码"""
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

# 安全中间件
async def security_middleware(request, call_next):
    """安全中间件"""

    # 添加安全头
    response = await call_next(request)

    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

    return response

# 输入验证
from pydantic import validator
import re

class SecurityValidator:
    """安全验证器"""

    @staticmethod
    def validate_input(text: str) -> str:
        """验证输入文本"""
        # 检查恶意内容
        malicious_patterns = [
            r'<script[^>]*>.*?</script>',
            r'javascript:',
            r'on\w+\s*=',
            r'eval\s*\(',
            r'document\.',
            r'window\.',
        ]

        for pattern in malicious_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                raise HTTPException(
                    status_code=400,
                    detail="输入包含不安全内容"
                )

        # 长度限制
        if len(text) > 8000:
            raise HTTPException(
                status_code=400,
                detail="输入过长"
            )

        return text

    @staticmethod
    def sanitize_output(text: str) -> str:
        """清理输出文本"""
        # 移除潜在的HTML标签
        import html
        return html.escape(text)
```

## 性能优化

### 缓存策略

```python
# cache.py
import redis
import json
import pickle
from typing import Any, Optional
from functools import wraps

class CacheManager:
    """缓存管理器"""

    def __init__(self, redis_url: str):
        self.redis_client = redis.from_url(redis_url)
        self.default_ttl = 3600  # 1小时

    def get(self, key: str) -> Optional[Any]:
        """获取缓存"""
        try:
            data = self.redis_client.get(key)
            if data:
                return pickle.loads(data)
        except Exception as e:
            print(f"缓存获取失败: {e}")
        return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """设置缓存"""
        try:
            ttl = ttl or self.default_ttl
            data = pickle.dumps(value)
            self.redis_client.setex(key, ttl, data)
        except Exception as e:
            print(f"缓存设置失败: {e}")

    def delete(self, key: str):
        """删除缓存"""
        try:
            self.redis_client.delete(key)
        except Exception as e:
            print(f"缓存删除失败: {e}")

def cache_result(ttl: int = 3600, key_prefix: str = ""):
    """缓存装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 生成缓存键
            cache_key = f"{key_prefix}:{func.__name__}:{hash(str(args) + str(kwargs))}"

            # 尝试从缓存获取
            cache_manager = get_cache_manager()
            cached_result = cache_manager.get(cache_key)
            if cached_result is not None:
                return cached_result

            # 执行函数并缓存结果
            result = func(*args, **kwargs)
            cache_manager.set(cache_key, result, ttl)
            return result

        return wrapper
    return decorator

# 全局缓存管理器实例
_cache_manager = None

def get_cache_manager() -> CacheManager:
    """获取缓存管理器实例"""
    global _cache_manager
    if _cache_manager is None:
        redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
        _cache_manager = CacheManager(redis_url)
    return _cache_manager
```

### 连接池管理

```python
# connection_pool.py
import asyncio
import aiohttp
from contextlib import asynccontextmanager

class ConnectionPoolManager:
    """连接池管理器"""

    def __init__(self):
        self.http_session = None
        self.db_pool = None

    async def initialize(self):
        """初始化连接池"""
        # HTTP连接池
        connector = aiohttp.TCPConnector(
            limit=100,  # 总连接数限制
            limit_per_host=20,  # 每个主机连接数限制
            ttl_dns_cache=300,  # DNS缓存时间
            use_dns_cache=True,
        )

        timeout = aiohttp.ClientTimeout(total=30, connect=10)

        self.http_session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout
        )

        # 数据库连接池
        # 根据实际数据库类型配置

    async def cleanup(self):
        """清理连接池"""
        if self.http_session:
            await self.http_session.close()
        # 清理数据库连接池

    @asynccontextmanager
    async def get_http_session(self):
        """获取HTTP会话"""
        if not self.http_session:
            await self.initialize()
        yield self.http_session

# 全局连接池管理器
_pool_manager = ConnectionPoolManager()

async def get_pool_manager() -> ConnectionPoolManager:
    """获取连接池管理器"""
    return _pool_manager
```

## 故障恢复

### 健康检查系统

```python
# health.py
import asyncio
import torch
import psutil
from fastapi import HTTPException
from typing import Dict, Any

class HealthChecker:
    """健康检查器"""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    async def check_health(self) -> Dict[str, Any]:
        """全面健康检查"""
        health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "checks": {}
        }

        checks = [
            ("model", self.check_model),
            ("gpu", self.check_gpu),
            ("memory", self.check_memory),
            ("disk", self.check_disk),
        ]

        for check_name, check_func in checks:
            try:
                result = await check_func()
                health_status["checks"][check_name] = result
                if result["status"] != "healthy":
                    health_status["status"] = "unhealthy"
            except Exception as e:
                health_status["checks"][check_name] = {
                    "status": "error",
                    "message": str(e)
                }
                health_status["status"] = "unhealthy"

        return health_status

    async def check_model(self) -> Dict[str, Any]:
        """检查模型状态"""
        try:
            # 简单推理测试
            test_input = torch.tensor([[1, 2, 3]], dtype=torch.long)
            if torch.cuda.is_available():
                test_input = test_input.cuda()

            with torch.no_grad():
                output = self.model(test_input)

            return {
                "status": "healthy",
                "message": "模型推理正常",
                "output_shape": list(output.shape)
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "message": f"模型推理失败: {str(e)}"
            }

    async def check_gpu(self) -> Dict[str, Any]:
        """检查GPU状态"""
        if not torch.cuda.is_available():
            return {
                "status": "healthy",
                "message": "未使用GPU"
            }

        try:
            gpu_info = []
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                memory_total = props.total_memory
                memory_used = torch.cuda.memory_allocated(i)
                memory_free = memory_total - memory_used

                gpu_info.append({
                    "id": i,
                    "name": props.name,
                    "memory_total_gb": memory_total / 1024**3,
                    "memory_used_gb": memory_used / 1024**3,
                    "memory_free_gb": memory_free / 1024**3,
                    "memory_utilization": memory_used / memory_total
                })

            # 检查内存使用率
            max_utilization = max(info["memory_utilization"] for info in gpu_info)
            status = "healthy" if max_utilization < 0.9 else "warning"

            return {
                "status": status,
                "message": f"GPU状态正常，最高内存使用率: {max_utilization:.1%}",
                "gpus": gpu_info
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"GPU检查失败: {str(e)}"
            }

    async def check_memory(self) -> Dict[str, Any]:
        """检查内存使用"""
        memory = psutil.virtual_memory()
        used_percent = memory.percent

        status = "healthy"
        if used_percent > 90:
            status = "critical"
        elif used_percent > 80:
            status = "warning"

        return {
            "status": status,
            "message": f"内存使用率: {used_percent:.1f}%",
            "total_gb": memory.total / 1024**3,
            "used_gb": memory.used / 1024**3,
            "available_gb": memory.available / 1024**3
        }

    async def check_disk(self) -> Dict[str, Any]:
        """检查磁盘空间"""
        disk = psutil.disk_usage('/')
        used_percent = disk.percent

        status = "healthy"
        if used_percent > 90:
            status = "critical"
        elif used_percent > 80:
            status = "warning"

        return {
            "status": status,
            "message": f"磁盘使用率: {used_percent:.1f}%",
            "total_gb": disk.total / 1024**3,
            "used_gb": disk.used / 1024**3,
            "free_gb": disk.free / 1024**3
        }

# 应用健康检查端点
@app.get("/health")
async def health_check():
    """健康检查端点"""
    health_checker = get_health_checker()
    health_status = await health_checker.check_health()

    if health_status["status"] == "healthy":
        return health_status
    else:
        raise HTTPException(status_code=503, detail=health_status)
```

## 部署自动化

### CI/CD流水线

```yaml
# .github/workflows/deploy.yml
name: Deploy NanoChat

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'

    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov

    - name: Run tests
      run: |
        pytest tests/ --cov=nanochat --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v3

  build:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'

    steps:
    - uses: actions/checkout@v3

    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v2

    - name: Login to Container Registry
      uses: docker/login-action@v2
      with:
        registry: ghcr.io
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}

    - name: Build and push Docker image
      uses: docker/build-push-action@v4
      with:
        context: .
        push: true
        tags: |
          ghcr.io/${{ github.repository }}:latest
          ghcr.io/${{ github.repository }}:${{ github.sha }}
        cache-from: type=gha
        cache-to: type=gha,mode=max

  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'

    steps:
    - name: Deploy to production
      uses: appleboy/ssh-action@v0.1.5
      with:
        host: ${{ secrets.PROD_HOST }}
        username: ${{ secrets.PROD_USER }}
        key: ${{ secrets.PROD_SSH_KEY }}
        script: |
          cd /opt/nanochat
          docker-compose pull
          docker-compose up -d
          docker system prune -f
```

### 配置管理

```python
# config.py
import os
from typing import Optional
from pydantic import BaseSettings

class Settings(BaseSettings):
    """应用配置"""

    # 应用配置
    app_name: str = "NanoChat"
    app_version: str = "1.0.0"
    debug: bool = False

    # 服务器配置
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1

    # GPU配置
    device_type: str = "cuda"
    num_gpus: int = 1
    model_path: str = "/app/models"

    # 推理配置
    max_tokens: int = 512
    temperature: float = 0.8
    top_k: int = 50

    # 缓存配置
    redis_url: str = "redis://localhost:6379"
    cache_ttl: int = 3600

    # 安全配置
    secret_key: str
    access_token_expire_minutes: int = 30

    # 监控配置
    enable_metrics: bool = True
    metrics_port: int = 8001

    # 日志配置
    log_level: str = "INFO"
    log_file: str = "/app/logs/nanochat.log"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# 全局设置实例
_settings = None

def get_settings() -> Settings:
    """获取设置实例"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
```

## 总结

NanoChat的生产环境部署体现了现代AI服务的最佳实践：

1. **容器化部署**：使用Docker和Kubernetes实现标准化部署
2. **负载均衡**：多层负载均衡确保高可用性
3. **监控告警**：全方位的监控和告警系统
4. **安全防护**：多层安全防护措施
5. **性能优化**：缓存、连接池等性能优化策略
6. **自动化运维**：CI/CD流水线和配置管理

这种部署架构确保了NanoChat能够在生产环境中稳定、高效、安全地运行，为用户提供可靠的服务。

## 下一步

在下一篇文章中，我们将深入分析NanoChat的性能瓶颈，并探讨各种优化方案。

---

**第十一篇文章预告**：《NanoChat深入解析(11)：性能瓶颈分析与优化方案》将详细解析性能瓶颈识别和优化策略。