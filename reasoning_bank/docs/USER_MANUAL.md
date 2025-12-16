# ReasoningBank 用户手册

> 基于论文 *ReasoningBank: Scaling Agent Self-Evolving with Reasoning Memory* 的完整复现实现

---

## 目录

1. [项目概述](#1-项目概述)
2. [安装与配置](#2-安装与配置)
3. [快速开始](#3-快速开始)
4. [核心概念](#4-核心概念)
5. [命令行使用](#5-命令行使用)
6. [API 参考](#6-api-参考)
7. [数据集说明](#7-数据集说明)
8. [记忆库管理](#8-记忆库管理)
9. [MaTTS 测试时扩展](#9-matts-测试时扩展)
10. [高级配置](#10-高级配置)
11. [扩展开发](#11-扩展开发)
12. [常见问题](#12-常见问题)

---

## 1. 项目概述

### 1.1 什么是 ReasoningBank？

ReasoningBank 是一个具备**自我进化能力**的智能体框架。与传统的 AI Agent 不同，它能够：

- **从成功中学习**：提取有效的问题解决策略
- **从失败中学习**：提取"避坑指南"，防止重复错误
- **记忆增强决策**：利用历史经验指导新任务的求解
- **持续进化**：随着任务积累，能力不断提升

### 1.2 核心工作流程

```
┌──────────────────────────────────────────────────────────────────┐
│                      ReasoningBank 工作流程                        │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│   新任务 ──┬──> [检索记忆] ──> [执行任务] ──> [评估结果]             │
│            │                                        │             │
│            │    ┌──────────────────────────────────┘             │
│            │    ▼                                                │
│            │  成功? ──┬─ Yes ──> [提取成功策略] ──┐               │
│            │         └─ No  ──> [提取失败教训] ──┤               │
│            │                                      │               │
│            │                                      ▼               │
│            │                              [存入记忆库]             │
│            │                                      │               │
│            └──────────────────────────────────────┘               │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### 1.3 支持的任务类型

| 类型 | 数据集 | 说明 |
|------|--------|------|
| **单轮 QA** | MATH-500, AIME, GPQA, MMLU-Pro | 数学推理、选择题 |
| **多轮交互** | ALFWorld, ScienceWorld | 文字游戏、科学实验 |

---

## 2. 安装与配置

### 2.1 环境要求

- Python 3.10+
- CUDA（可选，用于 GPU 加速 Embedding）

### 2.2 安装步骤

```bash
# 1. 克隆项目
git clone <repository-url>
cd reasoning-bank

# 2. 创建 conda 环境（推荐）
conda create -n icml26 python=3.10
conda activate icml26

# 3. 安装依赖
pip install -r reasoning_bank/requirements.txt

# 4. 配置 API Key
echo "OPENROUTER_API_KEY=your_api_key_here" > .env
```

### 2.3 配置文件

主配置文件位于 `reasoning_bank/config/config.yaml`：

```yaml
# LLM 服务配置
llm:
  api_base: "https://openrouter.ai/api/v1"
  default_model: "qwen/qwen-2.5-7b-instruct"
  temperature: 0.3
  max_tokens: 4096
  timeout: 120
  max_retries: 3

# Embedding 服务配置
embedding:
  model: "sentence-transformers/all-MiniLM-L6-v2"
  device: "cpu"  # 可选: cpu, cuda

# 记忆库配置
memory:
  top_k: 1  # 检索记忆数量
  storage_path: "./data/memory_banks"
  similarity_threshold: 0.5

# Agent 配置
agent:
  max_steps: 30  # 多轮交互最大步数
  react_format: true

# MaTTS 配置
matts:
  parallel_n: 5
  parallel_temperature: 0.7
  sequential_max_refine: 3
```

### 2.4 环境变量

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `OPENROUTER_API_KEY` | OpenRouter API 密钥 | 必填 |
| `OPENROUTER_API_BASE` | API 地址 | `https://openrouter.ai/api/v1` |

---

## 3. 快速开始

### 3.1 运行首个测试

```bash
# 激活环境
conda activate icml26

# 运行 5 道数学题（无记忆）
python -m reasoning_bank.main --dataset math500 --num-tasks 5

# 使用记忆库运行（启用自我进化）
python -m reasoning_bank.main --dataset math500 --num-tasks 10 --use-memory
```

### 3.2 查看运行结果

运行日志会显示：
- 每个任务的成功/失败状态
- 预测答案与标准答案
- 新增记忆数量
- 最终统计信息

```
2024-01-15 10:30:00 | INFO | [Task math_807] Result: ✅ SUCCESS
2024-01-15 10:30:00 | INFO |   Answer: \left(3, \frac{\pi}{2}\right)
2024-01-15 10:30:00 | INFO |   Ground Truth: \left( 3, \frac{\pi}{2} \right)
...
2024-01-15 10:35:00 | INFO | ==================================================
2024-01-15 10:35:00 | INFO | 进化循环统计
2024-01-15 10:35:00 | INFO | ==================================================
2024-01-15 10:35:00 | INFO | 总任务数: 10
2024-01-15 10:35:00 | INFO | 成功数: 7
2024-01-15 10:35:00 | INFO | 成功率: 70.00%
2024-01-15 10:35:00 | INFO | 新增记忆: 12
```

---

## 4. 核心概念

### 4.1 记忆项 (Memory Item)

记忆是从任务执行中提取的结构化经验：

```json
{
  "id": "mem_abc123",
  "original_query": "问题描述",
  "items": [
    {
      "title": "策略标题",
      "description": "一句话简介",
      "content": "详细的可复用建议"
    }
  ],
  "is_success": true,
  "timestamp": "2024-01-15T10:30:00"
}
```

**记忆类型**：
- **成功记忆**：从正确解答中提取的有效策略
- **失败记忆**：从错误尝试中提取的"避坑指南"

### 4.2 轨迹 (Trajectory)

轨迹记录了 Agent 解决问题的完整过程：

```python
trajectory = [
    {
        "step": 0,
        "observation": "题目内容",
        "thought": "我的思考过程...",
        "action": "我的解答..."
    },
    # ...更多步骤
]
```

### 4.3 ReAct 格式

Agent 使用 ReAct（Reasoning + Acting）范式：

```
THOUGHT: [分析当前情况，规划下一步]
ACTION: [执行具体操作或给出答案]
```

### 4.4 MaTTS (Memory-aware Test-Time Scaling)

通过增加测试时计算来获取更高质量的经验：

- **并行扩展 (Parallel)**：同时生成多条轨迹，对比成功与失败
- **串行扩展 (Sequential)**：自我检查和修正，从纠错中学习
- **组合扩展 (Combined)**：先并行探索，再串行优化

---

## 5. 命令行使用

### 5.1 基本命令

```bash
python -m reasoning_bank.main [选项]
```

### 5.2 完整参数列表

| 参数 | 简写 | 类型 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--dataset` | `-d` | str | `math500` | 数据集名称 |
| `--num-tasks` | `-n` | int | 全部 | 运行任务数量 |
| `--use-memory` | - | flag | False | 启用记忆库 |
| `--no-extract` | - | flag | False | 不提取新记忆 |
| `--clear-memory` | - | flag | False | 清空记忆库后运行 |
| `--model` | `-m` | str | 配置文件 | LLM 模型名称 |
| `--temperature` | `-t` | float | 0.3 | 生成温度 |
| `--matts` | - | str | `none` | MaTTS 模式 |
| `--parallel-n` | - | int | 5 | 并行轨迹数量 |
| `--verbose` | `-v` | flag | False | 详细输出 |
| `--seed` | - | int | 42 | 随机种子 |
| `--shuffle` | - | flag | False | 打乱数据集 |
| `--results-dir` | - | str | `./results` | 结果保存目录 |
| `--config` | - | str | 默认路径 | 配置文件路径 |

### 5.3 使用示例

```bash
# 基础测试（无记忆）
python -m reasoning_bank.main --dataset math500 --num-tasks 50

# 启用自我进化（推荐）
python -m reasoning_bank.main --dataset math500 --num-tasks 100 --use-memory

# 使用更强的模型
python -m reasoning_bank.main --dataset gpqa --model "anthropic/claude-3-sonnet" --use-memory

# MaTTS 并行扩展
python -m reasoning_bank.main --dataset math500 --matts parallel --parallel-n 5 --use-memory

# MaTTS 串行扩展（自我修正）
python -m reasoning_bank.main --dataset math500 --matts sequential --use-memory

# MaTTS 组合扩展（最佳质量）
python -m reasoning_bank.main --dataset math500 --matts combined --use-memory

# 从头开始（清空旧记忆）
python -m reasoning_bank.main --dataset aime24 --use-memory --clear-memory

# 仅使用记忆，不提取新记忆（测试模式）
python -m reasoning_bank.main --dataset math500 --use-memory --no-extract

# 详细输出调试
python -m reasoning_bank.main --dataset math500 --num-tasks 3 --verbose
```

---

## 6. API 参考

### 6.1 MemoryBank

```python
from reasoning_bank.core.memory import MemoryBank

# 创建记忆库
memory_bank = MemoryBank(bank_name="math500")

# 检索记忆
memories = memory_bank.retrieve(
    query="如何求解二次方程？",
    top_k=3
)

# 添加记忆
memory_bank.add(
    query="问题描述",
    items=[{
        "title": "策略标题",
        "description": "简介",
        "content": "详细内容"
    }],
    is_success=True,
    trajectory_id="task_001"
)

# 查看统计
print(memory_bank.get_stats())
# {'total': 50, 'success': 35, 'failure': 15}

# 清空记忆库
memory_bank.clear()
```

### 6.2 ReActAgent

```python
from reasoning_bank.core.agent import ReActAgent, AgentConfig
from reasoning_bank.core.memory import MemoryBank

# 创建配置
config = AgentConfig(
    temperature=0.3,
    max_steps=30,
    verbose=True
)

# 创建 Agent
agent = ReActAgent(
    memory_bank=MemoryBank("math500"),
    config=config
)

# 执行任务
result = agent.run(env, task_id="math_001")

print(f"答案: {result.answer}")
print(f"成功: {result.is_success}")
print(f"步数: {result.steps}")
```

### 6.3 EvolutionLoop

```python
from reasoning_bank.workflows.evolution import EvolutionLoop
from reasoning_bank.envs.single_turn import SingleTurnEnvRegistry
from reasoning_bank.core.memory import MemoryBank

# 创建环境
env = SingleTurnEnvRegistry.create("math500", max_samples=100)

# 创建记忆库
memory_bank = MemoryBank(bank_name="math500")

# 创建进化循环
loop = EvolutionLoop(
    env=env,
    memory_bank=memory_bank,
    extract_memories=True,  # 启用记忆提取
    save_results=True,
)

# 运行
stats = loop.run(num_tasks=50, show_progress=True)

print(f"成功率: {stats.success_rate:.2%}")
print(f"新增记忆: {stats.memories_added}")
```

### 6.4 MaTTSRunner

```python
from reasoning_bank.workflows.matts import MaTTSRunner, MaTTSConfig

# 创建配置
config = MaTTSConfig(
    parallel_n=5,
    parallel_temperature=0.7,
    sequential_max_refine=3
)

# 创建 Runner
runner = MaTTSRunner(
    env=env,
    memory_bank=memory_bank,
    config=config
)

# 并行扩展
result, memories = runner.run_parallel(task_id)

# 串行扩展
result, memories = runner.run_sequential(task_id)

# 组合扩展
result, memories = runner.run_combined(task_id)
```

### 6.5 LLMService

```python
from reasoning_bank.core.llm_service import LLMService

# 创建服务
llm = LLMService(
    model="qwen/qwen-2.5-7b-instruct",
    temperature=0.3
)

# 同步调用
result = llm.call(
    prompt="请解释量子计算的基本原理",
    system_prompt="你是一个物理学专家",
    max_tokens=1000
)

print(result.content)
print(f"状态: {result.status}")
print(f"Token: {result.usage}")

# 异步调用
import asyncio

async def main():
    result = await llm.call_async(prompt="...")
    return result

asyncio.run(main())
```

---

## 7. 数据集说明

### 7.1 支持的数据集

| 数据集 | 类型 | 数量 | 说明 |
|--------|------|------|------|
| `math500` | 数学 | 500 | MATH 数据集抽样 |
| `aime24` | 数学竞赛 | 30 | AIME 2024 |
| `aime25` | 数学竞赛 | 30 | AIME 2025 |
| `gpqa` | 研究生级 QA | 198 | GPQA Diamond |
| `mmlu_economics` | 选择题 | 844 | MMLU-Pro 经济学 |
| `mmlu_engineering` | 选择题 | 969 | MMLU-Pro 工程学 |
| `mmlu_philosophy` | 选择题 | 499 | MMLU-Pro 哲学 |

### 7.2 数据格式

单轮 QA 数据集使用统一的 JSONL 格式：

```json
{
  "id": "math_001",
  "question": "问题内容...",
  "solution": "详细解答过程（可选）",
  "answer": "最终答案",
  "options": ["A. ...", "B. ..."],  // 选择题专用
  "metadata": {
    "source": "MATH",
    "difficulty": "Level 4",
    "topic": "algebra"
  }
}
```

### 7.3 添加自定义数据集

1. 准备 JSONL 文件，放入 `bench/single_turn_bench/`
2. 在 `SingleTurnEnvRegistry.DATASETS` 中注册：

```python
# reasoning_bank/envs/single_turn.py

DATASETS = {
    # ... 现有数据集
    "my_dataset": {
        "file": "my_dataset.jsonl",
        "type": "math",  # 或 "mcq"
        "description": "我的自定义数据集"
    }
}
```

---

## 8. 记忆库管理

### 8.1 记忆库存储结构

```
data/memory_banks/
├── math500/
│   ├── memories.jsonl      # 记忆数据
│   └── embeddings.npy      # 向量缓存
├── gpqa/
│   └── ...
└── aime24/
    └── ...
```

### 8.2 记忆检索原理

```
用户问题 ──> Embedding ──> 余弦相似度 ──> Top-K 记忆
                              │
                              ▼
                      [相似度阈值过滤]
                              │
                              ▼
                      注入到 System Prompt
```

### 8.3 记忆格式示例

```json
{
  "id": "mem_20240115_abc123",
  "original_query": "Convert the point (0,3) from rectangular to polar coordinates.",
  "items": [
    {
      "title": "Polar Coordinates Conversion Strategy",
      "description": "Use r = √(x² + y²) and θ = arctan(y/x) formulas",
      "content": "To convert any point (x, y) from rectangular to polar coordinates: 1) Calculate r using the distance formula. 2) Determine θ using arctangent, adjusting for the correct quadrant based on the signs of x and y. 3) Ensure r > 0 and 0 ≤ θ < 2π."
    }
  ],
  "is_success": true,
  "trajectory_id": "math_807",
  "timestamp": "2024-01-15T10:30:00"
}
```

### 8.4 记忆提取过程

**成功时**：提取通用策略
```
"从成功解答中提取可复用的策略，不包含具体数值"
```

**失败时**：提取避坑指南
```
"分析错误原因，提取预防性建议，帮助避免类似错误"
```

---

## 9. MaTTS 测试时扩展

### 9.1 并行扩展 (Parallel Scaling)

```
同一问题 ──┬──> Agent 1 ──> 成功 ─┐
           ├──> Agent 2 ──> 失败 ─┤
           ├──> Agent 3 ──> 成功 ─┼──> 对比分析 ──> 高质量记忆
           ├──> Agent 4 ──> 失败 ─┤
           └──> Agent 5 ──> 成功 ─┘
```

**优势**：
- 通过对比识别成功的关键因素
- 过滤偶然成功，提取稳定策略
- 从多样化失败中学习

### 9.2 串行扩展 (Sequential Scaling)

```
初次尝试 ──> 自我检查 ──> 发现错误？ ─┬─ Yes ──> 修正 ──> 提取纠错经验
                                      └─ No  ──> 确认答案
```

**优势**：
- 捕获"纠错时刻"的宝贵经验
- 学习如何自我发现和修正错误
- 记忆包含"哪里容易出错"的信息

### 9.3 组合扩展 (Combined)

```
并行探索 (广度) ──> 选择最佳轨迹 ──> 串行优化 (深度) ──> 最终结果
```

### 9.4 使用建议

| 场景 | 推荐模式 | 原因 |
|------|----------|------|
| 快速测试 | `none` | 最少计算资源 |
| 日常进化 | `parallel` | 平衡质量与效率 |
| 困难任务 | `sequential` | 深度探索 |
| 构建高质量记忆库 | `combined` | 最佳记忆质量 |

---

## 10. 高级配置

### 10.1 模型选择指南

| 模型 | 适用场景 | 成本 |
|------|----------|------|
| `qwen/qwen-2.5-7b-instruct` | 日常测试 | 低 |
| `deepseek/deepseek-chat` | 数学推理 | 中 |
| `anthropic/claude-3-sonnet` | 复杂任务 | 高 |
| `openai/gpt-4o` | 最佳质量 | 高 |

### 10.2 温度设置

| 温度 | 效果 | 适用场景 |
|------|------|----------|
| 0.0-0.3 | 确定性高 | 数学计算、精确推理 |
| 0.5-0.7 | 平衡 | 并行扩展、探索 |
| 0.8-1.0 | 创造性高 | 多样化尝试 |

### 10.3 记忆检索调优

```yaml
memory:
  top_k: 1          # 检索数量，1-3 通常最佳
  similarity_threshold: 0.5  # 过滤阈值
```

- `top_k=1`：最相关的单条记忆，避免信息过载
- `top_k=3`：多角度参考，适合复杂任务
- `similarity_threshold`：过低会引入噪声，过高会遗漏有用记忆

---

## 11. 扩展开发

### 11.1 添加新环境

继承 `BaseEnv` 类：

```python
from reasoning_bank.envs.base import BaseEnv

class MyCustomEnv(BaseEnv):
    
    @property
    def name(self) -> str:
        return "my_custom_env"
    
    def reset(self, task_id: str = None) -> str:
        """初始化任务，返回初始观察"""
        pass
    
    def step(self, action: str) -> Tuple[str, float, bool, dict]:
        """执行动作，返回 (观察, 奖励, 是否结束, 信息)"""
        pass
    
    def evaluate(self, prediction: str) -> bool:
        """评估预测是否正确"""
        pass
    
    def get_ground_truth(self) -> str:
        """获取标准答案"""
        pass
```

### 11.2 自定义 Prompt

在 `PromptRegistry` 中添加：

```python
class PromptRegistry:
    
    # 添加自定义模板
    MY_CUSTOM_PROMPT = """Your custom prompt here...
    
    {memory_block}
    
    Problem: {problem}
    """
    
    @classmethod
    def get_my_prompt(cls, problem: str, memories=None):
        memory_block = cls._format_memories(memories)
        return cls.MY_CUSTOM_PROMPT.format(
            problem=problem,
            memory_block=memory_block
        )
```

### 11.3 自定义记忆提取器

```python
from reasoning_bank.core.extractor import MemoryExtractor

class MyExtractor(MemoryExtractor):
    
    def extract(self, agent_result, ground_truth=""):
        # 自定义提取逻辑
        items = super().extract(agent_result, ground_truth)
        
        # 后处理
        for item in items:
            item["source"] = "my_extractor"
        
        return items
```

---

## 12. 常见问题

### Q1: API 调用失败

**症状**：`Error: 401 Unauthorized` 或连接超时

**解决方案**：
```bash
# 检查 API Key
echo $OPENROUTER_API_KEY

# 确保 .env 文件存在
cat .env

# 测试 API 连接
curl https://openrouter.ai/api/v1/models \
  -H "Authorization: Bearer $OPENROUTER_API_KEY"
```

### Q2: 记忆提取为 0

**可能原因**：
1. 未使用 `--use-memory` 参数
2. JSON 解析失败（已在最新版本修复）

**解决方案**：
```bash
# 确保使用记忆模式
python -m reasoning_bank.main --dataset math500 --use-memory --num-tasks 5
```

### Q3: Embedding 加载缓慢

**解决方案**：
```yaml
# config.yaml
embedding:
  model: "sentence-transformers/all-MiniLM-L6-v2"
  device: "cuda"  # 使用 GPU 加速
```

### Q4: 内存不足

**解决方案**：
- 减少 `--parallel-n` 数量
- 使用较小的 Embedding 模型
- 分批运行任务

### Q5: 答案解析不准确

**说明**：系统会自动处理：
- LaTeX 格式差异（`\frac` vs `\dfrac`）
- 空格差异
- 等价数学表达式

如果仍有问题，可以检查 `answer_parser.py` 中的规范化逻辑。

---

## 附录

### A. 项目结构

```
reasoning_bank/
├── config/
│   └── config.yaml          # 主配置文件
├── core/
│   ├── llm_service.py       # LLM API 封装
│   ├── memory.py            # 记忆库实现
│   ├── agent.py             # ReAct Agent
│   └── extractor.py         # 记忆提取器
├── envs/
│   ├── base.py              # 环境基类
│   ├── single_turn.py       # 单轮 QA 环境
│   ├── alfworld_env.py      # ALFWorld 环境
│   └── scienceworld_env.py  # ScienceWorld 环境
├── prompts/
│   └── registry.py          # Prompt 模板
├── workflows/
│   ├── evolution.py         # 进化循环
│   └── matts.py             # MaTTS 实现
├── utils/
│   ├── config.py            # 配置管理
│   ├── logger.py            # 日志工具
│   ├── embedding.py         # 向量化服务
│   └── answer_parser.py     # 答案解析
├── data/
│   └── memory_banks/        # 记忆库存储
├── logs/                    # 运行日志
├── main.py                  # CLI 入口
└── requirements.txt         # 依赖列表
```

### B. 依赖列表

```
openai>=1.0.0
sentence-transformers>=2.2.0
numpy>=1.24.0
tqdm>=4.65.0
python-dotenv>=1.0.0
pyyaml>=6.0
chromadb>=0.4.0
```

### C. 更新日志

- **v0.1.0** (2024-01)
  - 初始版本
  - 支持单轮 QA 数据集
  - 基础记忆库功能
  - MaTTS 并行/串行扩展

---

*文档版本: 1.0 | 最后更新: 2024-01*

