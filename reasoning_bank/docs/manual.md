# ReasoningBank 详细使用手册

## 目录

1. [项目概述](#1-项目概述)
2. [安装与配置](#2-安装与配置)
3. [核心概念](#3-核心概念)
4. [架构设计](#4-架构设计)
5. [快速开始](#5-快速开始)
6. [命令行接口 (CLI)](#6-命令行接口-cli)
7. [核心模块详解](#7-核心模块详解)
8. [工作流程详解](#8-工作流程详解)
9. [配置参数详解](#9-配置参数详解)
10. [Prompt 模板系统](#10-prompt-模板系统)
11. [扩展开发指南](#11-扩展开发指南)
12. [常见问题解答 (FAQ)](#12-常见问题解答-faq)
13. [API 参考](#13-api-参考)

---

## 1. 项目概述

### 1.1 什么是 ReasoningBank？

ReasoningBank 是一个**自我进化的智能体框架**，基于论文 *"ReasoningBank: Scaling Agent Self-Evolving with Reasoning Memory"* 实现。

**核心理念**：传统的 LLM Agent 在做任务时往往"做完就忘"，下次遇到类似问题还会重蹈覆辙。ReasoningBank 让 Agent 能够：

1. **从经验中学习**：无论成功还是失败，都能提炼出可复用的策略
2. **记忆存储与检索**：将提炼的经验存入记忆库，遇到新问题时检索相关经验
3. **自我进化**：随着经验积累，Agent 的能力持续提升

### 1.2 核心特性

| 特性 | 描述 |
|------|------|
| **ReasoningBank** | 推理记忆库，存储结构化的推理策略而非原始轨迹 |
| **MaTTS** | 记忆感知测试时扩展，通过并行/串行扩展获取高质量经验 |
| **双向学习** | 同时从成功和失败中提取经验 |
| **多环境支持** | 支持单轮 QA（数学、选择题）和多轮交互（ALFWorld、ScienceWorld） |
| **模块化设计** | LLM、环境、记忆、Prompt 完全解耦，便于扩展 |

### 1.3 与传统方法的区别

| 维度 | 传统 RAG/Memory | ReasoningBank |
|------|-----------------|---------------|
| **存储内容** | 原始轨迹、文档片段 | 结构化推理策略 |
| **失败处理** | 忽略或丢弃 | 主动提取"避坑指南" |
| **泛化能力** | 基于表面相似度 | 基于推理逻辑相似度 |
| **进化性** | 静态 | 动态进化 |

---

## 2. 安装与配置

### 2.1 环境要求

- Python 3.10+
- Conda（推荐）
- 至少 4GB 可用内存（用于 Embedding 模型）

### 2.2 安装步骤

```bash
# 1. 克隆项目
git clone <repository_url>
cd reasoning-bank

# 2. 创建/激活 Conda 环境
conda create -n icml26 python=3.10
conda activate icml26

# 3. 安装依赖
pip install -r reasoning_bank/requirements.txt
```

### 2.3 依赖说明

核心依赖：

```
# LLM 服务
openai>=1.0.0           # OpenAI SDK（兼容 OpenRouter）
python-dotenv>=1.0.0    # 环境变量管理

# Embedding
sentence-transformers>=2.2.0  # 向量化模型
numpy>=1.24.0

# 数据处理
pyyaml>=6.0             # 配置文件解析
tqdm>=4.65.0            # 进度条

# 多轮环境（可选）
alfworld>=0.3.0         # ALFWorld 环境
scienceworld>=1.1.0     # ScienceWorld 环境
```

### 2.4 API Key 配置

在项目根目录创建 `.env` 文件：

```bash
# OpenRouter API（推荐）
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxx
OPENROUTER_API_BASE=https://openrouter.ai/api/v1

# 或使用其他兼容 OpenAI 的服务
# OPENAI_API_KEY=sk-xxxxxxxxxx
# OPENAI_API_BASE=https://api.openai.com/v1
```

### 2.5 验证安装

```bash
# 运行简单测试
python -m reasoning_bank.main --dataset math500 --num-tasks 1

# 预期输出：
# 2024-xx-xx | INFO | 加载数据集 MATH-500.jsonl，共 1 条数据
# Evolution: 100%|██████████| 1/1 [00:10<00:00, ...]
# ...
```

---

## 3. 核心概念

### 3.1 记忆项 (Memory Item)

记忆是 ReasoningBank 的核心数据单元：

```python
@dataclass
class MemoryItem:
    id: str                           # 唯一标识符
    original_query: str               # 触发此记忆的原始问题
    items: List[Dict[str, str]]       # 记忆内容列表
    is_success: bool                  # 是否来自成功经验
    source_trajectory_id: str         # 关联的轨迹 ID
    timestamp: str                    # 创建时间
```

每个记忆包含多个**记忆条目**：

```json
{
  "title": "二次方程配方法",
  "description": "处理形如 ax² + bx + c = 0 的方程",
  "content": "将方程改写为 (x + p)² = q 的形式，然后开方求解..."
}
```

### 3.2 轨迹 (Trajectory)

轨迹记录了 Agent 解决问题的完整过程：

```python
trajectory = [
    {
        "step": 0,
        "observation": "问题描述...",
        "thought": "我需要先分析...",
        "action": "计算 x = ...",
        "result": "得到 x = 5"
    },
    # ... 更多步骤
]
```

### 3.3 进化循环 (Evolution Loop)

ReasoningBank 的核心工作流程：

```
1. 检索 (Retrieval)     → 根据当前问题，检索相关记忆
2. 执行 (Execution)     → Agent 结合记忆解决问题
3. 评估 (Evaluation)    → 判断任务成功/失败
4. 提取 (Extraction)    → 从轨迹中提取新记忆
5. 整合 (Consolidation) → 将新记忆存入记忆库
```

### 3.4 MaTTS 扩展

**MaTTS (Memory-aware Test-Time Scaling)** 通过增加计算量获取更高质量的记忆：

| 模式 | 机制 | 优势 |
|------|------|------|
| **并行扩展** | 生成 N 条轨迹，对比成功/失败 | 过滤伪相关，提取一致性模式 |
| **串行扩展** | 强制 Agent 检查并修正答案 | 捕获"纠错瞬间"，提取自省经验 |
| **组合扩展** | 先并行后串行 | 兼具两者优势 |

---

## 4. 架构设计

### 4.1 系统架构图

```
┌─────────────────────────────────────────────────────────────┐
│                       CLI / Main Entry                       │
└─────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                      Workflows Layer                         │
│  ┌─────────────────┐  ┌─────────────────────────────────┐   │
│  │ EvolutionLoop   │  │ MaTTS (Parallel / Sequential)   │   │
│  └─────────────────┘  └─────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                        Core Layer                            │
│  ┌──────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐  │
│  │  Agent   │  │  Memory   │  │ Extractor │  │    LLM    │  │
│  │ (ReAct)  │  │   Bank    │  │  (Judge)  │  │  Service  │  │
│  └──────────┘  └───────────┘  └───────────┘  └───────────┘  │
└─────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                     Environment Layer                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │ SingleTurn   │  │  ALFWorld    │  │   ScienceWorld   │   │
│  │  (QA/Math)   │  │ (Household)  │  │    (Science)     │   │
│  └──────────────┘  └──────────────┘  └──────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                      Utility Layer                           │
│  ┌────────┐  ┌───────────┐  ┌────────┐  ┌────────────────┐  │
│  │ Config │  │ Embedding │  │ Logger │  │ Answer Parser  │  │
│  └────────┘  └───────────┘  └────────┘  └────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 目录结构

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
│   ├── base.py              # 环境抽象基类
│   ├── single_turn.py       # 单轮 QA 环境
│   ├── alfworld_env.py      # ALFWorld 适配器
│   └── scienceworld_env.py  # ScienceWorld 适配器
├── prompts/
│   └── registry.py          # Prompt 模板注册表
├── workflows/
│   ├── evolution.py         # 进化循环
│   └── matts.py             # MaTTS 扩展
├── utils/
│   ├── config.py            # 配置管理
│   ├── embedding.py         # Embedding 服务
│   ├── logger.py            # 日志工具
│   └── answer_parser.py     # 答案解析
├── data/
│   └── memory_banks/        # 记忆库存储目录
├── main.py                  # CLI 入口
└── requirements.txt         # 依赖列表
```

---

## 5. 快速开始

### 5.1 基础用法：无记忆运行

```bash
# 运行 MATH-500 数据集的前 10 道题
python -m reasoning_bank.main --dataset math500 --num-tasks 10
```

### 5.2 启用记忆库

```bash
# 使用记忆库运行（会自动提取并存储记忆）
python -m reasoning_bank.main --dataset math500 --num-tasks 50 --use-memory
```

### 5.3 使用已有记忆

```bash
# 后续运行会自动加载之前的记忆
python -m reasoning_bank.main --dataset math500 --num-tasks 50 --use-memory
```

### 5.4 MaTTS 扩展

```bash
# 并行扩展（生成 5 条轨迹对比）
python -m reasoning_bank.main --dataset math500 --num-tasks 10 --matts parallel --parallel-n 5

# 串行扩展（自我检查修正）
python -m reasoning_bank.main --dataset math500 --num-tasks 10 --matts sequential

# 组合扩展
python -m reasoning_bank.main --dataset math500 --num-tasks 10 --matts combined
```

### 5.5 Python API 用法

```python
from reasoning_bank.core.memory import MemoryBank
from reasoning_bank.core.agent import ReActAgent, AgentConfig
from reasoning_bank.core.llm_service import LLMService
from reasoning_bank.envs.single_turn import SingleTurnEnvRegistry
from reasoning_bank.workflows.evolution import EvolutionLoop

# 1. 创建环境
env = SingleTurnEnvRegistry.create('math500', max_samples=100)

# 2. 创建记忆库
memory_bank = MemoryBank(bank_name='math500')

# 3. 创建 LLM 服务
llm = LLMService(model="deepseek/deepseek-chat-v3-0324")

# 4. 创建进化循环
loop = EvolutionLoop(
    env=env,
    memory_bank=memory_bank,
    llm_service=llm,
    extract_memories=True,
)

# 5. 运行
stats = loop.run(num_tasks=50)
print(f"成功率: {stats.success_rate:.2%}")
print(f"新增记忆: {stats.memories_added}")
```

---

## 6. 命令行接口 (CLI)

### 6.1 完整参数列表

```bash
python -m reasoning_bank.main [OPTIONS]
```

| 参数 | 简写 | 类型 | 默认值 | 描述 |
|------|------|------|--------|------|
| `--dataset` | `-d` | str | `math500` | 数据集名称 |
| `--num-tasks` | `-n` | int | 全部 | 运行的任务数量 |
| `--use-memory` | - | flag | False | 是否使用记忆库 |
| `--no-extract` | - | flag | False | 不提取新记忆 |
| `--clear-memory` | - | flag | False | 清空记忆库后运行 |
| `--model` | `-m` | str | 配置文件值 | LLM 模型名称 |
| `--temperature` | `-t` | float | 0.3 | 生成温度 |
| `--matts` | - | str | `none` | MaTTS 模式 |
| `--parallel-n` | - | int | 5 | 并行轨迹数量 |
| `--config` | - | str | 默认路径 | 配置文件路径 |
| `--results-dir` | - | str | `./results` | 结果保存目录 |
| `--verbose` | `-v` | flag | False | 详细输出 |
| `--seed` | - | int | None | 随机种子 |
| `--shuffle` | - | flag | False | 打乱数据集顺序 |

### 6.2 支持的数据集

| 名称 | 类型 | 数量 | 描述 |
|------|------|------|------|
| `aime24` | 数学 | 30 | AIME 2024 竞赛题 |
| `aime25` | 数学 | 30 | AIME 2025 竞赛题 |
| `math500` | 数学 | 500 | MATH 数据集子集 |
| `gpqa` | MCQ | 198 | 研究生级选择题 |
| `mmlu_economics` | MCQ | 844 | MMLU-Pro 经济学 |
| `mmlu_engineering` | MCQ | 969 | MMLU-Pro 工程学 |
| `mmlu_philosophy` | MCQ | 499 | MMLU-Pro 哲学 |

### 6.3 使用示例

```bash
# 示例 1：基准测试
python -m reasoning_bank.main -d gpqa -n 50

# 示例 2：带记忆的完整测试
python -m reasoning_bank.main -d math500 --use-memory -n 100 -v

# 示例 3：清空记忆重新开始
python -m reasoning_bank.main -d math500 --use-memory --clear-memory -n 50

# 示例 4：使用特定模型
python -m reasoning_bank.main -d aime24 -m "qwen/qwen-2.5-72b-instruct" -n 10

# 示例 5：MaTTS 并行扩展
python -m reasoning_bank.main -d math500 --matts parallel --parallel-n 3 -n 20

# 示例 6：设置随机种子保证可复现
python -m reasoning_bank.main -d math500 -n 50 --seed 42 --shuffle
```

---

## 7. 核心模块详解

### 7.1 LLM Service (`core/llm_service.py`)

封装 LLM API 调用，支持 OpenRouter、OpenAI 等兼容服务。

```python
from reasoning_bank.core.llm_service import LLMService, LLMResult

# 创建服务
llm = LLMService(
    model="deepseek/deepseek-chat-v3-0324",
    temperature=0.3,
    max_tokens=4096,
)

# 调用
result: LLMResult = llm.call(
    prompt="What is 2 + 2?",
    system_prompt="You are a math expert.",
)

print(result.status)   # "success" 或 "error"
print(result.content)  # LLM 响应内容
print(result.tokens)   # Token 使用量
```

### 7.2 Memory Bank (`core/memory.py`)

推理记忆库的核心实现。

```python
from reasoning_bank.core.memory import MemoryBank, MemoryItem

# 创建记忆库
bank = MemoryBank(
    bank_name="math500",      # 记忆库名称
    storage_path="./data",    # 存储路径
    top_k=1,                  # 检索数量
    similarity_threshold=0.5, # 相似度阈值
)

# 添加记忆
bank.add(
    query="求解 x² + 5x + 6 = 0",
    items=[{
        "title": "因式分解法",
        "description": "适用于可分解的二次方程",
        "content": "将方程分解为 (x+a)(x+b)=0 的形式..."
    }],
    is_success=True,
    trajectory_id="task_001",
)

# 检索记忆
memories = bank.retrieve("求解 x² - 3x + 2 = 0", top_k=3)
for mem in memories:
    print(mem.format_for_prompt())

# 持久化
bank.save()

# 统计信息
print(bank.get_stats())
# {'total': 10, 'success': 7, 'failure': 3}
```

### 7.3 ReAct Agent (`core/agent.py`)

实现 ReAct 风格的推理智能体。

```python
from reasoning_bank.core.agent import ReActAgent, AgentConfig, AgentResult

# 配置
config = AgentConfig(
    max_steps=30,         # 最大步数
    use_react=True,       # 使用 ReAct 格式
    temperature=0.3,      # 生成温度
    verbose=True,         # 详细输出
)

# 创建 Agent
agent = ReActAgent(
    llm_service=llm,
    memory_bank=bank,
    config=config,
)

# 运行任务
result: AgentResult = agent.run(env, task_id="math_001")

print(result.answer)       # 最终答案
print(result.is_success)   # 是否成功
print(result.steps)        # 步数
print(result.trajectory)   # 完整轨迹
```

### 7.4 Memory Extractor (`core/extractor.py`)

从轨迹中提取可复用的记忆。

```python
from reasoning_bank.core.extractor import MemoryExtractor

extractor = MemoryExtractor(
    llm_service=llm,
    use_llm_judge=False,  # 是否使用 LLM 判断成功/失败
)

# 从成功轨迹提取
items = extractor.extract(agent_result, ground_truth="42")
# items = [{"title": "...", "description": "...", "content": "..."}]

# 对比多条轨迹提取（MaTTS 并行）
items = extractor.extract_contrastive(
    query="问题描述",
    results=[result1, result2, result3],  # 多条轨迹
)

# 从自我修正过程提取（MaTTS 串行）
items = extractor.extract_from_refinement(
    query="问题描述",
    initial_result=initial,
    corrected_result=corrected,
    correction_details="修正了计算错误",
)
```

### 7.5 Environment (`envs/`)

统一的环境接口。

```python
from reasoning_bank.envs.base import BaseEnv, TaskType, StepResult
from reasoning_bank.envs.single_turn import SingleTurnEnvRegistry

# 创建单轮环境
env = SingleTurnEnvRegistry.create(
    'math500',
    max_samples=100,
    shuffle=True,
    seed=42,
)

# 环境接口
observation = env.reset(task_id="math_001")  # 重置，获取问题
result = env.step(action="答案是 42")        # 执行动作
ground_truth = env.get_ground_truth()         # 获取标准答案
is_correct = env.evaluate("42")               # 评估答案

# 遍历所有任务
for task_id in env:
    obs = env.reset(task_id)
    # ... 处理任务
```

---

## 8. 工作流程详解

### 8.1 进化循环 (Evolution Loop)

```python
from reasoning_bank.workflows.evolution import EvolutionLoop, EvolutionStats

loop = EvolutionLoop(
    env=env,
    memory_bank=memory_bank,
    llm_service=llm,
    agent_config=AgentConfig(verbose=True),
    extract_memories=True,    # 是否提取记忆
    save_results=True,        # 是否保存结果
    results_dir="./results",  # 结果目录
)

# 运行完整循环
stats: EvolutionStats = loop.run(
    num_tasks=100,            # 任务数量
    show_progress=True,       # 显示进度条
)

# 查看统计
print(f"总任务: {stats.total_tasks}")
print(f"成功率: {stats.success_rate:.2%}")
print(f"平均步数: {stats.avg_steps:.2f}")
print(f"新增记忆: {stats.memories_added}")

# 获取详细结果
results = loop.get_results()
for r in results:
    print(f"{r.task_id}: {'✅' if r.is_success else '❌'}")
```

### 8.2 MaTTS 扩展

```python
from reasoning_bank.workflows.matts import MaTTSRunner, MaTTSConfig

config = MaTTSConfig(
    parallel_n=5,                  # 并行轨迹数
    parallel_temperature=0.7,      # 并行温度
    sequential_max_refine=3,       # 最大修正次数
)

runner = MaTTSRunner(
    env=env,
    memory_bank=memory_bank,
    llm_service=llm,
    config=config,
)

# 并行扩展
best_result, memories = runner.run_parallel(task_id="math_001")

# 串行扩展
result, memories = runner.run_sequential(task_id="math_001")

# 组合扩展
result, memories = runner.run_combined(task_id="math_001")
```

---

## 9. 配置参数详解

### 9.1 配置文件结构

`config/config.yaml`:

```yaml
# LLM 服务配置
llm:
  api_base: "https://openrouter.ai/api/v1"
  api_key: ""  # 通过环境变量设置
  default_model: "qwen/qwen-2.5-7b-instruct"
  temperature: 0.3
  max_tokens: 4096
  timeout: 120
  max_retries: 3

# Embedding 配置
embedding:
  model: "sentence-transformers/all-MiniLM-L6-v2"
  device: "cpu"  # 或 "cuda"
  batch_size: 32

# 记忆库配置
memory:
  top_k: 1
  storage_path: "./data/memory_banks"
  similarity_threshold: 0.5

# Agent 配置
agent:
  max_steps: 30
  react_format: true

# MaTTS 配置
matts:
  parallel_n: 5
  parallel_temperature: 0.7
  sequential_max_refine: 3

# 日志配置
logging:
  level: "INFO"
  log_dir: "./logs"
  log_to_file: true
  log_to_console: true

# 数据集路径
datasets:
  single_turn:
    base_path: "./bench/single_turn_bench"
  alfworld:
    data_path: "./bench/alfworld/data"
  scienceworld:
    simplifications: "easy"
```

### 9.2 环境变量

| 变量名 | 描述 | 示例 |
|--------|------|------|
| `OPENROUTER_API_KEY` | OpenRouter API 密钥 | `sk-or-v1-xxx` |
| `OPENROUTER_API_BASE` | API 基础 URL | `https://openrouter.ai/api/v1` |
| `CUDA_VISIBLE_DEVICES` | GPU 设备 | `0` |

### 9.3 动态配置

```python
from reasoning_bank.utils.config import get_config, load_config

# 加载配置
config = load_config("path/to/config.yaml")

# 获取配置值
model = get_config("llm.default_model")
top_k = get_config("memory.top_k", default=1)

# 运行时覆盖
llm = LLMService(
    model="different-model",  # 覆盖配置
    temperature=0.5,
)
```

---

## 10. Prompt 模板系统

### 10.1 模板类型

| 模板 | 用途 |
|------|------|
| `SYSTEM_BASE` | 基础系统提示词 |
| `SYSTEM_REACT` | ReAct 格式提示词 |
| `SYSTEM_SINGLE_TURN` | 单轮 QA 提示词 |
| `SYSTEM_ALFWORLD` | ALFWorld 环境提示词 |
| `SYSTEM_SCIENCEWORLD` | ScienceWorld 环境提示词 |
| `EXTRACT_SUCCESS_PROMPT` | 成功经验提取 |
| `EXTRACT_FAILURE_PROMPT` | 失败经验提取 |
| `MATTS_PARALLEL_CONTRAST_PROMPT` | MaTTS 对比提取 |
| `MATTS_SEQUENTIAL_CHECK_PROMPT` | MaTTS 检查提示词 |

### 10.2 使用方法

```python
from reasoning_bank.prompts.registry import PromptRegistry

# 获取系统提示词（带记忆注入）
prompt = PromptRegistry.get_system_prompt(
    prompt_type="single_turn",
    memories=[memory1, memory2],
    memory_formatter=lambda mems: "\n".join(m.format_for_prompt() for m in mems),
)

# 获取提取提示词
extract_prompt = PromptRegistry.get_extraction_prompt(
    is_success=True,
    query="问题描述",
    trajectory="轨迹文本",
    ground_truth="正确答案",
)

# 获取 MaTTS 对比提示词
contrast_prompt = PromptRegistry.get_matts_contrast_prompt(
    query="问题描述",
    trajectories="多条轨迹...",
)
```

### 10.3 自定义模板

```python
class CustomPromptRegistry(PromptRegistry):
    """自定义 Prompt 注册表"""
    
    MY_CUSTOM_PROMPT = """Your custom prompt here.
    
    {memory_block}
    
    Solve: {query}
    """
    
    @classmethod
    def get_custom_prompt(cls, query: str, memories: list = None) -> str:
        memory_block = ""
        if memories:
            memory_block = cls.MEMORY_BLOCK_TEMPLATE.format(
                memories="\n".join(m.format_for_prompt() for m in memories)
            )
        return cls.MY_CUSTOM_PROMPT.format(
            memory_block=memory_block,
            query=query,
        )
```

---

## 11. 扩展开发指南

### 11.1 添加新数据集

#### 步骤 1：准备数据文件

创建 JSONL 格式文件：

```jsonl
{"id": "q1", "question": "问题1", "answer": "答案1", "solution": "解答过程"}
{"id": "q2", "question": "问题2", "answer": "答案2", "options": ["A. ...", "B. ..."]}
```

#### 步骤 2：注册数据集

修改 `envs/single_turn.py`:

```python
class SingleTurnEnvRegistry:
    DATASETS = {
        # ... 现有数据集
        "my_dataset": {
            "file": "my_dataset.jsonl",
            "type": "math",  # 或 "mcq"
        },
    }
```

#### 步骤 3：使用

```bash
python -m reasoning_bank.main --dataset my_dataset --num-tasks 10
```

### 11.2 添加新环境

#### 步骤 1：创建适配器

```python
# envs/my_env.py
from reasoning_bank.envs.base import BaseEnv, TaskType, StepResult, TaskInfo

class MyEnv(BaseEnv):
    """我的自定义环境"""
    
    @property
    def task_type(self) -> TaskType:
        return TaskType.MULTI_TURN  # 或 SINGLE_TURN
    
    def reset(self, task_id: str = None) -> str:
        """重置环境，返回初始观察"""
        # 实现重置逻辑
        self.current_task = TaskInfo(
            task_id=task_id,
            task_type=self.task_type,
            query="任务描述",
        )
        return "初始观察"
    
    def step(self, action: str) -> StepResult:
        """执行动作"""
        # 实现步进逻辑
        return StepResult(
            observation="环境反馈",
            reward=0.0,
            done=False,
        )
    
    def get_ground_truth(self) -> str:
        return self.current_task.ground_truth
    
    def evaluate(self, prediction: str) -> bool:
        # 实现评估逻辑
        return prediction == self.get_ground_truth()
    
    def __len__(self) -> int:
        return len(self.tasks)
    
    def __iter__(self):
        return iter(self.task_ids)
```

#### 步骤 2：注册环境

```python
# envs/__init__.py
from .my_env import MyEnv

__all__ = [..., "MyEnv"]
```

### 11.3 自定义记忆提取策略

```python
from reasoning_bank.core.extractor import MemoryExtractor

class CustomExtractor(MemoryExtractor):
    """自定义记忆提取器"""
    
    def extract(self, agent_result, ground_truth: str = ""):
        # 自定义提取逻辑
        # 可以使用不同的 Prompt 或后处理规则
        
        # 调用父类方法
        items = super().extract(agent_result, ground_truth)
        
        # 自定义后处理
        filtered_items = [
            item for item in items
            if len(item["content"]) > 50  # 过滤太短的记忆
        ]
        
        return filtered_items
```

### 11.4 自定义 LLM 服务

```python
from reasoning_bank.core.llm_service import LLMService, LLMResult

class CustomLLMService(LLMService):
    """支持本地模型的 LLM 服务"""
    
    def __init__(self, model_path: str, **kwargs):
        super().__init__(**kwargs)
        self.local_model = self._load_local_model(model_path)
    
    def _load_local_model(self, path):
        # 加载本地模型
        pass
    
    def call(self, prompt: str, **kwargs) -> LLMResult:
        # 优先使用本地模型
        if self.local_model:
            response = self.local_model.generate(prompt)
            return LLMResult(status="success", content=response)
        
        # 回退到 API 调用
        return super().call(prompt, **kwargs)
```

---

## 12. 常见问题解答 (FAQ)

### Q1: 记忆库为什么没有提取到记忆？

**可能原因：**
1. 没有使用 `--use-memory` 参数
2. LLM 返回的 JSON 格式不正确（已修复 LaTeX 转义问题）
3. `extract_memories=False`

**解决方案：**
```bash
# 确保启用记忆提取
python -m reasoning_bank.main -d math500 --use-memory -n 10 -v
```

### Q2: 如何查看记忆库内容？

```python
from reasoning_bank.core.memory import MemoryBank

bank = MemoryBank(bank_name="math500")
print(f"记忆数量: {len(bank)}")
print(f"统计: {bank.get_stats()}")

for mem in bank.memories[:5]:
    print(f"- {mem.items[0]['title']}")
    print(f"  {mem.items[0]['content'][:100]}...")
```

### Q3: 如何重置记忆库？

```bash
# CLI 方式
python -m reasoning_bank.main -d math500 --use-memory --clear-memory

# Python 方式
bank = MemoryBank(bank_name="math500")
bank.clear()
```

### Q4: 如何使用自己的 LLM？

修改 `config.yaml` 或使用环境变量：

```yaml
llm:
  api_base: "http://localhost:8000/v1"  # 本地 vLLM
  default_model: "Qwen/Qwen2.5-7B-Instruct"
```

### Q5: 多轮环境（ALFWorld/ScienceWorld）如何使用？

```bash
# 安装环境
pip install alfworld scienceworld

# 运行
python -c "
from reasoning_bank.envs.alfworld_env import AlfworldEnv
env = AlfworldEnv()
obs = env.reset()
print(obs)
"
```

### Q6: 如何调试记忆检索？

```python
bank = MemoryBank(bank_name="math500")

# 检索并打印相似度
query = "求解方程 x² + 3x - 4 = 0"
memories = bank.retrieve(query, top_k=5)

for i, mem in enumerate(memories):
    print(f"{i+1}. 相似度: {mem.similarity:.4f}")
    print(f"   原始问题: {mem.original_query[:50]}...")
    print(f"   策略: {mem.items[0]['title']}")
```

### Q7: 如何提高成功率？

1. **使用记忆库**：积累经验后成功率会提升
2. **调整温度**：降低 `temperature` 使输出更稳定
3. **使用更强模型**：如 `gpt-4o`、`claude-3-sonnet`
4. **MaTTS 扩展**：使用并行/串行扩展

---

## 13. API 参考

### 13.1 核心类

#### LLMService

```python
class LLMService:
    def __init__(
        self,
        model: str = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
        timeout: int = 120,
    ) -> None: ...
    
    def call(
        self,
        prompt: str,
        system_prompt: str = None,
        temperature: float = None,
        max_tokens: int = None,
    ) -> LLMResult: ...
    
    async def acall(self, ...) -> LLMResult: ...
```

#### MemoryBank

```python
class MemoryBank:
    def __init__(
        self,
        storage_path: str = None,
        bank_name: str = "default",
        top_k: int = 1,
        similarity_threshold: float = 0.5,
    ) -> None: ...
    
    def add(
        self,
        query: str,
        items: List[Dict[str, str]],
        is_success: bool,
        trajectory_id: str = "",
    ) -> MemoryItem: ...
    
    def retrieve(
        self,
        query: str,
        top_k: int = None,
    ) -> List[MemoryItem]: ...
    
    def save(self) -> None: ...
    def load(self) -> None: ...
    def clear(self) -> None: ...
    def get_stats(self) -> Dict[str, int]: ...
```

#### ReActAgent

```python
class ReActAgent:
    def __init__(
        self,
        llm_service: LLMService = None,
        memory_bank: MemoryBank = None,
        config: AgentConfig = None,
    ) -> None: ...
    
    def run(
        self,
        env: BaseEnv,
        task_id: str = None,
    ) -> AgentResult: ...
```

#### EvolutionLoop

```python
class EvolutionLoop:
    def __init__(
        self,
        env: BaseEnv,
        memory_bank: MemoryBank = None,
        llm_service: LLMService = None,
        agent_config: AgentConfig = None,
        extract_memories: bool = True,
        save_results: bool = True,
        results_dir: str = "./results",
    ) -> None: ...
    
    def run(
        self,
        num_tasks: int = None,
        task_ids: List[str] = None,
        show_progress: bool = True,
    ) -> EvolutionStats: ...
    
    def run_episode(self, task_id: str = None) -> EpisodeResult: ...
    def get_results(self) -> List[EpisodeResult]: ...
    def reset_stats(self) -> None: ...
```

### 13.2 数据类

```python
@dataclass
class LLMResult:
    status: str          # "success" | "error"
    content: str         # 响应内容
    error: str = None    # 错误信息
    tokens: int = 0      # Token 使用量

@dataclass
class AgentResult:
    task_id: str
    query: str
    answer: str
    is_success: bool
    trajectory: List[Dict[str, Any]]
    steps: int
    memories_used: List[MemoryItem]
    raw_response: str

@dataclass
class EvolutionStats:
    total_tasks: int
    success_count: int
    failure_count: int
    total_steps: int
    memories_added: int
    
    @property
    def success_rate(self) -> float: ...
    @property
    def avg_steps(self) -> float: ...

@dataclass
class StepResult:
    observation: str
    reward: float
    done: bool
    info: Dict[str, Any]
```

---

## 附录

### A. 参考资源

- 论文：[ReasoningBank: Scaling Agent Self-Evolving with Reasoning Memory](https://arxiv.org/abs/2509.25140)
- 项目规格说明：`SPEC_overall.md`, `SPEC_detailed.md`
- 论文解读：`PAPER.md`

### B. 更新日志

- **v1.0.0** (2024-12)
  - 初始版本
  - 支持单轮 QA 环境（MATH、GPQA、MMLU-Pro）
  - 支持多轮环境（ALFWorld、ScienceWorld）
  - 实现 ReasoningBank 记忆库
  - 实现 MaTTS 扩展

### C. 贡献指南

1. Fork 项目
2. 创建特性分支：`git checkout -b feature/my-feature`
3. 提交更改：`git commit -m "Add my feature"`
4. 推送分支：`git push origin feature/my-feature`
5. 提交 Pull Request

---

*本手册最后更新：2024年12月*

