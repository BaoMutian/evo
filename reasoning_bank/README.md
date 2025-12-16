# ReasoningBank

基于论文 *ReasoningBank: Scaling Agent Self-Evolving with Reasoning Memory* 的复现实现。

## 项目概述

ReasoningBank 是一个具备自我进化能力的智能体框架，通过：
- **ReasoningBank（推理记忆库）**：从成功和失败中提取可泛化的推理策略
- **MaTTS（记忆感知测试时扩展）**：通过并行/串行扩展获取高质量经验

## 快速开始

### 1. 安装依赖

```bash
# 激活环境
conda activate icml26

# 安装依赖
pip install -r reasoning_bank/requirements.txt
```

### 2. 配置 API Key

在项目根目录创建 `.env` 文件：

```bash
OPENROUTER_API_KEY=your_api_key_here
```

### 3. 运行测试

```bash
# 运行单轮QA测试（MATH-500 数据集，5 道题）
python -m reasoning_bank.main --dataset math500 --num-tasks 5

# 使用记忆库运行
python -m reasoning_bank.main --dataset gpqa --use-memory --num-tasks 10

# 指定模型
python -m reasoning_bank.main --dataset aime24 --model "qwen/qwen3-32b" --num-tasks 5

# MaTTS 并行扩展
python -m reasoning_bank.main --dataset math500 --matts parallel --num-tasks 3
```

## 模块说明

```
reasoning_bank/
├── config/              # 配置文件
│   └── config.yaml      # 主配置（LLM、记忆库、Agent 参数）
├── core/                # 核心模块
│   ├── llm_service.py   # LLM API 封装
│   ├── memory.py        # ReasoningBank 记忆库
│   ├── agent.py         # ReAct Agent
│   └── extractor.py     # 记忆提取器
├── envs/                # 环境适配器
│   ├── base.py          # 抽象基类
│   ├── single_turn.py   # 单轮QA（MATH, GPQA, MMLU-Pro）
│   ├── alfworld_env.py  # ALFWorld 多轮交互
│   └── scienceworld_env.py  # ScienceWorld 多轮交互
├── prompts/             # Prompt 模板
│   └── registry.py      # Prompt 注册表
├── workflows/           # 工作流
│   ├── evolution.py     # 进化循环
│   └── matts.py         # MaTTS 扩展
└── utils/               # 工具
    ├── config.py        # 配置管理
    ├── logger.py        # 日志
    ├── embedding.py     # 向量化服务
    └── answer_parser.py # 答案解析
```

## 支持的数据集

### 单轮 QA
| 数据集 | 类型 | 数量 |
|--------|------|------|
| `math500` | 数学题 | 500 |
| `aime24` | AIME 2024 | 30 |
| `aime25` | AIME 2025 | 30 |
| `gpqa` | 研究生级选择题 | 198 |
| `mmlu_economics` | MMLU-Pro 经济学 | 844 |
| `mmlu_engineering` | MMLU-Pro 工程学 | 969 |
| `mmlu_philosophy` | MMLU-Pro 哲学 | 499 |

### 多轮交互
| 环境 | 说明 |
|------|------|
| ALFWorld | 家居任务环境 |
| ScienceWorld | 科学实验环境 |

## 核心 API

### 基础用法

```python
from reasoning_bank.core.memory import MemoryBank
from reasoning_bank.core.agent import ReActAgent
from reasoning_bank.envs.single_turn import SingleTurnEnvRegistry
from reasoning_bank.workflows.evolution import EvolutionLoop

# 创建环境
env = SingleTurnEnvRegistry.create('math500', max_samples=100)

# 创建记忆库
memory_bank = MemoryBank(bank_name='math500')

# 创建 Agent
agent = ReActAgent(memory_bank=memory_bank)

# 运行进化循环
loop = EvolutionLoop(
    env=env,
    memory_bank=memory_bank,
    extract_memories=True,
)
stats = loop.run(num_tasks=50)
```

### MaTTS 扩展

```python
from reasoning_bank.workflows.matts import MaTTSRunner, MaTTSConfig

runner = MaTTSRunner(
    env=env,
    memory_bank=memory_bank,
    config=MaTTSConfig(parallel_n=5),
)

# 并行扩展
result, memories = runner.run_parallel(task_id)

# 串行扩展（自我修正）
result, memories = runner.run_sequential(task_id)

# 组合扩展
result, memories = runner.run_combined(task_id)
```

## 配置说明

主要配置项（`config/config.yaml`）：

```yaml
llm:
  default_model: "deepseek/deepseek-chat-v3-0324"
  temperature: 0.3
  max_tokens: 4096

memory:
  top_k: 1  # 检索记忆数量
  similarity_threshold: 0.5

matts:
  parallel_n: 5  # 并行轨迹数
  parallel_temperature: 0.7
```

## 记忆库格式

记忆以 JSONL 格式存储：

```json
{
  "id": "abc123",
  "original_query": "问题文本",
  "items": [
    {
      "title": "策略标题",
      "description": "一句话简介",
      "content": "详细建议"
    }
  ],
  "is_success": true,
  "timestamp": "2024-01-01T00:00:00"
}
```

## 扩展开发

### 添加新数据集

1. 准备 JSONL 格式数据（参考 `bench/single_turn_bench/README.md`）
2. 在 `SingleTurnEnvRegistry.DATASETS` 中注册

### 添加新环境

1. 继承 `BaseEnv` 类
2. 实现 `reset()`, `step()`, `evaluate()` 等方法

## 参考

- 论文：ReasoningBank: Scaling Agent Self-Evolving with Reasoning Memory
- 项目文档：`PAPER.md`, `SPEC_overall.md`, `SPEC_detailed.md`

