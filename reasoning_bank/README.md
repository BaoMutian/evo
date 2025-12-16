# ReasoningBank

基于论文 *ReasoningBank: Scaling Agent Self-Evolving with Reasoning Memory* 的复现实现。

> 📖 **详细文档**：[完整使用手册](./docs/manual.md)

---

## 项目定位

**核心目标**：构建一个会"记笔记"并"自我进化"的 AI 智能体。

**解决的问题**：传统 LLM Agent 做完任务就忘，重复犯相同的错误。ReasoningBank 让 Agent：
1. **从经验中学习**：成功时提取"怎么做对的"，失败时提取"为什么做错了"
2. **检索并复用**：遇到新问题时，检索相关经验辅助决策
3. **持续进化**：记忆库越积累越丰富，Agent 能力持续提升

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                     main.py (CLI 入口)                       │
│            支持单轮 QA 和多轮交互两种任务类型                    │
└─────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                      Workflows 工作流层                       │
│  ┌─────────────────────┐  ┌─────────────────────────────┐   │
│  │   EvolutionLoop     │  │   MaTTS (测试时扩展)         │   │
│  │   进化循环主流程      │  │   - 并行扩展 (Self-Contrast) │   │
│  │                     │  │   - 串行扩展 (Self-Refine)   │   │
│  └─────────────────────┘  └─────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                        Core 核心层                           │
│  ┌──────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐  │
│  │  Agent   │  │  Memory   │  │ Extractor │  │    LLM    │  │
│  │ ReAct式  │  │   Bank    │  │ 记忆提取器 │  │  Service  │  │
│  │ 推理执行  │  │  记忆存储  │  │ Judge+提炼 │  │  API封装  │  │
│  └──────────┘  └───────────┘  └───────────┘  └───────────┘  │
└─────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                     Envs 环境适配层                          │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐ │
│  │  SingleTurnEnv │  │  AlfWorldEnv   │  │ ScienceWorldEnv│ │
│  │  单轮QA任务     │  │  家居多轮任务   │  │  科学多轮任务   │ │
│  │  MATH/GPQA等   │  │                │  │                │ │
│  └────────────────┘  └────────────────┘  └────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### 核心设计原则

1. **解耦**：LLM 调用、环境交互、记忆管理、Prompt 管理完全分离
2. **统一接口**：单轮和多轮任务通过相同的 `BaseEnv` 接口抽象
3. **配置化**：API Key、模型名称、参数等通过配置文件/环境变量管理

---

## 快速开始

### 1. 环境安装

```bash
# 创建/激活环境
conda activate icml26

# 安装核心依赖
pip install -r reasoning_bank/requirements.txt

# [可选] 多轮环境依赖
cd bench/alfworld && pip install -e .  # ALFWorld
pip install scienceworld                # ScienceWorld
```

### 2. 配置 API Key

```bash
# 项目根目录创建 .env 文件
echo "OPENROUTER_API_KEY=sk-or-v1-xxxxx" > .env

# ALFWorld 需要设置数据路径
export ALFWORLD_DATA=/path/to/alfworld/data
```

### 3. 运行任务

```bash
# ========== 单轮 QA 任务 ==========
python -m reasoning_bank.main --dataset math500 --num-tasks 10
python -m reasoning_bank.main -d gpqa --use-memory -n 50

# ========== 多轮交互任务 ==========
python -m reasoning_bank.main --env alfworld --num-tasks 5
python -m reasoning_bank.main --env scienceworld --num-tasks 10

# ========== 带记忆库 ==========
python -m reasoning_bank.main -d math500 --use-memory -n 100
python -m reasoning_bank.main -e alfworld --use-memory -n 20

# ========== MaTTS 扩展（仅单轮） ==========
python -m reasoning_bank.main -d math500 --matts parallel -n 5
python -m reasoning_bank.main -d math500 --matts sequential -n 5
```

---

## CLI 参数说明

| 参数 | 说明 | 示例 |
|------|------|------|
| `--dataset`, `-d` | 单轮 QA 数据集 | `math500`, `gpqa`, `aime24` |
| `--env`, `-e` | 多轮交互环境 | `alfworld`, `scienceworld` |
| `--num-tasks`, `-n` | 任务数量 | `10`, `50`, `100` |
| `--use-memory` | 启用记忆库 | - |
| `--no-extract` | 不提取新记忆 | - |
| `--clear-memory` | 清空记忆库 | - |
| `--model`, `-m` | 指定模型 | `qwen/qwen3-32b` |
| `--temperature`, `-t` | 生成温度 | `0.3` |
| `--matts` | MaTTS 模式 | `parallel`, `sequential`, `combined` |
| `--max-steps` | 多轮任务最大步数 | `30` |
| `--verbose`, `-v` | 详细输出 | - |

---

## 目录结构

```
reasoning_bank/
├── main.py                  # 统一 CLI 入口（单轮+多轮）
├── config/
│   └── config.yaml          # 主配置文件
├── core/                    # 核心模块
│   ├── llm_service.py       # LLM API 封装（OpenRouter/OpenAI 兼容）
│   ├── memory.py            # ReasoningBank 记忆库（JSONL + 向量检索）
│   ├── agent.py             # ReAct Agent（推理+执行）
│   └── extractor.py         # 记忆提取器（成功/失败 → 策略）
├── envs/                    # 环境适配器
│   ├── base.py              # 抽象基类 BaseEnv
│   ├── single_turn.py       # 单轮 QA（MATH, GPQA, MMLU-Pro, AIME）
│   ├── alfworld_env.py      # ALFWorld 多轮交互
│   └── scienceworld_env.py  # ScienceWorld 多轮交互
├── prompts/
│   └── registry.py          # Prompt 模板注册表
├── workflows/
│   ├── evolution.py         # 进化循环（检索→执行→评估→提取→存储）
│   └── matts.py             # MaTTS 测试时扩展
├── utils/
│   ├── config.py            # 配置加载
│   ├── embedding.py         # Sentence-Transformers 向量化
│   ├── logger.py            # 日志
│   └── answer_parser.py     # 答案解析（数学/选择题）
├── data/
│   └── memory_banks/        # 记忆库存储（按数据集分目录）
└── docs/
    └── manual.md            # 详细使用手册
```

---

## 核心工作流程

### 进化循环 (Evolution Loop)

```
对每个任务 task:
    1. Retrieval  → 根据 task.query 检索相关记忆
    2. Execution  → Agent 结合记忆生成推理轨迹和答案
    3. Evaluation → 判断成功/失败（对比标准答案或环境反馈）
    4. Extraction → 从轨迹中提取记忆（成功→策略，失败→教训）
    5. Storage    → 将新记忆存入 MemoryBank
```

### 记忆提取逻辑

```python
if task.is_success:
    # 提取："做对了什么？通用策略是什么？"
    items = extractor.extract_from_success(query, trajectory)
else:
    # 提取："哪里想错了？如何避免？"
    items = extractor.extract_from_failure(query, trajectory, ground_truth)
```

### MaTTS 扩展

| 模式 | 机制 | 记忆提取方式 |
|------|------|-------------|
| 并行 (parallel) | 生成 N 条轨迹 | 对比成功/失败，提取一致性模式 |
| 串行 (sequential) | 强制检查+修正 | 捕获"纠错瞬间" |
| 组合 (combined) | 先并行后串行 | 融合两者 |

---

## 数据集说明

### 单轮 QA

| ID | 数据集 | 类型 | 数量 | 文件 |
|----|--------|------|------|------|
| `math500` | MATH-500 | 数学 | 500 | `MATH-500.jsonl` |
| `aime24` | AIME 2024 | 数学竞赛 | 30 | `AIME24-30.jsonl` |
| `aime25` | AIME 2025 | 数学竞赛 | 30 | `AIME25-30.jsonl` |
| `gpqa` | GPQA-Diamond | 选择题 | 198 | `GPQA-Diamond-198.jsonl` |
| `mmlu_economics` | MMLU-Pro | 选择题 | 844 | `MMLU-Pro-economics-844.jsonl` |
| `mmlu_engineering` | MMLU-Pro | 选择题 | 969 | `MMLU-Pro-engineering-969.jsonl` |
| `mmlu_philosophy` | MMLU-Pro | 选择题 | 499 | `MMLU-Pro-philosophy-499.jsonl` |

数据存放：`bench/single_turn_bench/`

### 多轮交互

| 环境 | 说明 | 依赖 |
|------|------|------|
| ALFWorld | 家居任务（拿取、清洁、加热等） | `alfworld` 包 |
| ScienceWorld | 科学实验（沸腾、电路、遗传等） | `scienceworld` 包 |

---

## 记忆库格式

存储路径：`data/memory_banks/{bank_name}.jsonl`

```json
{
  "id": "abc12345",
  "original_query": "问题文本",
  "items": [
    {
      "title": "策略标题",
      "description": "一句话简介",
      "content": "详细建议：遇到...问题时，应该..."
    }
  ],
  "is_success": true,
  "source_trajectory_id": "task_001",
  "timestamp": "2024-12-17T10:30:00"
}
```

---

## 扩展开发

### 添加新单轮数据集

1. 准备 JSONL 文件（格式见 `bench/single_turn_bench/README.md`）
2. 在 `envs/single_turn.py` 的 `SingleTurnEnvRegistry.DATASETS` 中注册：

```python
DATASETS = {
    # ...
    "my_dataset": {
        "file": "my_dataset.jsonl",
        "type": "math",  # 或 "mcq"
    },
}
```

### 添加新多轮环境

1. 创建 `envs/my_env.py`，继承 `BaseEnv`
2. 实现必要方法：`reset()`, `step()`, `evaluate()`, `__len__()`, `__iter__()`
3. 在 `main.py` 的 `MULTI_TURN_ENVS` 中添加
4. 在 `run_multi_turn()` 中添加环境创建逻辑

### 自定义 Prompt

修改 `prompts/registry.py`，添加或修改模板：

```python
class PromptRegistry:
    MY_CUSTOM_PROMPT = """..."""
    
    @classmethod
    def get_my_prompt(cls, ...):
        return cls.MY_CUSTOM_PROMPT.format(...)
```

---

## 配置文件

`config/config.yaml` 主要配置项：

```yaml
# LLM 服务
llm:
  api_base: "https://openrouter.ai/api/v1"
  default_model: "qwen/qwen-2.5-7b-instruct"
  temperature: 0.3
  max_tokens: 4096
  timeout: 120

# 记忆库
memory:
  top_k: 1                    # 检索返回数量
  storage_path: "./data/memory_banks"
  similarity_threshold: 0.5   # 相似度阈值

# Agent
agent:
  max_steps: 30               # 多轮任务最大步数

# MaTTS
matts:
  parallel_n: 5               # 并行轨迹数
  parallel_temperature: 0.7   # 并行时的温度
```

---

## 关键 API

### MemoryBank

```python
from reasoning_bank.core.memory import MemoryBank

bank = MemoryBank(bank_name="math500")
bank.add(query="...", items=[...], is_success=True)
memories = bank.retrieve(query="...", top_k=3)
bank.save()
bank.clear()
```

### EvolutionLoop

```python
from reasoning_bank.workflows.evolution import EvolutionLoop

loop = EvolutionLoop(
    env=env,
    memory_bank=memory_bank,
    extract_memories=True,
)
stats = loop.run(num_tasks=100)
# stats.success_rate, stats.memories_added
```

### MaTTSRunner

```python
from reasoning_bank.workflows.matts import MaTTSRunner, MaTTSConfig

runner = MaTTSRunner(env=env, memory_bank=bank, config=MaTTSConfig(parallel_n=5))
result, memories = runner.run_parallel(task_id)
result, memories = runner.run_sequential(task_id)
```

---

## 参考资料

- 论文：[ReasoningBank: Scaling Agent Self-Evolving with Reasoning Memory](https://arxiv.org/abs/2509.25140)
- 论文解读：`PAPER.md`
- 设计文档：`SPEC_overall.md`, `SPEC_detailed.md`
- 详细手册：`docs/manual.md`
