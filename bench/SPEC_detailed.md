# ReasoningBank 复现项目说明文档

## 1. 项目概览

本项目旨在复现论文 _ReasoningBank: Scaling Agent Self-Evolving with Reasoning Memory_ 的核心机制。项目将构建一个具备自我进化能力的智能体框架，通过 **ReasoningBank**（推理记忆库）和 **MaTTS**（记忆感知测试时扩展）机制，在单轮推理任务（Math/MCQ）和多轮交互任务（Alfworld/ScienceWorld）中实现持续学习。

### 1.1 核心目标

1. **通用适配：** 统一接口支持 Single-turn (QA) 和 Multi-turn (Env-Interaction) 任务。
2. **闭环学习：** 实现 `Retrieval -> Inference -> Execution -> Evaluation -> Extraction -> Consolidation` 闭环。
3. **MaTTS支持：** 实现 Parallel Scaling (Self-Contrast) 和 Sequential Scaling (Self-Refine) 的记忆提取策略。

## 2. 系统架构设计

### 2.1 模块划分

系统应遵循模块化设计，主要包含以下组件：

+ `Core`: 包含 Agent 基类、Memory Bank 实现、LLM 接口封装。
+ `Environments`: 针对不同 Benchmark 的适配器层。
+ `Workflows`: 定义不同的运行模式（Vanilla, ReasoningBank-Loop, MaTTS）。
+ `Prompts`: 集中管理所有 Prompt 模板。
+ `Utils`: 包含 Embedding 服务、日志、评估指标计算等。

### 2.2 目录结构建议（仅供参考）

```latex
project_root/
├── config/              # 配置文件 (yaml/json)
├── core/
│   ├── llm.py           # LLM API Wrapper (OpenAI/Anthropic/Gemini)
│   ├── memory.py        # ReasoningBank (VectorDB + CRUD)
│   ├── agent.py         # ReAct Agent & Chain-of-Thought Agent
│   └── extractor.py     # 记忆提取逻辑 (Judge + Extractor)
├── envs/
│   ├── base.py          # 抽象基类
│   ├── math_env.py      # MATH, AIME 适配器
│   ├── qa_env.py        # MMLU-Pro, GPQA 适配器
│   └── textworld_env.py # Alfworld, ScienceWorld 适配器
├── prompts/
│   └── registry.py      # Prompt 模板注册表
├── workflows/
│   ├── inference.py     # 测试/评估流程
│   └── matts.py         # MaTTS 并行/串行扩展流程
└── main.py              # 入口文件
```

## 3. 数据结构定义 (Schema)

### 3.1 记忆项 (Memory Item)

存储于 JSON 或 数据库中，用于 RAG 检索。

```json
{
  "id": "uuid",
  "original_query": "用户原始问题或任务描述",
  "query_embedding": [float], // 向量化用于检索
  "items": [
    {
      "title": "策略标题 (e.g., 追及问题通用解法)",
      "description": "一句话简介",
      "content": "核心推理建议/避坑指南 (e.g., 务必先统一单位...)"
    }
  ],
  "source_trajectory_id": "关联的原始轨迹ID",
  "score": float // 可选，用于质量排序
}
```

### 3.2 轨迹记录 (Trajectory Trace)

用于记录交互过程，供 Extraction 模块使用。

```python
@dataclass
class TrajectoryStep:
    step_num: int
    observation: str  # 环境反馈 或 题目信息
    thought: str      # 模型思考 (CoT)
    action: str       # 模型行动 (Answer 或 Env Command)
    status: str       # 'ongoing', 'success', 'failure'
```

## 4. 核心模块详细规范

### 4.1 Environment Adapter (环境适配层)

必须定义一个抽象基类 `BaseEnv` 以统一单轮和多轮任务。

+ `BaseEnv`** 接口定义：**
  - `reset() -> observation`: 初始化任务。
  - `step(action) -> (observation, reward, done, info)`: 执行动作。
  - `get_ground_truth() -> answer`: 获取标准答案（用于 Judge）。
  - `evaluate(prediction) -> bool`: 判定任务是否成功。
+ **实现策略：**
  - **对于 Single-turn (e.g., MATH):**
    * `reset()` 返回题目 `question`。
    * `step(action)` 接收最终答案或推理步骤。如果是最终答案，直接返回 `done=True`。
    * `evaluate()` 对比 prediction 和 ground_truth (支持 Exact Match, Math Equivalence, Regex)。
  - **对于 Multi-turn (e.g., Alfworld):**
    * 封装 `alfworld` 或 `scienceworld` 的原生 Gym 接口。
    * `step()` 传递文本指令，返回环境文本观察。

### 4.2 ReasoningBank Memory (记忆库)

+ **存储后端：** 推荐使用轻量级向量库 (如 `ChromaDB` 或 `FAISS` 本地化实现)。
+ **Embedding Model：** 支持插件式替换 (OpenAI `text-embedding-3-small` 或 HuggingFace `all-MiniLM-L6-v2`)。
+ **方法：**
  - `retrieve(query, top_k=1) -> List[MemoryItem]`: 基于 Query 相似度检索。
  - `add(query, trajectory, extracted_items)`: 存入新记忆 (Append-only 策略)。

### 4.3 Agent (智能体)

+ **System Prompt 注入：**
  - 在 System Prompt 中预留 `{memory_block}` 占位符。
  - 如果检索到记忆，将其格式化为：

```latex
【相关过往经验】:
Title: ...
Content: ...
```

    - 如果无记忆，该部分留空。

+ **思考模式：** 强制模型在 Action 前输出 Thought (CoT)。

### 4.4 Extractor (记忆提取器)

这是 ReasoningBank 的核心。包含两个阶段：

1. **Judge (判题)：**
   - 若 Benchmark 提供 `evaluate()` (有 Ground Truth)，优先使用规则判定 (Oracle Judge)。
   - 若无 Ground Truth (Test-Time setting)，使用 `LLM-as-a-Judge` (Self-Correction/Verify)。
2. **Distill (提炼)：**
   - 根据 Judge 结果 (Success/Failure) 调用不同的 Prompt。
   - **输入：** Query + Trajectory + (Optional: Ground Truth if training).
   - **输出：** JSON 格式的 Title, Description, Content。

### 4.5 MaTTS Module (测试时扩展)

实现两种扩展策略：

1. **Parallel Scaling (Best-of-N):**
   - 针对同一 Query，并发运行 $ N $ 个 Agent 实例 (Temperature > 0)。
   - **Selection:** 运行 Judge 选出最佳轨迹作为最终答案。
   - **Contrastive Extraction:** 将 $ N $ 条轨迹打包发给 LLM，提示其 _"对比成功与失败的轨迹，找出导致成功的关键策略"_，生成更通用的记忆。
2. **Sequential Scaling (Self-Refine):**
   - Agent 生成轨迹后，不立即提交。
   - 进入 "Check Mode"：注入检查 Prompt，让 Agent 审查之前的步骤。
   - **Refinement Extraction:** 如果 Agent 在检查中修正了错误，提取 _"初次犯错原因及修正逻辑"_ 作为记忆。

## 5. Prompt Registry (提示词注册表)

代码中应建立 `prompts.py`，需包含以下 Intent 的 Prompt (无需具体内容，只需对应论文附录)：

1. `SYSTEM_INSTRUCTION_WITH_MEMORY`: 基础 Agent Prompt，包含 Memory 注入槽位。
2. `JUDGE_PROMPT`: 用于 Test-Time 判断轨迹是否成功 (LLM Judge)。
3. `EXTRACT_SUCCESS_PROMPT`: 从成功轨迹中提取通用策略 (Step-by-step reasoning -> General Rule)。
4. `EXTRACT_FAILURE_PROMPT`: 从失败轨迹中提取反事实教训 ("Don't do X, do Y instead")。
5. `MATTS_PARALLEL_CONTRAST_PROMPT`: 对比多条轨迹，提取一致性成功模式。
6. `MATTS_SEQUENTIAL_CHECK_PROMPT`: 引导模型进行自我反思和检查。

## 6. 工作流伪代码 (Implementation Logic)

### 6.1 训练/进化循环 (Evolution Loop)

```python
def run_evolution_episode(env, agent, memory_bank):
    # 1. Reset Env & Get Query
    query = env.reset()
    
    # 2. Retrieval
    memories = memory_bank.retrieve(query, top_k=1)
    
    # 3. Trajectory Generation
    trajectory = agent.act(query, memories) 
    
    # 4. Evaluation
    result = env.evaluate(trajectory.final_answer)
    
    # 5. Extraction (Learning)
    if result.is_success:
        new_items = extractor.extract_from_success(query, trajectory)
    else:
        # 传入 Ground Truth 帮助提取更有价值的失败教训
        new_items = extractor.extract_from_failure(query, trajectory, env.get_ground_truth())
    
    # 6. Consolidation
    memory_bank.add(query, new_items)
```

### 6.2 MaTTS 并行流程

```python
def run_matts_parallel(env, agent, memory_bank, K=5):
    query = env.reset()
    memories = memory_bank.retrieve(query)
    
    trajectories = []
    # Parallel Execution
    for _ in range(K):
        traj = agent.act(query, memories, temperature=0.7)
        trajectories.append(traj)
        
    # Contrastive Extraction
    # 即使所有轨迹都失败，对比不同失败路径也能提取信息
    new_items = extractor.extract_contrastive(query, trajectories)
    memory_bank.add(query, new_items)
    
    # Return best for inference metrics
    return select_best(trajectories) 
```

## 7. 基准测试适配细节

### 7.1 Single-turn (AIME, MATH, GPQA)

+ **数据集加载：** 数据集已经保存在本地single_turn_bench目录。
+ **Environment:** 
  - 这是一个 "Static Environment"。
  - `reset()`: pop next question.
  - `step()`: 这里的 step 可以是一次性生成 (generation) 或者 step-by-step CoT。建议统一为 CoT 格式。
+ **Answer Parsing:** 必须实现鲁棒的解析器（例如提取 `\boxed{}` 中的内容或最后一行数字）。

### 7.2 Multi-turn (Alfworld, ScienceWorld)

+ **环境依赖：** 需要预先安装 `alfworld` 和 `scienceworld` 包。
+ **Loop:** 标准的 `while not done:` 循环。
+ **Error Handling:** 处理环境返回的 "Invalid Command" 错误，这本身也是很好的 Failure Memory 来源。