# ScienceWorld LLM Agent 测试指南

## 目录

- [1. ScienceWorld 简介](#1-scienceworld-简介)
- [2. 环境架构](#2-环境架构)
- [3. 任务类型详解](#3-任务类型详解)
- [4. 环境简化 (Simplifications)](#4-环境简化-simplifications)
- [5. 交互命令](#5-交互命令)
- [6. 环境安装](#6-环境安装)
- [7. 测试脚本使用](#7-测试脚本使用)
- [8. 评估指标](#8-评估指标)
- [9. Prompt 设计](#9-prompt-设计)
- [10. 示例交互](#10-示例交互)
- [11. 常见问题](#11-常见问题)

---

## 1. ScienceWorld 简介

### 1.1 什么是 ScienceWorld？

**ScienceWorld** 是一个基于文本的虚拟环境，专为测试 AI Agent 在小学科学课程任务中的能力而设计。它由 Allen Institute for AI 开发，涵盖物理、化学、生物等多个学科领域的实验任务。

ScienceWorld 的核心特点是将科学实验转换为文本交互形式，要求 Agent 通过一系列动作完成诸如"融化物质"、"测量熔点"、"种植植物"等科学任务。

### 1.2 为什么用 ScienceWorld 测试 LLM？

| 优势           | 说明                                                      |
| -------------- | --------------------------------------------------------- |
| **科学推理**   | 测试 LLM 对基础科学概念的理解（如物态变化、电路、遗传学） |
| **多步骤规划** | 任务通常需要 10-50 步才能完成，考验长期规划能力           |
| **因果理解**   | 需要理解动作与结果之间的因果关系                          |
| **常识应用**   | 需要应用常识知识（如"加热水会沸腾"）                      |
| **状态追踪**   | 需要追踪物体状态变化（温度、位置、相态等）                |
| **任务多样性** | 30 种任务类型，7000+ 变体，全面评估能力                   |

### 1.3 论文引用

```bibtex
@inproceedings{wang-etal-2022-scienceworld,
    title = "{S}cience{W}orld: Is your Agent Smarter than a 5th Grader?",
    author = "Wang, Ruoyao  and
      Jansen, Peter  and
      C{\^o}t{\'e}, Marc-Alexandre  and
      Ammanabrolu, Prithviraj",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2022",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.emnlp-main.775/",
    doi = "10.18653/v1/2022.emnlp-main.775",
    pages = "11279--11298",
}
```

---

## 2. 环境架构

### 2.1 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                         ScienceWorld                            │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    py4j    ┌─────────────────────────────────┐ │
│  │   Python    │ <========> │    Java (scienceworld.jar)      │ │
│  │   API       │            │    Scala 模拟器核心              │ │
│  └─────────────┘            └─────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
         │                                    │
         │ step(action)                       │ observation, reward, done, info
         ▼                                    ▼
┌─────────────────┐                ┌─────────────────────────────┐
│    LLM Agent    │                │   虚拟科学实验室环境         │
│  (OpenRouter)   │                │   - 多个房间（厨房、外面等） │
│                 │                │   - 物品（温度计、炉子等）   │
│                 │                │   - 动植物                   │
└─────────────────┘                └─────────────────────────────┘
```

### 2.2 环境特点

- **基于文本**: 所有交互都是文本形式
- **分数系统**: 0-100 分，完成子目标获得部分分数
- **步数限制**: 默认 100 步，超过则任务失败
- **多房间**: 包括厨房、浴室、卧室、户外、工坊等
- **丰富物品**: 温度计、炉子、冰箱、花盆、种子、动物等

### 2.3 数据流

```
┌─────────────┐     观察 (obs)        ┌─────────────┐
│             │ ───────────────────>  │             │
│ ScienceWorld│                       │  LLM Agent  │
│ Environment │ <───────────────────  │             │
│             │     动作 (action)     │             │
└─────────────┘                       └─────────────┘
       │                                     │
       │  info['valid'] (有效动作列表)        │
       │  info['score'] (当前分数)           │
       │  info['taskDesc'] (任务描述)        │
       └─────────────────────────────────────┘
```

---

## 3. 任务类型详解

ScienceWorld 包含 **30 种任务**，分为 **10 个主题**：

### 3.1 物质 (Matter) - 物态变化

| 任务 ID | 任务名                        | 描述                 | 变体数 |
| ------- | ----------------------------- | -------------------- | ------ |
| 1-1     | boil                          | 将物质沸腾           | 30     |
| 1-2     | melt                          | 将物质融化           | 30     |
| 1-3     | freeze                        | 将物质冷冻           | 30     |
| 1-4     | change-the-state-of-matter-of | 改变物质状态（任意） | 30     |

**示例任务**:

```
Your task is to boil water. First, focus on the substance. Then, move it to the appropriate location to boil.
```

**典型解决步骤**:

1. `look around` - 查看环境
2. `open cupboard` - 打开橱柜
3. `pick up metal pot` - 拿起金属锅
4. `move metal pot to sink` - 把锅移到水槽
5. `activate sink` - 打开水龙头（装水）
6. `pick up metal pot` - 拿起装了水的锅
7. `move metal pot to stove` - 把锅放到炉子上
8. `activate stove` - 打开炉子
9. `wait` (多次) - 等待水沸腾

---

### 3.2 测量 (Measurement) - 温度测量

| 任务 ID | 任务名                                  | 描述               | 变体数 |
| ------- | --------------------------------------- | ------------------ | ------ |
| 2-1     | use-thermometer                         | 使用温度计测量温度 | 540    |
| 2-2     | measure-melting-point-known-substance   | 测量已知物质的熔点 | 436    |
| 2-3     | measure-melting-point-unknown-substance | 测量未知物质的熔点 | 300    |

**示例任务**:

```
Your task is to measure the melting point of chocolate. Focus on the thermometer and the substance.
```

---

### 3.3 电学 (Electricity) - 电路与导电性

| 任务 ID | 任务名                                           | 描述                        | 变体数 |
| ------- | ------------------------------------------------ | --------------------------- | ------ |
| 3-1     | power-component                                  | 创建电路，为组件供电        | 20     |
| 3-2     | power-component-renewable-vs-nonrenewable-energy | 使用可再生/不可再生能源供电 | 20     |
| 3-3     | test-conductivity                                | 测试已知物质的导电性        | 900    |
| 3-4     | test-conductivity-of-unknown-substances          | 测试未知物质的导电性        | 600    |

**示例任务**:

```
Your task is to determine if a paper clip is electrically conductive by connecting it in a circuit.
```

---

### 3.4 分类 (Classification) - 生物/非生物分类

| 任务 ID | 任务名                | 描述           | 变体数 |
| ------- | --------------------- | -------------- | ------ |
| 4-1     | find-living-thing     | 找到一个生物   | 300    |
| 4-2     | find-non-living-thing | 找到一个非生物 | 300    |
| 4-3     | find-plant            | 找到一株植物   | 300    |
| 4-4     | find-animal           | 找到一只动物   | 300    |

**示例任务**:

```
Your task is to find a living thing and place it in the red box.
```

---

### 3.5 生物 - 植物生长

| 任务 ID | 任务名     | 描述           | 变体数 |
| ------- | ---------- | -------------- | ------ |
| 5-1     | grow-plant | 种植一株植物   | 126    |
| 5-2     | grow-fruit | 种植并获得果实 | 126    |

**示例任务**:

```
Your task is to grow an apple tree. Find the seed, plant it, and provide the necessary conditions.
```

**典型解决步骤**:

1. 找到种子
2. 找到花盆
3. 将种子放入花盆
4. 给花盆浇水
5. 将花盆放到阳光下
6. 等待植物生长

---

### 3.6 化学 (Chemistry) - 混合实验

| 任务 ID | 任务名                              | 描述               | 变体数 |
| ------- | ----------------------------------- | ------------------ | ------ |
| 6-1     | chemistry-mix                       | 通用混合任务       | 32     |
| 6-2     | chemistry-mix-paint-secondary-color | 混合颜料（二次色） | 36     |
| 6-3     | chemistry-mix-paint-tertiary-color  | 混合颜料（三次色） | 36     |

**示例任务**:

```
Your task is to mix red paint and yellow paint to create orange paint.
```

---

### 3.7 生物 - 寿命比较

| 任务 ID | 任务名                                     | 描述                         | 变体数 |
| ------- | ------------------------------------------ | ---------------------------- | ------ |
| 7-1     | lifespan-longest-lived                     | 找出寿命最长的动物           | 125    |
| 7-2     | lifespan-shortest-lived                    | 找出寿命最短的动物           | 125    |
| 7-3     | lifespan-longest-lived-then-shortest-lived | 依次找出最长和最短寿命的动物 | 125    |

---

### 3.8 生物 - 生命周期

| 任务 ID | 任务名                 | 描述             | 变体数 |
| ------- | ---------------------- | ---------------- | ------ |
| 8-1     | identify-life-stages-1 | 识别植物生命阶段 | 14     |
| 8-2     | identify-life-stages-2 | 识别动物生命阶段 | 10     |

---

### 3.9 力学 (Forces) - 斜面实验

| 任务 ID | 任务名                                   | 描述                 | 变体数 |
| ------- | ---------------------------------------- | -------------------- | ------ |
| 9-1     | inclined-plane-determine-angle           | 确定斜面角度         | 168    |
| 9-2     | inclined-plane-friction-named-surfaces   | 测试已知表面的摩擦力 | 1386   |
| 9-3     | inclined-plane-friction-unnamed-surfaces | 测试未知表面的摩擦力 | 162    |

---

### 3.10 生物 - 孟德尔遗传学

| 任务 ID | 任务名                           | 描述                 | 变体数 |
| ------- | -------------------------------- | -------------------- | ------ |
| 10-1    | mendelian-genetics-known-plant   | 已知植物的遗传学实验 | 120    |
| 10-2    | mendelian-genetics-unknown-plant | 未知植物的遗传学实验 | 480    |

---

## 4. 环境简化 (Simplifications)

ScienceWorld 提供了一组**环境简化选项**，用于降低任务难度，让 Agent 更容易完成任务。这对于测试 LLM 的科学推理能力（而非导航/操作能力）非常有用。

### 4.1 为什么需要简化？

ScienceWorld 的原始环境非常复杂：

1. **导航困难**：需要多步移动才能到达目标位置（如从厨房到户外）
2. **动作空间巨大**：电路任务有大量 `connect X to Y` 动作组合
3. **时间敏感**：植物需要定期浇水，否则会死亡
4. **容器状态**：很多物品在关闭的容器里，需要先打开才能访问

启用简化后，Agent 可以更专注于任务的**核心科学推理**，而不是被这些"繁琐"操作困扰。

### 4.2 简化选项详解

| 简化选项                 | 说明                                                                     | 适用场景                |
| ------------------------ | ------------------------------------------------------------------------ | ----------------------- |
| `teleportAction`         | 允许 Agent 直接传送到任意位置（`teleport to kitchen`），无需逐步导航     | 所有任务                |
| `openDoors`              | 所有门默认打开，Agent 无需执行 `open door` 动作                          | 所有任务                |
| `selfWateringFlowerPots` | 花盆自动浇水，植物不会因缺水死亡                                         | 植物生长任务 (5-1, 5-2) |
| `noElectricalAction`     | 移除所有电路相关动作（`connect X to Y`），大幅减小动作空间               | **非**电路任务          |
| `openContainers`         | 所有容器（冰箱、橱柜、抽屉等）默认打开，无需先 `open` 才能访问里面的物品 | 需要从容器取物的任务    |

### 4.3 预设模式

**`easy`** 预设包含以下简化：

- ✅ `teleportAction` - 传送
- ✅ `openDoors` - 门打开
- ✅ `selfWateringFlowerPots` - 自动浇水
- ✅ `noElectricalAction` - 无电路动作
- ❌ `openContainers` - **不包含**（需手动添加）

```bash
# 使用 easy 预设
python scienceworld_test.py --simplifications easy

# easy 预设 + openContainers
python scienceworld_test.py --simplifications "easy,openContainers"
```

### 4.4 自定义简化

可以用逗号分隔多个简化选项：

```bash
# 只启用传送和开门
python scienceworld_test.py --simplifications "teleportAction,openDoors"

# 完全简化（所有选项）
python scienceworld_test.py --simplifications "teleportAction,openDoors,selfWateringFlowerPots,noElectricalAction,openContainers"

# 不使用任何简化（最难模式）
python scienceworld_test.py --simplifications ""
```

### 4.5 重要限制

> ⚠️ **电路任务不能使用 `noElectricalAction`**
>
> 以下任务需要电路动作，使用 `noElectricalAction` 会报错：
>
> - 3-1: power-component
> - 3-2: power-component-renewable-vs-nonrenewable-energy
> - 3-3: test-conductivity
> - 3-4: test-conductivity-of-unknown-substances
>
> 测试这些任务时，请使用自定义简化或不使用 `easy` 预设。

```bash
# 测试电路任务的推荐简化设置
python scienceworld_test.py --task_ids 3-1 3-2 --simplifications "teleportAction,openDoors,openContainers"
```

---

## 5. 交互命令

### 5.1 导航命令

| 命令        | 格式                     | 示例                  | 说明                           |
| ----------- | ------------------------ | --------------------- | ------------------------------ |
| look around | `look around`            | `look around`         | 查看当前位置的物品和可去的地方 |
| go to       | `go to [location]`       | `go to kitchen`       | 移动到指定位置                 |
| teleport to | `teleport to [location]` | `teleport to outside` | 传送到位置（需启用简化）       |

### 5.2 物品操作

| 命令     | 格式                          | 示例                  | 说明                   |
| -------- | ----------------------------- | --------------------- | ---------------------- |
| pick up  | `pick up [object]`            | `pick up thermometer` | 拿起物品               |
| put down | `put down [object]`           | `put down apple`      | 放下物品               |
| move     | `move [object] to [location]` | `move pot to stove`   | 移动物品到指定位置     |
| examine  | `examine [object]`            | `examine thermometer` | 检查物品详情           |
| read     | `read [object]`               | `read thermometer`    | 读取（温度计、书籍等） |

### 5.3 容器操作

| 命令  | 格式                                | 示例                  | 说明        |
| ----- | ----------------------------------- | --------------------- | ----------- |
| open  | `open [container]`                  | `open fridge`         | 打开容器/门 |
| close | `close [container]`                 | `close cupboard`      | 关闭容器/门 |
| pour  | `pour [substance] into [container]` | `pour water into cup` | 倒入液体    |

### 5.4 设备操作

| 命令       | 格式                       | 示例                       | 说明         |
| ---------- | -------------------------- | -------------------------- | ------------ |
| activate   | `activate [device]`        | `activate stove`           | 启动设备     |
| deactivate | `deactivate [device]`      | `deactivate sink`          | 关闭设备     |
| use        | `use [object] on [target]` | `use thermometer on water` | 使用物品     |
| connect    | `connect [obj1] to [obj2]` | `connect wire to battery`  | 连接（电路） |

### 5.5 其他命令

| 命令      | 格式                | 示例             | 说明                         |
| --------- | ------------------- | ---------------- | ---------------------------- |
| wait      | `wait`              | `wait`           | 等待一个时间步               |
| wait1     | `wait1`             | `wait1`          | 等待（同 wait）              |
| inventory | `inventory`         | `inventory`      | 查看携带的物品               |
| task      | `task`              | `task`           | 查看当前任务描述             |
| focus on  | `focus on [object]` | `focus on water` | 聚焦特定物体（某些任务需要） |

### 5.6 重要规则

> ⚠️ **Agent 可以携带多个物品**（与 ALFWorld 不同）
>
> ⚠️ **某些容器需要先 `open` 才能看到/取出里面的物品**
>
> ⚠️ **使用 `wait` 命令让时间流逝（植物生长、水沸腾等）**
>
> ⚠️ **物态变化需要时间，多次 `wait` 直到完成**

---

## 6. 环境安装

### 6.1 系统要求

- **Java 1.8+**: ScienceWorld 核心是 Java/Scala 编写

```
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install openjdk-11-jdk

# 验证安装
java -version
```

- **Python 3.8+**: Python API 接口

### 6.2 安装步骤

```bash
# 1. 创建 conda 环境
conda create --name scienceworld python=3.8
conda activate scienceworld

# 2. 从 PyPI 安装
pip install scienceworld

# 或者从源码安装
git clone https://github.com/allenai/ScienceWorld.git
cd ScienceWorld
pip install .
```

### 6.3 验证安装

```bash
# 运行随机 agent 示例
python examples/random_agent.py --task-num=13 --num-episodes=1

# 运行人类交互模式
python examples/human.py --task-num=3 --num-episodes=1
```

### 6.4 环境变量

```bash
# 可选：设置 OpenRouter API 密钥用于 LLM 测试
export OPENROUTER_API_KEY="your-api-key"
```

---

## 7. 测试脚本使用

### 7.1 脚本位置

```
/home/bmt/evo/bench/scienceworld_test.py
```

### 7.2 命令行参数

| 参数                | 类型  | 默认值                             | 说明                          |
| ------------------- | ----- | ---------------------------------- | ----------------------------- |
| `--model`           | str   | `qwen/qwen3-30b-a3b-instruct-2507` | OpenRouter 上的模型标识       |
| `--num_episodes`    | int   | `5`                                | 每个任务测试的 episode 数量   |
| `--task_ids`        | str[] | `all`                              | 任务 ID 列表 (如 "1-1" "2-1") |
| `--simplifications` | str   | `easy`                             | 简化设置预设                  |
| `--max_steps`       | int   | `50`                               | 每个 episode 的最大步数       |
| `--no_few_shot`     | flag  | False                              | 禁用 few-shot 示例            |
| `--quiet`           | flag  | False                              | 减少输出（只显示结果）        |
| `--output`          | str   | 自动生成                           | 结果保存的 JSON 文件路径      |
| `--demo`            | flag  | -                                  | 运行单个任务演示              |
| `--seed`            | int   | `42`                               | 随机种子                      |
| `--split`           | str   | `dev`                              | 数据集划分 (train/dev/test)   |

### 7.3 使用示例

#### 运行单个任务演示

```bash
conda activate scienceworld
python scienceworld_test.py --demo --task_ids 1-2 --model "qwen/qwen3-8b"
```

#### 运行完整测试

```bash
# 测试所有任务，每个任务 3 个 episode
python scienceworld_test.py --model "qwen/qwen3-8b" --num_episodes 3

# 只测试物态变化任务
python scienceworld_test.py --model "qwen/qwen3-8b" --task_ids 1-1 1-2 1-3 1-4

# 测试分类任务
python scienceworld_test.py --model "qwen/qwen3-8b" --task_ids 4-1 4-2 4-3 4-4 --num_episodes 5

# 安静模式，只显示最终结果
python scienceworld_test.py --model "qwen/qwen3-8b" --quiet
```

#### 测试不同模型

```bash
# 测试 Claude
python scienceworld_test.py --model "anthropic/claude-3.5-sonnet"

# 测试 GPT-4
python scienceworld_test.py --model "openai/gpt-4-turbo"

# 测试 DeepSeek
python scienceworld_test.py --model "deepseek/deepseek-chat-v3-0324"
```

### 7.4 输出文件格式

测试完成后会生成 JSON 格式的结果文件：

```json
{
  "model": "qwen/qwen3-8b",
  "timestamp": "2025-12-16T10:30:00",
  "config": {
    "num_episodes": 5,
    "task_ids": ["1-1", "1-2", "4-1"],
    "simplifications": "easy",
    "max_steps": 50,
    "temperature": 0.3,
    "seed": 42,
    "split": "dev"
  },
  "summary": {
    "total_episodes": 15,
    "successes": 8,
    "success_rate": 0.533,
    "avg_score": 65.2,
    "avg_steps": 28.4
  },
  "by_task": {
    "1-1": {
      "task_name": "boil",
      "topic": "Matter",
      "episodes": 5,
      "successes": 3,
      "success_rate": 0.6,
      "avg_score": 72.0
    }
  },
  "results": [...]
}
```

---

## 8. 评估指标

### 8.1 主要指标

| 指标                      | 说明               | 计算方式                      |
| ------------------------- | ------------------ | ----------------------------- |
| **成功率 (Success Rate)** | 完成任务的比例     | `score=100 的数量 / 总任务数` |
| **平均分数 (Avg Score)**  | 平均获得的分数     | `总分数 / 任务数` (0-100)     |
| **平均步数 (Avg Steps)**  | 完成任务的平均步数 | `总步数 / 任务数`             |

### 8.2 分任务统计

```bash
# 分任务类型统计
python scienceworld_test.py --task_ids 1-1 1-2 1-3 1-4 --num_episodes 10
# 会输出每个任务的详细统计
```

### 8.3 业界基准参考

| 方法                  | 平均分数 | 成功率 |
| --------------------- | -------- | ------ |
| Random Agent          | ~5       | ~0%    |
| DRRN                  | ~18      | ~5%    |
| KG-A2C                | ~20      | ~7%    |
| CALM                  | ~25      | ~10%   |
| Behavior Cloning (T5) | ~30      | ~12%   |
| Human Expert          | ~90+     | ~85%+  |

---

## 9. Prompt 设计

### 9.1 ReAct 风格

测试脚本采用 **ReAct（Reasoning + Acting）** 风格的 prompt，要求模型输出：

```
THINK: [对当前观察的推理]
ACTION: [执行的动作]
```

### 9.2 Prompt 结构

```
┌─────────────────────────────────────┐
│  SYSTEM_PROMPT                      │  <- 介绍环境、任务类型和可用动作
├─────────────────────────────────────┤
│  FEW_SHOT_EXAMPLES (可选)           │  <- 示例交互
├─────────────────────────────────────┤
│  TASK_PROMPT                        │  <- 当前任务描述和初始观察
├─────────────────────────────────────┤
│  CONVERSATION_HISTORY               │  <- 之前的交互历史
└─────────────────────────────────────┘
```

### 9.3 Few-shot 示例

脚本包含针对不同任务类型的示例：

1. **物态变化任务** - 演示如何融化/沸腾物质
2. **分类任务** - 演示如何找到并放置生物/非生物
3. **测量任务** - 演示如何使用温度计

---

## 10. 示例交互

### 10.1 成功案例：融化任务 (melt)

```
=== NEW TASK ===
Your task is to melt ice. First, focus on the substance. Then, take actions to melt it.

You are in the kitchen. In one part of the room is a large kitchen table.
There is a cupboard, fridge, counter, and sink.

--- Step 1 ---
LLM Think: I need to find ice first. Let me check the fridge since ice is usually stored there.
LLM Action: open fridge
Observation: The fridge is now open. In the fridge, you see: ice cube, butter, milk.

--- Step 2 ---
LLM Think: Found ice cube in the fridge. Let me take it.
LLM Action: pick up ice cube
Observation: You pick up the ice cube.

--- Step 3 ---
LLM Think: To melt ice, I need heat. I should put it on the stove or somewhere warm.
LLM Action: focus on ice cube
Observation: You focus on the ice cube.

--- Step 4 ---
LLM Think: Now let me move the ice to the stove to heat it.
LLM Action: move ice cube to stove
Observation: You move the ice cube to the stove.

--- Step 5 ---
LLM Think: Let me activate the stove to heat the ice.
LLM Action: activate stove
Observation: The stove is now activated.

--- Step 6 ---
LLM Think: I need to wait for the ice to melt.
LLM Action: wait
Observation: You wait. The ice cube is melting...

--- Step 7 ---
LLM Action: wait
Observation: The ice cube has melted into water!

🎉 任务完成! 分数: 100, 步数: 7
```

### 10.2 成功案例：分类任务 (find-living-thing)

```
=== NEW TASK ===
Your task is to find a living thing and put it in the red box.

--- Step 1 ---
LLM Think: I need to find a living thing. Let me look around first.
LLM Action: look around
Observation: You are outside. You see a tree, a bee, flowers, a red box, and a path to the house.

--- Step 2 ---
LLM Think: I can see a bee which is a living thing. Let me pick it up.
LLM Action: pick up bee
Observation: You carefully pick up the bee.

--- Step 3 ---
LLM Think: Now I need to put the bee in the red box.
LLM Action: move bee to red box
Observation: You move the bee to the red box.

🎉 任务完成! 分数: 100, 步数: 3
```

---

## 11. 常见问题

### Q1: Java 相关错误怎么办？

**问题**: `Could not find or load main class`

**解决**:

```bash
# 确保 Java 已安装
java -version  # 应该显示 1.8 或更高

# Ubuntu 安装 Java
sudo apt-get install openjdk-11-jdk
```

### Q2: 如何提高成功率？

1. **使用 few-shot 示例**（默认启用）
2. **启用简化模式**（`--simplifications easy`）
3. **使用更强的模型**（如 GPT-4、Claude-3.5）
4. **增加 max_steps**（某些任务需要更多步骤）

### Q3: 某些任务总是失败？

不同任务难度差异很大：

- **简单**: 分类任务 (4-x)、部分物态变化 (1-x)
- **中等**: 测量任务 (2-x)、化学混合 (6-x)
- **困难**: 电路任务 (3-x)、遗传学 (10-x)、植物生长 (5-x)

建议先用简单任务测试，再逐步尝试困难任务。

### Q4: 如何查看可用动作？

在测试脚本中，`info['valid']` 包含当前状态下所有有效的动作。你也可以运行人类交互模式查看：

```bash
python examples/human.py --task-num=3
# 输入 'valid' 查看有效动作
```

### Q5: 简化模式有哪些选项？

| 简化选项                 | 说明                                |
| ------------------------ | ----------------------------------- |
| `teleportAction`         | 允许传送到任意位置                  |
| `openDoors`              | 所有门默认打开                      |
| `selfWateringFlowerPots` | 花盆自动浇水                        |
| `noElectricalAction`     | 移除电路相关动作                    |
| `openContainers`         | 所有容器默认打开                    |
| `easy`                   | 预设：前四项（不含 openContainers） |

---

## 附录 A：任务 ID 对照表

| ID   | 任务名                                           | 主题 | 描述               | 变体数 |
| ---- | ------------------------------------------------ | ---- | ------------------ | ------ |
| 1-1  | boil                                             | 物质 | 沸腾               | 30     |
| 1-2  | melt                                             | 物质 | 融化               | 30     |
| 1-3  | freeze                                           | 物质 | 冷冻               | 30     |
| 1-4  | change-the-state-of-matter-of                    | 物质 | 任意物态变化       | 30     |
| 2-1  | use-thermometer                                  | 测量 | 使用温度计         | 540    |
| 2-2  | measure-melting-point-known-substance            | 测量 | 测量已知物质熔点   | 436    |
| 2-3  | measure-melting-point-unknown-substance          | 测量 | 测量未知物质熔点   | 300    |
| 3-1  | power-component                                  | 电学 | 创建电路           | 20     |
| 3-2  | power-component-renewable-vs-nonrenewable-energy | 电学 | 可再生能源         | 20     |
| 3-3  | test-conductivity                                | 电学 | 测试导电性（已知） | 900    |
| 3-4  | test-conductivity-of-unknown-substances          | 电学 | 测试导电性（未知） | 600    |
| 4-1  | find-living-thing                                | 分类 | 找生物             | 300    |
| 4-2  | find-non-living-thing                            | 分类 | 找非生物           | 300    |
| 4-3  | find-plant                                       | 分类 | 找植物             | 300    |
| 4-4  | find-animal                                      | 分类 | 找动物             | 300    |
| 5-1  | grow-plant                                       | 生物 | 种植物             | 126    |
| 5-2  | grow-fruit                                       | 生物 | 种果实             | 126    |
| 6-1  | chemistry-mix                                    | 化学 | 通用混合           | 32     |
| 6-2  | chemistry-mix-paint-secondary-color              | 化学 | 二次色             | 36     |
| 6-3  | chemistry-mix-paint-tertiary-color               | 化学 | 三次色             | 36     |
| 7-1  | lifespan-longest-lived                           | 生物 | 最长寿命           | 125    |
| 7-2  | lifespan-shortest-lived                          | 生物 | 最短寿命           | 125    |
| 7-3  | lifespan-longest-lived-then-shortest-lived       | 生物 | 寿命排序           | 125    |
| 8-1  | identify-life-stages-1                           | 生物 | 植物生命周期       | 14     |
| 8-2  | identify-life-stages-2                           | 生物 | 动物生命周期       | 10     |
| 9-1  | inclined-plane-determine-angle                   | 力学 | 斜面角度           | 168    |
| 9-2  | inclined-plane-friction-named-surfaces           | 力学 | 已知表面摩擦力     | 1386   |
| 9-3  | inclined-plane-friction-unnamed-surfaces         | 力学 | 未知表面摩擦力     | 162    |
| 10-1 | mendelian-genetics-known-plant                   | 生物 | 已知遗传学         | 120    |
| 10-2 | mendelian-genetics-unknown-plant                 | 生物 | 未知遗传学         | 480    |

---

## 附录 B：推荐测试任务组合

### 快速测试（约 10 分钟）

```bash
python scienceworld_test.py --task_ids 4-1 4-2 1-2 --num_episodes 2
```

### 标准测试（约 30 分钟）

```bash
python scienceworld_test.py --task_ids 1-1 1-2 4-1 4-2 4-3 6-2 --num_episodes 3
```

### 完整测试（约 2 小时）

```bash
python scienceworld_test.py --num_episodes 3  # 测试所有 30 个任务
```

---

## 更新日志

- **2025-12-16**: 初始版本，支持 ScienceWorld 1.1.6 测试
