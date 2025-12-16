# ALFWorld LLM Agent 测试指南

## 目录

- [1. ALFWorld 简介](#1-alfworld-简介)
- [2. 环境架构](#2-环境架构)
- [3. 任务类型详解](#3-任务类型详解)
- [4. 交互命令](#4-交互命令)
- [5. 环境安装](#5-环境安装)
- [6. 测试脚本使用](#6-测试脚本使用)
- [7. 评估指标](#7-评估指标)
- [8. Prompt 设计](#8-prompt-设计)
- [9. 示例交互](#9-示例交互)
- [10. 常见问题](#10-常见问题)

---

## 1. ALFWorld 简介

### 1.1 什么是 ALFWorld？

**ALFWorld**（Aligning Text and Embodied Environments for Interactive Learning）是一个结合了文本游戏和具身AI的交互式学习环境。它基于两个重要项目：

- **ALFRED**（A Benchmark for Interpreting Grounded Instructions for Everyday Tasks）：一个视觉-语言导航与交互数据集
- **TextWorld**：微软开发的文本冒险游戏框架

ALFWorld 将 ALFRED 中的 3D 家居环境转换为纯文本交互格式，使得我们可以在不需要视觉渲染的情况下测试 Agent 的规划和推理能力。

### 1.2 为什么用 ALFWorld 测试 LLM？

| 优势 | 说明 |
|------|------|
| **多轮交互** | 测试 LLM 在多步骤任务中的规划能力 |
| **状态追踪** | 评估 LLM 对环境状态的理解和记忆 |
| **常识推理** | 任务需要家居常识（如"清洗物品需要水槽"） |
| **纠错能力** | 观察 LLM 在错误后能否调整策略 |
| **指令遵循** | 测试 LLM 对任务目标的理解 |

### 1.3 论文引用

```bibtex
@inproceedings{ALFWorld20,
  title={{ALFWorld: Aligning Text and Embodied Environments for Interactive Learning}},
  author={Mohit Shridhar and Xingdi Yuan and Marc-Alexandre Côté and 
          Yonatan Bisk and Adam Trischler and Matthew Hausknecht},
  booktitle={ICLR},
  year={2021}
}
```

---

## 2. 环境架构

### 2.1 数据目录结构

```
alfworld/data/
├── json_2.1.1/
│   ├── train/              # 训练集
│   ├── valid_seen/         # 验证集（见过的场景）
│   ├── valid_unseen/       # 验证集（未见场景）
│   └── valid_train/        # 训练验证集
├── logic/
│   ├── alfred.pddl         # PDDL 领域定义
│   └── alfred.twl2         # TextWorld 语法文件
└── detectors/
    └── mrcnn.pth           # MaskRCNN 检测器权重
```

### 2.2 游戏文件结构

每个任务实例包含以下文件：

```
pick_and_place_simple-Book-None-SideTable-329/
└── trial_T20190908_050633_745514/
    ├── game.tw-pddl        # 游戏配置文件（包含 PDDL 和语法）
    ├── initial_state.pddl  # 初始状态定义
    └── traj_data.json      # 任务轨迹数据
```

### 2.3 环境交互流程

```
┌─────────────┐     观察 (obs)      ┌─────────────┐
│             │ ──────────────────> │             │
│   ALFWorld  │                     │  LLM Agent  │
│  Environment│ <────────────────── │             │
│             │     动作 (action)   │             │
└─────────────┘                     └─────────────┘
       │                                   │
       │  info['admissible_commands']      │
       │  info['won']                      │
       └───────────────────────────────────┘
```

---

## 3. 任务类型详解

ALFWorld 包含 **6 种任务类型**，覆盖不同的家居场景：

### 3.1 pick_and_place_simple（拾取放置）

**目标**：将物品从 A 位置移动到 B 位置

**示例任务**：
```
Your task is to: put some book on sidetable.
```

**典型解决步骤**：
1. `look` - 查看周围环境
2. `go to bed 1` - 前往床
3. `take book 1 from bed 1` - 拿起书
4. `go to sidetable 1` - 前往边桌
5. `move book 1 to sidetable 1` - 放下书

---

### 3.2 look_at_obj_in_light（灯下检查）

**目标**：在灯光下检查物品（打开灯，拿着物品）

**示例任务**：
```
Your task is to: examine the alarmclock with the desklamp.
```

**典型解决步骤**：
1. `go to desk 1` - 前往书桌
2. `take alarmclock 1 from desk 1` - 拿起闹钟
3. `use desklamp 1` - 打开台灯
4. 任务完成！

---

### 3.3 pick_clean_then_place_in_recep（清洗放置）

**目标**：用水槽清洗物品后放到指定位置

**示例任务**：
```
Your task is to: clean some mug and put it in coffeemachine.
```

**典型解决步骤**：
1. `go to countertop 1` - 前往台面
2. `take mug 1 from countertop 1` - 拿起杯子
3. `go to sinkbasin 1` - 前往水槽
4. `clean mug 1 with sinkbasin 1` - 清洗杯子
5. `go to coffeemachine 1` - 前往咖啡机
6. `move mug 1 to coffeemachine 1` - 放下杯子

---

### 3.4 pick_heat_then_place_in_recep（加热放置）

**目标**：用微波炉加热物品后放到指定位置

**示例任务**：
```
Your task is to: heat some egg and put it in fridge.
```

**典型解决步骤**：
1. 找到并拿起鸡蛋
2. `go to microwave 1` - 前往微波炉
3. `heat egg 1 with microwave 1` - 加热鸡蛋
4. `go to fridge 1` - 前往冰箱
5. `move egg 1 to fridge 1` - 放入冰箱

---

### 3.5 pick_cool_then_place_in_recep（冷却放置）

**目标**：用冰箱冷却物品后放到指定位置

**示例任务**：
```
Your task is to: cool some apple and put it in countertop.
```

**典型解决步骤**：
1. 找到并拿起苹果
2. `go to fridge 1` - 前往冰箱
3. `cool apple 1 with fridge 1` - 冷却苹果
4. `go to countertop 1` - 前往台面
5. `move apple 1 to countertop 1` - 放下苹果

---

### 3.6 pick_two_obj_and_place（双物品放置）

**目标**：将两个相同类型的物品放到指定位置

**示例任务**：
```
Your task is to: put two cellphone in drawer.
```

**典型解决步骤**：
1. 找到并拿起第一个手机
2. 放到抽屉
3. 找到并拿起第二个手机
4. 放到抽屉

---

## 4. 交互命令

### 4.1 导航命令

| 命令 | 格式 | 示例 | 说明 |
|------|------|------|------|
| look | `look` | `look` | 查看当前位置周围的物品和可到达的位置 |
| go to | `go to [receptacle]` | `go to dresser 1` | 移动到指定容器/位置 |

### 4.2 物品操作

| 命令 | 格式 | 示例 | 说明 |
|------|------|------|------|
| take | `take [object] from [receptacle]` | `take apple 1 from fridge 1` | 从容器拿起物品 |
| move | `move [object] to [receptacle]` | `move apple 1 to countertop 1` | 放下物品到容器 |
| inventory | `inventory` | `inventory` | 查看当前携带的物品 |

### 4.3 容器操作

| 命令 | 格式 | 示例 | 说明 |
|------|------|------|------|
| open | `open [receptacle]` | `open fridge 1` | 打开可开关的容器 |
| close | `close [receptacle]` | `close drawer 1` | 关闭容器 |

### 4.4 物品处理

| 命令 | 格式 | 示例 | 说明 |
|------|------|------|------|
| heat | `heat [object] with [receptacle]` | `heat potato 1 with microwave 1` | 用微波炉加热 |
| clean | `clean [object] with [receptacle]` | `clean mug 1 with sinkbasin 1` | 用水槽清洗 |
| cool | `cool [object] with [receptacle]` | `cool apple 1 with fridge 1` | 用冰箱冷却 |

### 4.5 其他命令

| 命令 | 格式 | 示例 | 说明 |
|------|------|------|------|
| use | `use [object]` | `use desklamp 1` | 使用/切换物品状态（如开灯） |
| examine | `examine [object/receptacle]` | `examine apple 1` | 检查物品详情 |

### 4.6 重要规则

> ⚠️ **Agent 每次只能携带一个物品**
> 
> ⚠️ **必须先 `go to` 某位置才能与那里的物品交互**
> 
> ⚠️ **某些容器（如冰箱、抽屉）需要先 `open` 才能看到/取出里面的物品**

---

## 5. 环境安装

### 5.1 快速安装

```bash
# 1. Clone Repo
git clone https://github.com/alfworld/alfworld.git alfworld
cd alfworld

# 2. 从本地仓库安装 ALFWorld
pip install -e .

# 3. 下载 PDDL & Game Files
export ALFWORLD_DATA=<storage_path>
python scripts/alfworld-download
```

### 5.2 验证安装

```bash
python3 -c "
import alfworld
import textworld
print('✅ ALFWorld 安装成功')
"
```

---

## 6. 测试脚本使用

### 6.1 脚本位置

```
/home/bmt/evo/bench/alfworld_test.py
```

### 6.2 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model` | str | `qwen/qwen3-8b` | OpenRouter 上的模型标识 |
| `--num_games` | int | `5` | 测试的游戏数量 |
| `--task_types` | int[] | `1-6` | 任务类型 ID 列表 |
| `--no_few_shot` | flag | False | 禁用 few-shot 示例 |
| `--quiet` | flag | False | 减少输出（只显示结果） |
| `--output` | str | 自动生成 | 结果保存的 JSON 文件路径 |
| `--demo` | flag | - | 运行单个游戏演示 |
| `--game_idx` | int | `0` | 演示模式下的游戏索引 |
| `--seed` | int | `42` | 随机种子（可复现的游戏选择） |
| `--no_seed` | flag | False | 不设置随机种子（完全随机，结果不可复现） |

### 6.3 使用示例

#### 运行单个游戏演示

```bash
python3 alfworld_test.py --demo --model "qwen/qwen3-8b"
```

#### 运行完整测试

```bash
# 测试 5 个游戏
python3 alfworld_test.py --model "qwen/qwen3-8b" --num_games 5

# 测试 10 个游戏，只测试拾取放置任务
python3 alfworld_test.py --model "qwen/qwen3-8b" --num_games 10 --task_types 1

# 安静模式，只显示最终结果
python3 alfworld_test.py --model "qwen/qwen3-8b" --num_games 20 --quiet

# 指定输出文件
python3 alfworld_test.py --model "qwen/qwen3-8b" --output results.json
```

#### 测试不同模型

```bash
# 测试 Claude
python3 alfworld_test.py --model "anthropic/claude-3.5-sonnet"

# 测试 GPT-4
python3 alfworld_test.py --model "openai/gpt-4-turbo"

# 测试 DeepSeek
python3 alfworld_test.py --model "deepseek/deepseek-chat-v3-0324"
```

### 6.4 输出文件格式

测试完成后会生成 JSON 格式的结果文件：

```json
{
  "model": "qwen/qwen3-8b",
  "timestamp": "2025-12-16T10:30:00",
  "config": {
    "num_games": 5,
    "task_types": [1, 2, 3, 4, 5, 6],
    "use_few_shot": true,
    "max_steps": 30,
    "temperature": 0.3
  },
  "summary": {
    "total_games": 5,
    "successes": 3,
    "success_rate": 0.6,
    "avg_steps": 12.4
  },
  "results": [
    {
      "success": true,
      "steps": 8,
      "actions": ["look", "go to bed 1", "take book 1 from bed 1", ...],
      "observations": [...],
      "game_file": "/path/to/game.tw-pddl"
    },
    ...
  ]
}
```

---

## 7. 评估指标

### 7.1 主要指标

| 指标 | 说明 | 计算方式 |
|------|------|----------|
| **成功率 (Success Rate)** | 成功完成任务的比例 | `成功数 / 总任务数` |
| **平均步数 (Avg Steps)** | 完成任务的平均步数 | `总步数 / 任务数` |
| **成功任务平均步数** | 成功完成任务的平均步数 | 只计算成功的任务 |

### 7.2 分任务类型统计

可以按任务类型分析模型表现：

```python
# 按任务类型测试
python3 alfworld_test.py --task_types 1 --num_games 10  # 拾取放置
python3 alfworld_test.py --task_types 2 --num_games 10  # 灯下检查
python3 alfworld_test.py --task_types 3 --num_games 10  # 清洗放置
```

### 7.3 业界基准参考

| 方法 | 成功率 |
|------|--------|
| Random Agent | ~5% |
| BUTLER (TextDAgger) | ~26% |
| ReAct (GPT-3) | ~45% |
| ReAct (GPT-4) | ~70%+ |

---

## 8. Prompt 设计

### 8.1 ReAct 风格

测试脚本采用 **ReAct（Reasoning + Acting）** 风格的 prompt，要求模型输出：

```
THINK: [对当前观察的推理]
ACTION: [执行的动作]
```

### 8.2 Prompt 结构

```
┌─────────────────────────────────────┐
│  SYSTEM_PROMPT                      │  <- 介绍环境和可用动作
├─────────────────────────────────────┤
│  FEW_SHOT_EXAMPLES (可选)           │  <- 示例交互
├─────────────────────────────────────┤
│  TASK_PROMPT                        │  <- 当前任务观察
├─────────────────────────────────────┤
│  CONVERSATION_HISTORY               │  <- 之前的交互历史
└─────────────────────────────────────┘
```

### 8.3 Few-shot 示例

脚本包含两个内置示例：
1. **拾取放置任务** - 演示基本的导航和物品操作
2. **灯下检查任务** - 演示设备使用

---

## 9. 示例交互

### 9.1 成功案例：拾取放置

```
=== NEW TASK ===
-= Welcome to TextWorld, ALFRED! =-

You are in the middle of a room. Looking quickly around you, you see 
a bed 1, a desk 1, a drawer 1, a drawer 2, a garbagecan 1, a sidetable 1.

Your task is to: put some book on sidetable.

--- Step 1 ---
LLM Think: I need to find a book first. Let me check the bed since books are often there.
LLM Action: go to bed 1
环境响应: On the bed 1, you see a book 1, a laptop 1, a pillow 1.

--- Step 2 ---
LLM Think: Found a book on the bed. Let me take it.
LLM Action: take book 1 from bed 1
环境响应: You pick up the book 1 from the bed 1.

--- Step 3 ---
LLM Think: I have the book now. Let me go to the sidetable to complete the task.
LLM Action: go to sidetable 1
环境响应: On the sidetable 1, you see a alarmclock 1, a pen 1.

--- Step 4 ---
LLM Think: I'm at the sidetable. Now I'll put the book here.
LLM Action: move book 1 to sidetable 1
环境响应: You move the book 1 to the sidetable 1.

🎉 任务完成! 步数: 4
```

### 9.2 失败案例：需要开门

```
--- Step 5 ---
LLM Think: Let me check the fridge for an apple.
LLM Action: go to fridge 1
环境响应: The fridge 1 is closed.

--- Step 6 ---
LLM Think: The fridge is closed. I need to open it first.
LLM Action: take apple 1 from fridge 1  ❌ 错误！应该先 open
环境响应: Nothing happens.

--- Step 7 ---
LLM Think: I couldn't take the apple. Let me open the fridge first.
LLM Action: open fridge 1
环境响应: You open the fridge 1. The fridge 1 is open. In it, you see 
a apple 1, a cup 1, a egg 1.
```

---

## 10. 常见问题

### Q1: 为什么模型的动作不在可用命令中？

**原因**：模型生成的命令格式可能与环境期望的略有不同。

**解决**：脚本会自动尝试模糊匹配，但建议：
- 使用精确的物品名称（如 `apple 1` 而非 `apple`）
- 检查 prompt 中的示例是否正确

### Q2: 如何提高成功率？

1. **使用 few-shot 示例**（默认启用）
2. **降低温度参数**（当前为 0.3）
3. **使用更强的模型**（如 GPT-4）
4. **增加 max_steps**（当前为 30）

### Q3: 如何测试自己的模型？

只需确保模型可以通过 OpenRouter API 调用：

```bash
python3 alfworld_test.py --model "your-provider/your-model"
```

### Q4: 内存不足怎么办？

减少 `--num_games` 或使用 `--quiet` 模式减少日志输出。

### Q5: 如何添加自定义 prompt？

编辑 `alfworld_test.py` 中的以下变量：
- `SYSTEM_PROMPT` - 系统提示
- `FEW_SHOT_EXAMPLES` - 示例
- `TASK_PROMPT_TEMPLATE` - 任务模板

---

## 附录：任务类型 ID 对照表

| ID | 英文名称 | 中文名称 |
|----|---------|---------|
| 1 | pick_and_place_simple | 简单拾取放置 |
| 2 | look_at_obj_in_light | 灯下检查 |
| 3 | pick_clean_then_place_in_recep | 清洗后放置 |
| 4 | pick_heat_then_place_in_recep | 加热后放置 |
| 5 | pick_cool_then_place_in_recep | 冷却后放置 |
| 6 | pick_two_obj_and_place | 双物品放置 |

---

## 更新日志

- **2025-12-16**: 初始版本，支持 qwen3-8b 测试

