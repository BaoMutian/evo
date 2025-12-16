# ReasoningBank

> 🧠 自我进化智能体框架 - 基于论文 *ReasoningBank: Scaling Agent Self-Evolving with Reasoning Memory*

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## ✨ 特性

- 🎯 **从成功中学习**：提取有效的问题解决策略
- 🛡️ **从失败中学习**：提取"避坑指南"，防止重复错误
- 🔍 **记忆增强决策**：利用历史经验指导新任务求解
- 📈 **持续进化**：随着任务积累，能力不断提升
- ⚡ **MaTTS 扩展**：通过测试时计算获取高质量经验

---

## 🚀 快速开始

### 安装

```bash
# 克隆项目
git clone <repository-url>
cd reasoning-bank

# 创建环境
conda create -n icml26 python=3.10
conda activate icml26

# 安装依赖
pip install -r reasoning_bank/requirements.txt

# 配置 API Key
echo "OPENROUTER_API_KEY=your_key" > .env
```

### 运行

```bash
# 基础测试（5 道数学题）
python -m reasoning_bank.main --dataset math500 --num-tasks 5

# 启用自我进化
python -m reasoning_bank.main --dataset math500 --num-tasks 50 --use-memory

# MaTTS 并行扩展（更高质量记忆）
python -m reasoning_bank.main --dataset math500 --matts parallel --use-memory
```

---

## 📊 支持的数据集

| 数据集 | 类型 | 数量 | 命令 |
|--------|------|------|------|
| MATH-500 | 数学 | 500 | `--dataset math500` |
| AIME 2024 | 竞赛 | 30 | `--dataset aime24` |
| AIME 2025 | 竞赛 | 30 | `--dataset aime25` |
| GPQA Diamond | 研究生级 | 198 | `--dataset gpqa` |
| MMLU-Pro | 选择题 | 2312 | `--dataset mmlu_*` |

---

## 🔧 核心命令

```bash
# 查看帮助
python -m reasoning_bank.main --help

# 使用不同模型
python -m reasoning_bank.main -d gpqa -m "anthropic/claude-3-sonnet" --use-memory

# 清空记忆重新开始
python -m reasoning_bank.main -d math500 --use-memory --clear-memory

# 详细输出模式
python -m reasoning_bank.main -d math500 -n 3 --verbose
```

---

## 📁 项目结构

```
reasoning_bank/
├── config/config.yaml    # 配置文件
├── core/                 # 核心模块
│   ├── llm_service.py   # LLM 封装
│   ├── memory.py        # 记忆库
│   ├── agent.py         # ReAct Agent
│   └── extractor.py     # 记忆提取
├── envs/                 # 环境适配器
├── prompts/              # Prompt 模板
├── workflows/            # 工作流
│   ├── evolution.py     # 进化循环
│   └── matts.py         # MaTTS
├── utils/                # 工具函数
└── main.py               # CLI 入口
```

---

## 📖 详细文档

完整的用户手册请参阅：[docs/USER_MANUAL.md](docs/USER_MANUAL.md)

包含：
- 完整 API 参考
- 高级配置说明
- MaTTS 使用指南
- 扩展开发教程
- 常见问题解答

---

## 🔬 核心工作流

```
新任务 ──> [检索记忆] ──> [执行任务] ──> [评估结果]
                                              │
           ┌──────────────────────────────────┘
           ▼
         成功? ─┬─ Yes ──> [提取成功策略] ──┐
               └─ No  ──> [提取失败教训] ──┤
                                            ▼
                                     [存入记忆库] ──> 下一任务
```

---

## 📚 参考文献

```bibtex
@article{reasoningbank2024,
  title={ReasoningBank: Scaling Agent Self-Evolving with Reasoning Memory},
  author={...},
  journal={arXiv preprint arXiv:2509.25140},
  year={2024}
}
```

---

## 📝 许可证

MIT License
