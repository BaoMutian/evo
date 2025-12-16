# 单轮问答统一格式数据集说明

## 概述

本目录包含经过统一处理的单轮交互问答的 MCQ（多选题）和 Math（数学）数据集，所有数据集都已转换为统一的 schema 格式。

## 统一 Schema

每条数据包含以下字段：

| 字段       | 类型   | 说明                                                               | 是否可为空         |
| ---------- | ------ | ------------------------------------------------------------------ | ------------------ |
| `id`       | string | 问题的唯一标识符                                                   | 否                 |
| `question` | string | 问题文本                                                           | 否                 |
| `solution` | string | 解题步骤或解释                                                     | 是（可为空字符串） |
| `answer`   | string | 正确答案的文本内容                                                 | 否                 |
| `options`  | array  | 所有选项的列表                                                     | 是（可为空列表） |
| `metadata` | object | 元数据信息（必须包含 domain、source、answer_letter、answer_index） | 否                 |

### Metadata 统一字段

所有数据集的 metadata 必须包含以下字段：

- `domain`: 学科领域（统一使用此字段，不再使用 category）
- `source`: 数据来源
- `answer_letter`: 答案对应的字母（A/B/C...），无选项时为空字符串
- `answer_index`: 答案在选项中的索引，无选项时为-1

## 数据集文件

### MCQ 数据集（多选题）

#### 1. GPQA-Diamond-198.jsonl

- **来源**: GPQA (Graduate-level Google-Proof Q&A)
- **数量**: 198 条
- **特点**:
  - 包含详细的解题步骤（solution 字段非空）
  - 通常有 4 个选项
  - **选项已随机打乱**，避免正确答案总是在第一位
  - 涵盖 Physics、Chemistry 等高级学科领域

**示例**:

```json
{
	"id": "rec06pnAkLOr2t2mp",
	"question": "Two quantum states with energies E1 and E2...",
	"solution": "According to the uncertainty principle...",
	"answer": "10^-4 eV",
	"options": ["10^-8 eV", "10^-11 eV", "10^-4 eV", "10^-9 eV"],
	"metadata": {
		"domain": "Physics",
		"source": "GPQA",
		"answer_letter": "C",
		"answer_index": 2
	}
}
```

#### 2. MMLU-Pro 系列

所有 MMLU-Pro 数据集共享相同的格式：

##### MMLU-Pro-economics-844.jsonl

- **数量**: 844 条
- **领域**: 经济学

##### MMLU-Pro-engineering-969.jsonl

- **数量**: 969 条
- **领域**: 工程学

##### MMLU-Pro-philosophy-499.jsonl

- **数量**: 499 条
- **领域**: 哲学

**MMLU-Pro 特点**:

- solution 字段通常为空（原始数据中的 cot_content 为空）
- 选项数量不固定（3-10 个选项）
- id 格式为 `{domain}_{question_id}`

**示例**:

```json
{
	"id": "economics_6826",
	"question": "Which of the following would be classified under C...",
	"solution": "",
	"answer": "$50.00 spent eating out at a restaurant",
	"options": ["The purchase of a new car...", "..."],
	"metadata": {
		"domain": "economics",
		"source": "ori_mmlu-high_school_macroeconomics",
		"answer_letter": "G",
		"answer_index": 6
	}
}
```

### Math 数据集（数学问题）

#### 3. AIME24-30.jsonl

- **来源**: AIME 2024 (American Invitational Mathematics Examination)
- **数量**: 30 条
- **特点**:
  - 包含详细的解题步骤
  - 答案为数值
  - 无选项（options 为空列表）
  - 高难度竞赛数学题

**示例**:

```json
{
	"id": "2024-II-4",
	"question": "Let $x,y$ and $z$ be positive real numbers...",
	"solution": "Denote $\\log_2(x) = a$...",
	"answer": "33",
	"options": [],
	"metadata": {
		"domain": "Mathematics",
		"source": "AIME-2024",
		"answer_letter": "",
		"answer_index": -1
	}
}
```

#### 4. AIME25-30.jsonl

- **来源**: AIME 2025
- **数量**: 30 条
- **特点**:
  - **无 solution 字段**（原始数据不包含解题步骤）
  - 包含 problem_type 信息作为子领域
  - 答案为数值
  - 无选项

**示例**:

```json
{
	"id": "aime25_1",
	"question": "Find the sum of all integer bases...",
	"solution": "",
	"answer": "70",
	"options": [],
	"metadata": {
		"domain": "Number Theory",
		"source": "AIME-2025",
		"answer_letter": "",
		"answer_index": -1
	}
}
```

#### 5. MATH-500.jsonl

- **来源**: MATH Dataset
- **数量**: 500 条
- **特点**:
  - 包含详细的解题步骤
  - 按 subject 分类（Algebra, Geometry, Precalculus 等）
  - 包含难度 level（1-5）
  - 答案格式多样（数值、表达式、坐标等）
  - 无选项

**示例**:

```json
{
	"id": "math_807",
	"question": "Convert the point $(0,3)$ in rectangular coordinates...",
	"solution": "We have that $r = \\sqrt{0^2 + 3^2} = 3.$...",
	"answer": "\\left( 3, \\frac{\\pi}{2} \\right)",
	"options": [],
	"metadata": {
		"domain": "Precalculus",
		"source": "MATH",
		"level": 2,
		"answer_letter": "",
		"answer_index": -1
	}
}
```

## 数据统计

### MCQ 数据集

| 文件名                         | 数据量   | 类型             |
| ------------------------------ | -------- | ---------------- |
| GPQA-Diamond-198.jsonl         | 198      | 多选题（有解释） |
| MMLU-Pro-economics-844.jsonl   | 844      | 多选题           |
| MMLU-Pro-engineering-969.jsonl | 969      | 多选题           |
| MMLU-Pro-philosophy-499.jsonl  | 499      | 多选题           |
| **MCQ 小计**                   | **2510** |                  |

### Math 数据集

| 文件名          | 数据量  | 类型               |
| --------------- | ------- | ------------------ |
| AIME24-30.jsonl | 30      | 数学竞赛（有解释） |
| AIME25-30.jsonl | 30      | 数学竞赛（无解释） |
| MATH-500.jsonl  | 500     | 数学题库（有解释） |
| **Math 小计**   | **560** |                    |

### 总计

**总数据量**: 3070 条

## 注意事项

1. 所有必需字段都存在，但某些字段可能为空（如 solution、options）
2. metadata 统一使用 domain 字段表示学科领域
3. 所有数据集的 metadata 包含 answer_letter 和 answer_index 字段
4. MCQ 数据集（有选项）：answer_letter 为 A-Z，answer_index 为 0 开始的索引
5. Math 数据集（无选项）：answer_letter 为空字符串，answer_index 为-1
6. GPQA 数据集的选项已随机打乱
7. Math 数据集的答案格式多样，可能是数值、表达式、坐标等
8. 所有文本使用 UTF-8 编码，支持 LaTeX 数学公式
9. 每个 JSON 对象占一行（JSONL 格式）