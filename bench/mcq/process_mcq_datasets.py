#!/usr/bin/env python3
"""
统一处理MCQ和Math数据集，将不同格式的数据集转换为统一的schema。

统一格式包含以下字段：
- id: 问题的唯一标识
- question: 问题文本
- solution: 解题步骤或解释（可为空）
- answer: 答案
- options: 选项列表（可为空列表）
- metadata: 元数据（必须包含domain、source、answer_letter、answer_index）
"""

import json
import os
import random
from pathlib import Path
from typing import Dict, List, Any


def process_gpqa_dataset(input_file: str, output_file: str):
    """
    处理GPQA数据集
    
    原始schema:
    - Question
    - Correct Answer
    - Incorrect Answer 1, 2, 3
    - Explanation
    - Record ID
    - High-level domain
    """
    print(f"处理GPQA数据集: {input_file}")
    
    processed_data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f, 1):
            item = json.loads(line.strip())
            
            correct_answer = item.get('Correct Answer', '').strip()
            
            # 构造选项列表（包括正确答案和错误答案）
            options = [
                correct_answer,
                item.get('Incorrect Answer 1', ''),
                item.get('Incorrect Answer 2', ''),
                item.get('Incorrect Answer 3', '')
            ]
            # 移除空选项
            options = [opt.strip() for opt in options if opt and opt.strip()]
            
            # 随机打乱选项顺序，避免答案总是在第一位
            random.shuffle(options)
            
            # 计算正确答案在打乱后的位置
            answer_index = options.index(correct_answer)
            answer_letter = chr(ord('A') + answer_index)
            
            processed_item = {
                'id': item.get('Record ID', f'gpqa_{idx}'),
                'question': item.get('Question', '').strip(),
                'solution': item.get('Explanation', '').strip(),
                'answer': correct_answer,
                'options': options,
                'metadata': {
                    'domain': item.get('High-level domain', ''),
                    'source': 'GPQA',
                    'answer_letter': answer_letter,
                    'answer_index': answer_index
                }
            }
            
            processed_data.append(processed_item)
    
    # 写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in processed_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"  ✓ 处理完成: {len(processed_data)} 条数据 -> {output_file}")
    return len(processed_data)


def process_mmlu_pro_dataset(input_file: str, output_file: str):
    """
    处理MMLU-Pro数据集
    
    原始schema:
    - question_id
    - question
    - options (列表)
    - answer (字母)
    - answer_index
    - cot_content
    - category
    - src
    """
    print(f"处理MMLU-Pro数据集: {input_file}")
    
    processed_data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f, 1):
            item = json.loads(line.strip())
            
            # 获取选项和答案
            options = item.get('options', [])
            answer_index = item.get('answer_index', -1)
            answer_text = ''
            
            # 根据answer_index获取实际答案文本
            if answer_index >= 0 and answer_index < len(options):
                answer_text = options[answer_index]
            else:
                # 如果没有answer_index，尝试使用answer字段（字母）
                answer_letter = item.get('answer', '')
                if answer_letter and options:
                    # 将字母转换为索引 (A=0, B=1, ...)
                    letter_index = ord(answer_letter) - ord('A')
                    if 0 <= letter_index < len(options):
                        answer_text = options[letter_index]
            
            # 构造solution（使用cot_content或为空）
            solution = item.get('cot_content', '').strip()
            
            processed_item = {
                'id': f"{item.get('category', 'mmlu')}_{item.get('question_id', idx)}",
                'question': item.get('question', '').strip(),
                'solution': solution,
                'answer': answer_text,
                'options': options,
                'metadata': {
                    'domain': item.get('category', ''),
                    'source': item.get('src', 'MMLU-Pro'),
                    'answer_letter': item.get('answer', ''),
                    'answer_index': answer_index
                }
            }
            
            processed_data.append(processed_item)
    
    # 写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in processed_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"  ✓ 处理完成: {len(processed_data)} 条数据 -> {output_file}")
    return len(processed_data)


def process_aime24_dataset(input_file: str, output_file: str):
    """
    处理AIME24数据集
    
    原始schema:
    - ID
    - Problem
    - Solution
    - Answer (数值)
    """
    print(f"处理AIME24数据集: {input_file}")
    
    processed_data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f, 1):
            item = json.loads(line.strip())
            
            # 转换答案为字符串
            answer = str(item.get('Answer', ''))
            
            processed_item = {
                'id': item.get('ID', f'aime24_{idx}'),
                'question': item.get('Problem', '').strip(),
                'solution': item.get('Solution', '').strip(),
                'answer': answer,
                'options': [],  # AIME问题没有选项
                'metadata': {
                    'domain': 'Mathematics',
                    'source': 'AIME-2024',
                    'answer_letter': '',
                    'answer_index': -1
                }
            }
            
            processed_data.append(processed_item)
    
    # 写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in processed_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"  ✓ 处理完成: {len(processed_data)} 条数据 -> {output_file}")
    return len(processed_data)


def process_aime25_dataset(input_file: str, output_file: str):
    """
    处理AIME25数据集
    
    原始schema:
    - problem_idx
    - problem
    - answer (数值)
    - problem_type (列表)
    """
    print(f"处理AIME25数据集: {input_file}")
    
    processed_data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f, 1):
            item = json.loads(line.strip())
            
            # 转换答案为字符串
            answer = str(item.get('answer', ''))
            
            # 获取problem_type作为子domain
            problem_types = item.get('problem_type', [])
            subdomain = ', '.join(problem_types) if problem_types else 'Mathematics'
            
            processed_item = {
                'id': f"aime25_{item.get('problem_idx', idx)}",
                'question': item.get('problem', '').strip(),
                'solution': '',  # AIME25没有solution
                'answer': answer,
                'options': [],
                'metadata': {
                    'domain': subdomain,
                    'source': 'AIME-2025',
                    'answer_letter': '',
                    'answer_index': -1
                }
            }
            
            processed_data.append(processed_item)
    
    # 写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in processed_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"  ✓ 处理完成: {len(processed_data)} 条数据 -> {output_file}")
    return len(processed_data)


def process_math_dataset(input_file: str, output_file: str):
    """
    处理MATH数据集
    
    原始schema:
    - problem
    - solution
    - answer (字符串)
    - subject
    - level
    - unique_id
    """
    print(f"处理MATH数据集: {input_file}")
    
    processed_data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f, 1):
            item = json.loads(line.strip())
            
            # 使用unique_id或生成一个
            unique_id = item.get('unique_id', f'math_{idx}')
            # 从unique_id中提取简短的id
            if '/' in unique_id:
                short_id = unique_id.split('/')[-1].replace('.json', '')
            else:
                short_id = unique_id
            
            processed_item = {
                'id': f"math_{short_id}",
                'question': item.get('problem', '').strip(),
                'solution': item.get('solution', '').strip(),
                'answer': str(item.get('answer', '')),
                'options': [],  # MATH问题没有选项
                'metadata': {
                    'domain': item.get('subject', 'Mathematics'),
                    'source': 'MATH',
                    'level': item.get('level', 0),
                    'answer_letter': '',
                    'answer_index': -1
                }
            }
            
            processed_data.append(processed_item)
    
    # 写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in processed_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"  ✓ 处理完成: {len(processed_data)} 条数据 -> {output_file}")
    return len(processed_data)


def main():
    # 设置输入输出目录
    mcq_dir = Path('/home/bmt/evo/bench/mcq')
    math_dir = Path('/home/bmt/evo/bench/math')
    output_dir = Path('/home/bmt/evo/bench/single_turn_bench')
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("开始处理数据集")
    print("=" * 60)
    
    total_count = 0
    
    # 处理MCQ数据集
    print("\n>>> 处理MCQ数据集")
    for input_file in sorted(mcq_dir.glob('*.jsonl')):
        file_name = input_file.name
        output_file = output_dir / file_name
        
        # 跳过大文件
        if file_name == 'MMLU-Pro-12032.jsonl':
            print(f"跳过大文件: {file_name}")
            continue
        
        try:
            if file_name.startswith('GPQA'):
                count = process_gpqa_dataset(str(input_file), str(output_file))
            elif file_name.startswith('MMLU-Pro'):
                count = process_mmlu_pro_dataset(str(input_file), str(output_file))
            else:
                print(f"跳过未知格式文件: {file_name}")
                continue
            
            total_count += count
            
        except Exception as e:
            print(f"  ✗ 处理失败: {file_name}")
            print(f"    错误: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # 处理Math数据集
    print("\n>>> 处理Math数据集")
    for input_file in sorted(math_dir.glob('*.jsonl')):
        file_name = input_file.name
        output_file = output_dir / file_name
        
        try:
            if file_name.startswith('AIME24'):
                count = process_aime24_dataset(str(input_file), str(output_file))
            elif file_name.startswith('AIME25'):
                count = process_aime25_dataset(str(input_file), str(output_file))
            elif file_name.startswith('MATH'):
                count = process_math_dataset(str(input_file), str(output_file))
            else:
                print(f"跳过未知格式文件: {file_name}")
                continue
            
            total_count += count
            
        except Exception as e:
            print(f"  ✗ 处理失败: {file_name}")
            print(f"    错误: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"处理完成! 总共处理了 {total_count} 条数据")
    print(f"输出目录: {output_dir}")
    print("=" * 60)
    
    # 显示输出目录中的文件
    print("\n生成的文件:")
    for output_file in sorted(output_dir.glob('*.jsonl')):
        # 统计行数
        with open(output_file, 'r') as f:
            line_count = sum(1 for _ in f)
        print(f"  - {output_file.name}: {line_count} 条数据")


if __name__ == '__main__':
    main()
