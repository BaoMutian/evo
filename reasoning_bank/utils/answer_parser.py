"""
答案解析模块

用于从模型输出中提取答案，支持数学表达式、选择题等多种格式
"""

import re
from typing import Optional, Tuple


def extract_boxed_answer(text: str) -> Optional[str]:
    """从 LaTeX \\boxed{} 中提取答案
    
    Args:
        text: 包含 \\boxed{} 的文本
        
    Returns:
        提取的答案，如果没有找到返回 None
    """
    # 匹配 \boxed{...}，支持嵌套大括号
    pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
    matches = re.findall(pattern, text)
    
    if matches:
        # 返回最后一个 boxed 内容
        return matches[-1].strip()
    
    return None


def extract_final_answer(text: str) -> Optional[str]:
    """从模型输出中提取最终答案
    
    支持多种格式:
    - \\boxed{answer}
    - The answer is: answer
    - Final Answer: answer
    - Answer: answer
    - 最后一行数字/表达式
    
    Args:
        text: 模型输出文本
        
    Returns:
        提取的答案
    """
    # 优先尝试 boxed
    boxed = extract_boxed_answer(text)
    if boxed:
        return boxed
    
    # 尝试匹配常见答案格式
    patterns = [
        r'(?:The |the )?(?:final |Final )?[Aa]nswer(?:\s+is)?[:\s]+(.+?)(?:\.|$)',
        r'(?:Therefore|thus|Hence|So),?\s+(?:the answer is\s+)?(.+?)(?:\.|$)',
        r'=\s*(.+?)(?:\s*$|\s*\.)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            answer = match.group(1).strip()
            # 清理答案
            answer = answer.strip('.')
            if answer:
                return answer
    
    # 尝试提取最后一行的数学表达式
    lines = text.strip().split('\n')
    for line in reversed(lines):
        line = line.strip()
        # 跳过空行和纯文字行
        if not line or line.startswith('Therefore') or line.startswith('So'):
            continue
        # 如果是数字或简单表达式
        if re.match(r'^[\d\s\+\-\*/\^\(\)\.\,\\a-zA-Z]+$', line):
            return line
    
    return None


def extract_choice_answer(text: str) -> Optional[str]:
    """从模型输出中提取选择题答案
    
    Args:
        text: 模型输出文本
        
    Returns:
        选项字母（A, B, C, D 等）
    """
    # 常见的选择题答案格式
    patterns = [
        r'(?:The |the )?(?:correct |best )?(?:answer|choice|option)(?:\s+is)?[:\s]+\(?([A-Za-z])\)?',
        r'\b([A-Z])\s*[\.\)]\s*(?:is correct|is the answer)',
        r'(?:I (?:would )?(?:choose|select|pick))\s+\(?([A-Za-z])\)?',
        r'^\s*\(?([A-Z])\)?\s*$',  # 单独一行的选项字母
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).upper()
    
    # 如果没有明确答案，查找最后出现的独立字母
    matches = re.findall(r'\b([A-J])\b', text)
    if matches:
        return matches[-1].upper()
    
    return None


def normalize_math_answer(answer: str) -> str:
    """规范化数学答案
    
    Args:
        answer: 原始答案字符串
        
    Returns:
        规范化后的答案
    """
    if not answer:
        return ""
    
    # 移除多余空格
    answer = " ".join(answer.split())
    
    # 移除 $ 符号
    answer = answer.replace("$", "")
    
    # 规范化分数
    answer = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'(\1)/(\2)', answer)
    
    # 规范化常见 LaTeX
    replacements = [
        (r'\\pi', 'π'),
        (r'\\sqrt', '√'),
        (r'\\times', '×'),
        (r'\\cdot', '·'),
        (r'\\left', ''),
        (r'\\right', ''),
        (r'\\,', ' '),
        (r'\\;', ' '),
        (r'\\!', ''),
    ]
    
    for old, new in replacements:
        answer = re.sub(old, new, answer)
    
    return answer.strip()


def compare_answers(
    prediction: str,
    ground_truth: str,
    is_mcq: bool = False,
) -> bool:
    """比较预测答案和标准答案
    
    Args:
        prediction: 预测答案
        ground_truth: 标准答案
        is_mcq: 是否为选择题
        
    Returns:
        是否匹配
    """
    if not prediction or not ground_truth:
        return False
    
    # 选择题：只比较字母
    if is_mcq:
        pred_letter = extract_choice_answer(prediction)
        gt_letter = ground_truth.strip().upper()
        return pred_letter == gt_letter
    
    # 数学题：规范化后比较
    pred_norm = normalize_math_answer(prediction)
    gt_norm = normalize_math_answer(ground_truth)
    
    # 完全匹配
    if pred_norm.lower() == gt_norm.lower():
        return True
    
    # 数值比较
    try:
        # 尝试转换为数值
        pred_val = float(eval(pred_norm.replace('π', str(3.14159265359))))
        gt_val = float(eval(gt_norm.replace('π', str(3.14159265359))))
        return abs(pred_val - gt_val) < 1e-6
    except:
        pass
    
    return False


def parse_react_response(response: str) -> Tuple[str, str]:
    """解析 ReAct 格式的响应
    
    Args:
        response: 模型响应文本
        
    Returns:
        (thought, action) 元组
    """
    thought = ""
    action = ""
    
    # 清理响应
    response = response.strip()
    
    # 尝试解析 THINK 和 ACTION
    lines = response.split('\n')
    
    for line in lines:
        line_stripped = line.strip()
        
        # 解析 THINK/Thought
        if line_stripped.upper().startswith('THINK:'):
            thought = line_stripped[6:].strip()
        elif line_stripped.upper().startswith('THOUGHT:'):
            thought = line_stripped[8:].strip()
        
        # 解析 ACTION/Act
        if line_stripped.upper().startswith('ACTION:'):
            action = line_stripped[7:].strip()
            break
        elif line_stripped.upper().startswith('ACT:'):
            action = line_stripped[4:].strip()
            break
    
    # 如果没找到 ACTION，尝试提取最后一行非思考内容
    if not action:
        for line in reversed(lines):
            line = line.strip()
            if line and not line.upper().startswith(('THINK', 'THOUGHT')):
                action = line
                break
    
    return thought, action

