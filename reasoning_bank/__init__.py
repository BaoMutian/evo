"""
ReasoningBank: Scaling Agent Self-Evolving with Reasoning Memory

一个具备自我进化能力的智能体框架，通过推理记忆库（ReasoningBank）
和记忆感知测试时扩展（MaTTS）机制，实现持续学习。
"""

__version__ = "0.1.0"
__author__ = "ReasoningBank Team"

from reasoning_bank.core.memory import MemoryBank
from reasoning_bank.core.agent import ReActAgent
from reasoning_bank.core.extractor import MemoryExtractor

__all__ = [
    "MemoryBank",
    "ReActAgent", 
    "MemoryExtractor",
]

