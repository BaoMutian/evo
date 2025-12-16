"""
日志管理模块

提供统一的日志配置和输出
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

from colorama import init, Fore, Style

# 初始化 colorama（Windows 兼容）
init(autoreset=True)

# 全局 logger 存储
_loggers: dict = {}


class ColoredFormatter(logging.Formatter):
    """带颜色的日志格式化器"""
    
    COLORS = {
        logging.DEBUG: Fore.CYAN,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.RED + Style.BRIGHT,
    }
    
    def format(self, record):
        # 添加颜色
        color = self.COLORS.get(record.levelno, "")
        record.levelname = f"{color}{record.levelname}{Style.RESET_ALL}"
        
        # 对于特定标签添加颜色
        if hasattr(record, "tag"):
            record.tag = f"{Fore.BLUE}[{record.tag}]{Style.RESET_ALL}"
        
        return super().format(record)


def setup_logger(
    name: str = "reasoning_bank",
    level: str = "INFO",
    log_dir: Optional[str] = None,
    log_to_file: bool = True,
    log_to_console: bool = True,
) -> logging.Logger:
    """设置日志器
    
    Args:
        name: 日志器名称
        level: 日志级别
        log_dir: 日志目录
        log_to_file: 是否输出到文件
        log_to_console: 是否输出到控制台
        
    Returns:
        配置好的日志器
    """
    logger = logging.getLogger(name)
    
    # 如果已经配置过，直接返回
    if logger.handlers:
        return logger
    
    logger.setLevel(getattr(logging, level.upper()))
    
    # 日志格式
    file_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    console_format = "%(asctime)s | %(levelname)-8s | %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    # 控制台处理器
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(ColoredFormatter(console_format, date_format))
        logger.addHandler(console_handler)
    
    # 文件处理器
    if log_to_file and log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_path / f"{name}_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(file_format, date_format))
        logger.addHandler(file_handler)
    
    # 防止日志传播到父 logger
    logger.propagate = False
    
    _loggers[name] = logger
    return logger


def get_logger(name: str = "reasoning_bank") -> logging.Logger:
    """获取日志器
    
    Args:
        name: 日志器名称
        
    Returns:
        日志器实例
    """
    if name in _loggers:
        return _loggers[name]
    
    # 如果不存在，创建一个基础的
    return setup_logger(name)


class TaskLogger:
    """任务专用日志器，用于记录任务执行过程"""
    
    def __init__(self, task_id: str, logger: Optional[logging.Logger] = None):
        self.task_id = task_id
        self.logger = logger or get_logger()
        self.steps: list = []
    
    def log_step(
        self,
        step_num: int,
        observation: str,
        thought: str,
        action: str,
        result: str = "",
    ):
        """记录一步执行"""
        step_info = {
            "step": step_num,
            "observation": observation,
            "thought": thought,
            "action": action,
            "result": result,
        }
        self.steps.append(step_info)
        
        self.logger.info(f"[Task {self.task_id}] Step {step_num}")
        self.logger.debug(f"  Observation: {observation[:100]}...")
        self.logger.debug(f"  Thought: {thought[:100]}...")
        self.logger.info(f"  Action: {action}")
    
    def log_memory_retrieval(self, query: str, memories: list):
        """记录记忆检索"""
        self.logger.info(f"[Task {self.task_id}] Memory Retrieval")
        self.logger.debug(f"  Query: {query[:100]}...")
        self.logger.info(f"  Retrieved {len(memories)} memories")
        for i, mem in enumerate(memories):
            self.logger.debug(f"    [{i+1}] {mem.get('title', 'N/A')}")
    
    def log_memory_extraction(self, is_success: bool, items: list):
        """记录记忆提取"""
        status = "SUCCESS" if is_success else "FAILURE"
        self.logger.info(f"[Task {self.task_id}] Memory Extraction ({status})")
        self.logger.info(f"  Extracted {len(items)} memory items")
        for item in items:
            self.logger.debug(f"    - {item.get('title', 'N/A')}")
    
    def log_result(self, is_success: bool, answer: str, ground_truth: str = ""):
        """记录任务结果"""
        status = "✅ SUCCESS" if is_success else "❌ FAILURE"
        self.logger.info(f"[Task {self.task_id}] Result: {status}")
        self.logger.info(f"  Answer: {answer}")
        if ground_truth:
            self.logger.info(f"  Ground Truth: {ground_truth}")
    
    def get_trajectory(self) -> list:
        """获取完整轨迹"""
        return self.steps

