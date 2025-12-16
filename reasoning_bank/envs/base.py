"""
环境抽象基类

定义统一的环境接口，支持单轮和多轮任务
"""

from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


class TaskType(Enum):
    """任务类型枚举"""
    SINGLE_TURN = "single_turn"  # 单轮问答
    MULTI_TURN = "multi_turn"    # 多轮交互


@dataclass
class StepResult:
    """环境步进结果"""
    observation: str       # 观察/反馈
    reward: float = 0.0    # 奖励
    done: bool = False     # 是否结束
    info: Dict[str, Any] = field(default_factory=dict)  # 额外信息


@dataclass
class TaskInfo:
    """任务信息"""
    task_id: str           # 任务 ID
    task_type: TaskType    # 任务类型
    query: str             # 问题/任务描述
    ground_truth: str = "" # 标准答案（如果有）
    options: List[str] = field(default_factory=list)  # 选项（MCQ）
    metadata: Dict[str, Any] = field(default_factory=dict)  # 元数据


class BaseEnv(ABC):
    """环境抽象基类"""
    
    def __init__(self, name: str = "base"):
        """初始化环境
        
        Args:
            name: 环境名称
        """
        self.name = name
        self.current_task: Optional[TaskInfo] = None
        self.step_count: int = 0
        self.trajectory: List[Dict[str, Any]] = []
    
    @property
    @abstractmethod
    def task_type(self) -> TaskType:
        """获取任务类型"""
        pass
    
    @abstractmethod
    def reset(self, task_id: Optional[str] = None) -> str:
        """重置环境，开始新任务
        
        Args:
            task_id: 指定任务 ID（可选）
            
        Returns:
            初始观察/问题描述
        """
        pass
    
    @abstractmethod
    def step(self, action: str) -> StepResult:
        """执行动作
        
        Args:
            action: 动作/回答
            
        Returns:
            StepResult 对象
        """
        pass
    
    @abstractmethod
    def get_ground_truth(self) -> str:
        """获取标准答案"""
        pass
    
    @abstractmethod
    def evaluate(self, prediction: str) -> bool:
        """评估预测是否正确
        
        Args:
            prediction: 预测答案
            
        Returns:
            是否正确
        """
        pass
    
    def get_task_info(self) -> Optional[TaskInfo]:
        """获取当前任务信息"""
        return self.current_task
    
    def get_trajectory(self) -> List[Dict[str, Any]]:
        """获取当前轨迹"""
        return self.trajectory
    
    def record_step(
        self,
        observation: str,
        thought: str,
        action: str,
        result: str = "",
    ):
        """记录一步轨迹
        
        Args:
            observation: 观察
            thought: 思考
            action: 动作
            result: 结果
        """
        self.trajectory.append({
            "step": self.step_count,
            "observation": observation,
            "thought": thought,
            "action": action,
            "result": result,
        })
    
    def format_trajectory(self) -> str:
        """格式化轨迹为字符串"""
        lines = []
        for step in self.trajectory:
            lines.append(f"Step {step['step']}:")
            lines.append(f"  Observation: {step['observation'][:200]}...")
            lines.append(f"  Thought: {step['thought']}")
            lines.append(f"  Action: {step['action']}")
            if step.get('result'):
                lines.append(f"  Result: {step['result']}")
            lines.append("")
        return "\n".join(lines)
    
    @abstractmethod
    def __len__(self) -> int:
        """返回任务总数"""
        pass
    
    @abstractmethod
    def __iter__(self):
        """迭代所有任务"""
        pass
    
    def close(self):
        """关闭环境（清理资源）"""
        pass

