"""
Envs 模块：环境适配器层，统一不同 Benchmark 的接口
"""

from reasoning_bank.envs.base import BaseEnv, TaskType
from reasoning_bank.envs.single_turn import SingleTurnEnv, SingleTurnEnvRegistry

# 多轮环境（可选导入，依赖可能未安装）
try:
    from reasoning_bank.envs.alfworld_env import AlfWorldEnv
except ImportError:
    AlfWorldEnv = None

try:
    from reasoning_bank.envs.scienceworld_env import ScienceWorldEnv
except ImportError:
    ScienceWorldEnv = None

__all__ = [
    "BaseEnv",
    "TaskType",
    "SingleTurnEnv",
    "SingleTurnEnvRegistry",
    "AlfWorldEnv",
    "ScienceWorldEnv",
]

