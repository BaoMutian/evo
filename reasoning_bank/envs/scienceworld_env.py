"""
ScienceWorld 环境适配器

封装 ScienceWorld 科学实验环境，支持多轮交互任务
"""

import random
from typing import List, Optional, Iterator, Dict, Any

from reasoning_bank.envs.base import BaseEnv, TaskType, StepResult, TaskInfo
from reasoning_bank.utils.config import get_config
from reasoning_bank.utils.logger import get_logger

logger = get_logger("scienceworld_env")

# 任务信息映射
TASK_INFO = {
    "1-1": {"name": "boil", "topic": "Matter", "desc": "沸腾"},
    "1-2": {"name": "melt", "topic": "Matter", "desc": "融化"},
    "1-3": {"name": "freeze", "topic": "Matter", "desc": "冷冻"},
    "1-4": {"name": "change-the-state-of-matter-of", "topic": "Matter", "desc": "物态变化"},
    "2-1": {"name": "use-thermometer", "topic": "Measurement", "desc": "使用温度计"},
    "2-2": {"name": "measure-melting-point-known-substance", "topic": "Measurement", "desc": "测量已知熔点"},
    "2-3": {"name": "measure-melting-point-unknown-substance", "topic": "Measurement", "desc": "测量未知熔点"},
    "3-1": {"name": "power-component", "topic": "Electricity", "desc": "创建电路"},
    "3-2": {"name": "power-component-renewable-vs-nonrenewable-energy", "topic": "Electricity", "desc": "可再生能源"},
    "3-3": {"name": "test-conductivity", "topic": "Electricity", "desc": "测试导电性"},
    "3-4": {"name": "test-conductivity-of-unknown-substances", "topic": "Electricity", "desc": "未知导电性"},
    "4-1": {"name": "find-living-thing", "topic": "Classification", "desc": "找生物"},
    "4-2": {"name": "find-non-living-thing", "topic": "Classification", "desc": "找非生物"},
    "4-3": {"name": "find-plant", "topic": "Classification", "desc": "找植物"},
    "4-4": {"name": "find-animal", "topic": "Classification", "desc": "找动物"},
    "5-1": {"name": "grow-plant", "topic": "Biology", "desc": "种植物"},
    "5-2": {"name": "grow-fruit", "topic": "Biology", "desc": "种果实"},
    "6-1": {"name": "chemistry-mix", "topic": "Chemistry", "desc": "化学混合"},
    "6-2": {"name": "chemistry-mix-paint-secondary-color", "topic": "Chemistry", "desc": "混合二次色"},
    "6-3": {"name": "chemistry-mix-paint-tertiary-color", "topic": "Chemistry", "desc": "混合三次色"},
    "7-1": {"name": "lifespan-longest-lived", "topic": "Biology", "desc": "最长寿命"},
    "7-2": {"name": "lifespan-shortest-lived", "topic": "Biology", "desc": "最短寿命"},
    "7-3": {"name": "lifespan-longest-lived-then-shortest-lived", "topic": "Biology", "desc": "寿命排序"},
    "8-1": {"name": "identify-life-stages-1", "topic": "Biology", "desc": "植物生命周期"},
    "8-2": {"name": "identify-life-stages-2", "topic": "Biology", "desc": "动物生命周期"},
    "9-1": {"name": "inclined-plane-determine-angle", "topic": "Forces", "desc": "斜面角度"},
    "9-2": {"name": "inclined-plane-friction-named-surfaces", "topic": "Forces", "desc": "已知摩擦力"},
    "9-3": {"name": "inclined-plane-friction-unnamed-surfaces", "topic": "Forces", "desc": "未知摩擦力"},
    "10-1": {"name": "mendelian-genetics-known-plant", "topic": "Biology", "desc": "已知遗传学"},
    "10-2": {"name": "mendelian-genetics-unknown-plant", "topic": "Biology", "desc": "未知遗传学"},
}


class ScienceWorldEnv(BaseEnv):
    """ScienceWorld 环境适配器"""
    
    def __init__(
        self,
        task_ids: Optional[List[str]] = None,
        simplifications: str = "easy",
        max_steps: int = 50,
        episodes_per_task: int = 1,
        split: str = "dev",
        seed: int = 42,
    ):
        """初始化 ScienceWorld 环境
        
        Args:
            task_ids: 任务 ID 列表（如 ["1-1", "1-2", "4-1"]），默认全部
            simplifications: 简化设置（"easy" 或自定义）
            max_steps: 每个任务的最大步数
            episodes_per_task: 每个任务的变体数量
            split: 数据集划分 (train/dev/test)
            seed: 随机种子
        """
        super().__init__(name="scienceworld")
        
        self.task_ids = task_ids or list(TASK_INFO.keys())
        self.simplifications = simplifications
        self.max_steps = max_steps
        self.episodes_per_task = episodes_per_task
        self.split = split
        self.seed = seed
        
        # 环境实例
        self._env = None
        self._current_task_id: str = ""
        self._current_variation: int = 0
        self._task_queue: List[tuple] = []  # [(task_id, variation), ...]
        
        # 初始化
        self._init_environment()
        self._build_task_queue()
    
    def _init_environment(self):
        """初始化 ScienceWorld 环境"""
        try:
            from scienceworld import ScienceWorldEnv as SWEnv
            self._env = SWEnv("", envStepLimit=self.max_steps + 10)
            logger.info("ScienceWorld 环境初始化成功")
        except ImportError as e:
            logger.error(f"ScienceWorld 依赖未安装: {e}")
            self._env = None
    
    def _build_task_queue(self):
        """构建任务队列"""
        if self._env is None:
            return
        
        self._task_queue = []
        
        if self.seed is not None:
            random.seed(self.seed)
        
        for task_id in self.task_ids:
            if task_id not in TASK_INFO:
                logger.warning(f"未知的任务 ID: {task_id}")
                continue
            
            task_name = TASK_INFO[task_id]["name"]
            
            try:
                # 加载任务获取变体信息
                self._env.load(task_name, 0, self.simplifications)
                
                # 获取变体列表
                if self.split == "train":
                    variations = self._env.get_variations_train()
                elif self.split == "dev":
                    variations = self._env.get_variations_dev()
                else:
                    variations = self._env.get_variations_test()
                
                if not variations:
                    continue
                
                # 随机选择变体
                selected = random.sample(
                    variations,
                    min(self.episodes_per_task, len(variations))
                )
                
                for var in selected:
                    self._task_queue.append((task_id, var))
                    
            except Exception as e:
                logger.warning(f"加载任务 {task_id} 失败: {e}")
        
        logger.info(f"构建任务队列，共 {len(self._task_queue)} 个任务实例")
    
    @property
    def task_type(self) -> TaskType:
        return TaskType.MULTI_TURN
    
    def reset(self, task_id: Optional[str] = None) -> str:
        """重置环境
        
        Args:
            task_id: 任务 ID（格式："task_id:variation" 或索引）
        """
        if self._env is None:
            raise RuntimeError("ScienceWorld 环境未初始化")
        
        self.step_count = 0
        self.trajectory = []
        
        # 选择任务
        if task_id:
            if ":" in task_id:
                # 格式: "1-1:5"
                tid, var = task_id.split(":")
                self._current_task_id = tid
                self._current_variation = int(var)
            elif task_id.isdigit():
                # 队列索引
                idx = int(task_id)
                if 0 <= idx < len(self._task_queue):
                    self._current_task_id, self._current_variation = self._task_queue[idx]
                else:
                    raise ValueError(f"任务索引超出范围: {idx}")
            else:
                # 直接指定 task_id，使用第一个变体
                self._current_task_id = task_id
                self._current_variation = 0
        else:
            # 顺序选择
            if not self._task_queue:
                raise RuntimeError("任务队列为空")
            
            # 这里简化处理，每次从队列头部取
            self._current_task_id, self._current_variation = self._task_queue[0]
            self._task_queue = self._task_queue[1:] + [self._task_queue[0]]  # 循环
        
        # 加载任务
        task_name = TASK_INFO[self._current_task_id]["name"]
        self._env.load(task_name, self._current_variation, self.simplifications)
        
        # 重置
        obs, info = self._env.reset()
        
        # 获取任务描述
        task_desc = self._env.get_task_description()
        
        self.current_task = TaskInfo(
            task_id=f"{self._current_task_id}:{self._current_variation}",
            task_type=TaskType.MULTI_TURN,
            query=task_desc,
            metadata={
                "task_name": task_name,
                "topic": TASK_INFO[self._current_task_id]["topic"],
                "desc": TASK_INFO[self._current_task_id]["desc"],
                "variation": self._current_variation,
            },
        )
        
        logger.info(f"开始 ScienceWorld 任务: {self._current_task_id} ({TASK_INFO[self._current_task_id]['desc']})")
        
        return f"Task: {task_desc}\n\nObservation:\n{obs}"
    
    def step(self, action: str) -> StepResult:
        """执行动作"""
        if self._env is None:
            raise RuntimeError("环境未初始化")
        
        self.step_count += 1
        
        obs, reward, done, info = self._env.step(action)
        
        score = info.get('score', 0)
        is_success = score >= 100
        
        return StepResult(
            observation=obs,
            reward=reward,
            done=done or is_success,
            info={
                "score": score,
                "is_correct": is_success,
                "won": is_success,
                "valid_actions": info.get("valid", []),
                "step": self.step_count,
            },
        )
    
    def get_ground_truth(self) -> str:
        """获取标准答案（多轮任务无固定答案）"""
        return ""
    
    def evaluate(self, prediction: str) -> bool:
        """评估"""
        return False
    
    def get_valid_actions(self) -> List[str]:
        """获取当前有效动作列表"""
        if self._env:
            return self._env.get_valid_action_object_combinations()
        return []
    
    def close(self):
        """关闭环境"""
        if self._env is not None:
            self._env.close()
            self._env = None
    
    def __len__(self) -> int:
        return len(self._task_queue)
    
    def __iter__(self) -> Iterator[str]:
        for i in range(len(self._task_queue)):
            yield str(i)
    
    @classmethod
    def list_tasks(cls) -> Dict[str, Dict]:
        """列出所有可用任务"""
        return TASK_INFO.copy()

