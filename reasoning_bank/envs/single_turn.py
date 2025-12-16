"""
单轮问答环境适配器

支持 AIME, MATH, GPQA, MMLU-Pro 等单轮 QA 数据集
"""

import json
import random
from pathlib import Path
from typing import List, Optional, Iterator, Dict, Any

from reasoning_bank.envs.base import BaseEnv, TaskType, StepResult, TaskInfo
from reasoning_bank.utils.answer_parser import (
    extract_final_answer,
    extract_choice_answer,
    compare_answers,
    extract_boxed_answer,
)
from reasoning_bank.utils.config import get_config
from reasoning_bank.utils.logger import get_logger

logger = get_logger("single_turn_env")


class SingleTurnEnv(BaseEnv):
    """单轮问答环境"""
    
    def __init__(
        self,
        dataset_path: Optional[str] = None,
        dataset_name: Optional[str] = None,
        shuffle: bool = False,
        seed: int = 42,
        max_samples: Optional[int] = None,
    ):
        """初始化单轮环境
        
        Args:
            dataset_path: 数据集文件路径（JSONL 格式）
            dataset_name: 数据集名称（用于从配置中查找路径）
            shuffle: 是否打乱顺序
            seed: 随机种子
            max_samples: 最大样本数（用于调试）
        """
        super().__init__(name=dataset_name or "single_turn")
        
        self.dataset_path = self._resolve_path(dataset_path, dataset_name)
        self.shuffle = shuffle
        self.seed = seed
        self.max_samples = max_samples
        
        # 加载数据
        self.tasks: List[Dict[str, Any]] = []
        self._load_data()
        
        # 当前任务索引
        self._current_index: int = -1
        self._is_done: bool = False
    
    def _resolve_path(
        self,
        dataset_path: Optional[str],
        dataset_name: Optional[str],
    ) -> Path:
        """解析数据集路径"""
        if dataset_path:
            return Path(dataset_path)
        
        if dataset_name:
            base_path = get_config("datasets.single_turn.base_path", "./bench/single_turn_bench")
            return Path(base_path) / dataset_name
        
        raise ValueError("必须指定 dataset_path 或 dataset_name")
    
    def _load_data(self):
        """加载数据集"""
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"数据集文件不存在: {self.dataset_path}")
        
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.tasks.append(json.loads(line))
        
        # 打乱顺序
        if self.shuffle:
            random.seed(self.seed)
            random.shuffle(self.tasks)
        
        # 限制样本数
        if self.max_samples and self.max_samples < len(self.tasks):
            self.tasks = self.tasks[:self.max_samples]
        
        logger.info(f"加载数据集 {self.dataset_path.name}，共 {len(self.tasks)} 条数据")
    
    @property
    def task_type(self) -> TaskType:
        return TaskType.SINGLE_TURN
    
    @property
    def is_mcq(self) -> bool:
        """当前任务是否为选择题"""
        if self.current_task:
            return bool(self.current_task.options)
        return False
    
    def reset(self, task_id: Optional[str] = None) -> str:
        """重置环境，开始新任务
        
        Args:
            task_id: 指定任务 ID（可选）
            
        Returns:
            问题文本
        """
        # 重置状态
        self.step_count = 0
        self.trajectory = []
        self._is_done = False
        
        # 选择任务
        if task_id:
            # 查找指定 ID 的任务
            task_data = None
            for i, t in enumerate(self.tasks):
                if t["id"] == task_id:
                    task_data = t
                    self._current_index = i
                    break
            if task_data is None:
                raise ValueError(f"找不到任务 ID: {task_id}")
        else:
            # 顺序选择下一个任务
            self._current_index += 1
            if self._current_index >= len(self.tasks):
                self._current_index = 0  # 循环
            task_data = self.tasks[self._current_index]
        
        # 构建任务信息
        self.current_task = TaskInfo(
            task_id=task_data["id"],
            task_type=TaskType.SINGLE_TURN,
            query=task_data["question"],
            ground_truth=task_data["answer"],
            options=task_data.get("options", []),
            metadata=task_data.get("metadata", {}),
        )
        
        # 构建观察（问题文本）
        observation = self._format_question(task_data)
        
        logger.info(f"开始任务: {self.current_task.task_id}")
        
        return observation
    
    def _format_question(self, task_data: Dict[str, Any]) -> str:
        """格式化问题文本"""
        lines = [task_data["question"]]
        
        # 添加选项（如果有）
        options = task_data.get("options", [])
        if options:
            lines.append("\nOptions:")
            for i, opt in enumerate(options):
                letter = chr(ord('A') + i)
                lines.append(f"  {letter}. {opt}")
        
        return "\n".join(lines)
    
    def step(self, action: str) -> StepResult:
        """执行动作（提交答案）
        
        对于单轮任务，step 通常只调用一次
        
        Args:
            action: 回答/答案
            
        Returns:
            StepResult
        """
        self.step_count += 1
        
        # 单轮任务，提交答案即结束
        self._is_done = True
        
        # 评估答案
        is_correct = self.evaluate(action)
        
        # 构建结果
        result = StepResult(
            observation="Task completed.",
            reward=1.0 if is_correct else 0.0,
            done=True,
            info={
                "is_correct": is_correct,
                "prediction": action,
                "ground_truth": self.get_ground_truth(),
            },
        )
        
        logger.info(f"任务 {self.current_task.task_id} 完成, 正确={is_correct}")
        
        return result
    
    def get_ground_truth(self) -> str:
        """获取标准答案"""
        if self.current_task:
            return self.current_task.ground_truth
        return ""
    
    def evaluate(self, prediction: str) -> bool:
        """评估预测是否正确
        
        Args:
            prediction: 预测答案
            
        Returns:
            是否正确
        """
        if not self.current_task:
            return False
        
        ground_truth = self.current_task.ground_truth
        
        # 选择题
        if self.is_mcq:
            # 从预测中提取选项字母
            pred_letter = extract_choice_answer(prediction)
            gt_letter = self.current_task.metadata.get("answer_letter", "")
            
            if pred_letter and gt_letter:
                return pred_letter.upper() == gt_letter.upper()
            
            # 回退到文本匹配
            return compare_answers(prediction, ground_truth, is_mcq=True)
        
        # 数学题
        else:
            # 尝试提取答案
            extracted = extract_final_answer(prediction)
            if extracted:
                prediction = extracted
            
            return compare_answers(prediction, ground_truth, is_mcq=False)
    
    def get_task_by_index(self, index: int) -> Optional[Dict[str, Any]]:
        """按索引获取任务数据"""
        if 0 <= index < len(self.tasks):
            return self.tasks[index]
        return None
    
    def get_task_by_id(self, task_id: str) -> Optional[Dict[str, Any]]:
        """按 ID 获取任务数据"""
        for task in self.tasks:
            if task["id"] == task_id:
                return task
        return None
    
    def __len__(self) -> int:
        return len(self.tasks)
    
    def __iter__(self) -> Iterator[str]:
        """迭代所有任务，返回 task_id"""
        for task in self.tasks:
            yield task["id"]
    
    def iter_tasks(self) -> Iterator[TaskInfo]:
        """迭代所有任务，返回 TaskInfo"""
        for task in self.tasks:
            yield TaskInfo(
                task_id=task["id"],
                task_type=TaskType.SINGLE_TURN,
                query=task["question"],
                ground_truth=task["answer"],
                options=task.get("options", []),
                metadata=task.get("metadata", {}),
            )


class SingleTurnEnvRegistry:
    """单轮环境注册表，方便创建不同数据集的环境"""
    
    # 预定义的数据集配置
    DATASETS = {
        "aime24": "AIME24-30.jsonl",
        "aime25": "AIME25-30.jsonl",
        "math500": "MATH-500.jsonl",
        "gpqa": "GPQA-Diamond-198.jsonl",
        "mmlu_economics": "MMLU-Pro-economics-844.jsonl",
        "mmlu_engineering": "MMLU-Pro-engineering-969.jsonl",
        "mmlu_philosophy": "MMLU-Pro-philosophy-499.jsonl",
    }
    
    @classmethod
    def create(
        cls,
        dataset_key: str,
        base_path: Optional[str] = None,
        **kwargs,
    ) -> SingleTurnEnv:
        """创建环境实例
        
        Args:
            dataset_key: 数据集键名
            base_path: 基础路径
            **kwargs: 其他参数传递给 SingleTurnEnv
            
        Returns:
            SingleTurnEnv 实例
        """
        if dataset_key not in cls.DATASETS:
            raise ValueError(f"未知数据集: {dataset_key}，可用: {list(cls.DATASETS.keys())}")
        
        base_path = base_path or get_config("datasets.single_turn.base_path", "./bench/single_turn_bench")
        dataset_path = Path(base_path) / cls.DATASETS[dataset_key]
        
        return SingleTurnEnv(
            dataset_path=str(dataset_path),
            dataset_name=dataset_key,
            **kwargs,
        )
    
    @classmethod
    def list_datasets(cls) -> List[str]:
        """列出所有可用数据集"""
        return list(cls.DATASETS.keys())

