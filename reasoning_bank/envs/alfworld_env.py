"""
ALFWorld 环境适配器

封装 ALFWorld 文本游戏环境，支持多轮交互任务
"""

import os
import json
import random
from pathlib import Path
from typing import List, Optional, Iterator, Dict, Any

from reasoning_bank.envs.base import BaseEnv, TaskType, StepResult, TaskInfo
from reasoning_bank.utils.config import get_config
from reasoning_bank.utils.logger import get_logger

logger = get_logger("alfworld_env")

# 任务类型映射
TASK_TYPE_MAP = {
    1: "pick_and_place_simple",
    2: "look_at_obj_in_light",
    3: "pick_clean_then_place_in_recep",
    4: "pick_heat_then_place_in_recep",
    5: "pick_cool_then_place_in_recep",
    6: "pick_two_obj_and_place"
}


class AlfWorldEnv(BaseEnv):
    """ALFWorld 环境适配器"""
    
    def __init__(
        self,
        data_path: Optional[str] = None,
        split: str = "valid_seen",
        task_types: Optional[List[int]] = None,
        max_games: int = -1,
        max_steps: int = 30,
        seed: int = 42,
    ):
        """初始化 ALFWorld 环境
        
        Args:
            data_path: 数据目录路径
            split: 数据集划分 (train/valid_seen/valid_unseen)
            task_types: 任务类型列表 (1-6)
            max_games: 最大游戏数量
            max_steps: 每个任务的最大步数
            seed: 随机种子
        """
        super().__init__(name="alfworld")
        
        self.data_path = data_path or os.getenv("ALFWORLD_DATA") or get_config("datasets.alfworld.data_path")
        self.split = split
        self.task_types = task_types or [1, 2, 3, 4, 5, 6]
        self.max_games = max_games
        self.max_steps = max_steps
        self.seed = seed
        
        # 游戏文件列表
        self.game_files: List[str] = []
        self._current_game_index: int = -1
        self._env = None
        
        # 加载游戏文件
        self._load_game_files()
    
    def _load_game_files(self):
        """加载游戏文件列表"""
        if not self.data_path or not Path(self.data_path).exists():
            logger.warning(f"ALFWorld 数据路径不存在: {self.data_path}")
            return
        
        task_type_names = [TASK_TYPE_MAP[t] for t in self.task_types if t in TASK_TYPE_MAP]
        split_path = Path(self.data_path) / "json_2.1.1" / self.split
        
        for root, dirs, files in os.walk(split_path):
            if 'game.tw-pddl' in files:
                traj_path = os.path.join(root, 'traj_data.json')
                if os.path.exists(traj_path):
                    with open(traj_path, 'r') as f:
                        traj_data = json.load(f)
                    
                    if traj_data['task_type'] in task_type_names:
                        game_file = os.path.join(root, 'game.tw-pddl')
                        
                        with open(game_file, 'r') as f:
                            game_data = json.load(f)
                        
                        if game_data.get('solvable', False):
                            if 'movable' not in root and 'Sliced' not in root:
                                self.game_files.append(game_file)
        
        # 排序并打乱
        self.game_files.sort()
        if self.seed is not None:
            rng = random.Random(self.seed)
            rng.shuffle(self.game_files)
        
        if self.max_games > 0:
            self.game_files = self.game_files[:self.max_games]
        
        logger.info(f"加载 {len(self.game_files)} 个 ALFWorld 游戏")
    
    def _create_environment(self, game_file: str):
        """创建单个游戏环境"""
        try:
            # 延迟导入
            import textworld
            import textworld.gym
            from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos
            
            alfred_demangler = AlfredDemangler(shuffle=False)
            wrappers = [alfred_demangler, AlfredInfos]
            
            request_infos = textworld.EnvInfos(
                won=True,
                admissible_commands=True,
                extras=["gamefile"]
            )
            
            env_id = textworld.gym.register_game(
                game_file,
                request_infos,
                max_episode_steps=self.max_steps,
                wrappers=wrappers
            )
            
            return textworld.gym.make(env_id)
            
        except ImportError as e:
            logger.error(f"ALFWorld 依赖未安装: {e}")
            raise
    
    @property
    def task_type(self) -> TaskType:
        return TaskType.MULTI_TURN
    
    def reset(self, task_id: Optional[str] = None) -> str:
        """重置环境
        
        Args:
            task_id: 游戏文件路径或索引
        """
        self.step_count = 0
        self.trajectory = []
        
        # 关闭旧环境
        if self._env is not None:
            self._env.close()
        
        # 选择游戏
        if task_id:
            if task_id.isdigit():
                idx = int(task_id)
                if 0 <= idx < len(self.game_files):
                    game_file = self.game_files[idx]
                    self._current_game_index = idx
                else:
                    raise ValueError(f"游戏索引超出范围: {idx}")
            else:
                game_file = task_id
                self._current_game_index = -1
        else:
            self._current_game_index += 1
            if self._current_game_index >= len(self.game_files):
                self._current_game_index = 0
            game_file = self.game_files[self._current_game_index]
        
        # 创建环境
        self._env = self._create_environment(game_file)
        obs, info = self._env.reset()
        
        # 提取任务描述
        task_desc = self._extract_task_description(obs)
        
        self.current_task = TaskInfo(
            task_id=str(self._current_game_index),
            task_type=TaskType.MULTI_TURN,
            query=task_desc,
            metadata={
                "game_file": game_file,
                "admissible_commands": info.get("admissible_commands", []),
            },
        )
        
        logger.info(f"开始 ALFWorld 游戏: {Path(game_file).parent.name}")
        
        return obs
    
    def _extract_task_description(self, observation: str) -> str:
        """从观察中提取任务描述"""
        lines = observation.strip().split('\n')
        for line in lines:
            if line.startswith("Your task is to:"):
                return line
        return observation[:200]
    
    def step(self, action: str) -> StepResult:
        """执行动作"""
        if self._env is None:
            raise RuntimeError("环境未初始化，请先调用 reset()")
        
        self.step_count += 1
        
        obs, reward, done, info = self._env.step(action)
        
        is_won = info.get('won', False)
        
        return StepResult(
            observation=obs,
            reward=reward,
            done=done,
            info={
                "won": is_won,
                "admissible_commands": info.get("admissible_commands", []),
                "step": self.step_count,
            },
        )
    
    def get_ground_truth(self) -> str:
        """获取标准答案（多轮任务无固定答案）"""
        return ""
    
    def evaluate(self, prediction: str) -> bool:
        """评估（根据环境的 won 状态）"""
        # 多轮任务的评估在 step 中进行
        return False
    
    def close(self):
        """关闭环境"""
        if self._env is not None:
            self._env.close()
            self._env = None
    
    def __len__(self) -> int:
        return len(self.game_files)
    
    def __iter__(self) -> Iterator[str]:
        for i in range(len(self.game_files)):
            yield str(i)

