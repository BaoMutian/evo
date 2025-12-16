"""
进化循环模块

实现 ReasoningBank 的核心闭环：检索 -> 推理 -> 执行 -> 评估 -> 提取 -> 存储
"""

import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Iterator
from dataclasses import dataclass, field, asdict
from tqdm import tqdm

from reasoning_bank.core.llm_service import LLMService, get_llm_service
from reasoning_bank.core.memory import MemoryBank
from reasoning_bank.core.agent import ReActAgent, AgentConfig, AgentResult
from reasoning_bank.core.extractor import MemoryExtractor
from reasoning_bank.envs.base import BaseEnv, TaskType
from reasoning_bank.utils.config import get_config
from reasoning_bank.utils.logger import get_logger, setup_logger

logger = get_logger("evolution")


@dataclass
class EpisodeResult:
    """单个任务的执行结果"""
    task_id: str
    query: str
    answer: str
    ground_truth: str
    is_success: bool
    steps: int
    memories_retrieved: int
    memories_extracted: int
    trajectory: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class EvolutionStats:
    """进化循环统计"""
    total_tasks: int = 0
    success_count: int = 0
    failure_count: int = 0
    total_steps: int = 0
    memories_added: int = 0

    @property
    def success_rate(self) -> float:
        if self.total_tasks == 0:
            return 0.0
        return self.success_count / self.total_tasks

    @property
    def avg_steps(self) -> float:
        if self.total_tasks == 0:
            return 0.0
        return self.total_steps / self.total_tasks

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_tasks": self.total_tasks,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "success_rate": self.success_rate,
            "total_steps": self.total_steps,
            "avg_steps": self.avg_steps,
            "memories_added": self.memories_added,
        }


class EvolutionLoop:
    """进化循环主类"""

    def __init__(
        self,
        env: BaseEnv,
        memory_bank: Optional[MemoryBank] = None,
        llm_service: Optional[LLMService] = None,
        agent_config: Optional[AgentConfig] = None,
        extract_memories: bool = True,
        save_results: bool = True,
        results_dir: str = "./results",
    ):
        """初始化进化循环

        Args:
            env: 环境实例
            memory_bank: 记忆库实例（可选，不提供则不使用记忆）
            llm_service: LLM 服务实例
            agent_config: 智能体配置
            extract_memories: 是否提取记忆
            save_results: 是否保存结果
            results_dir: 结果保存目录
        """
        self.env = env
        self.memory_bank = memory_bank
        self.llm = llm_service or get_llm_service()
        self.agent_config = agent_config or AgentConfig()
        self.extract_memories = extract_memories
        self.save_results = save_results
        self.results_dir = Path(results_dir)

        # 初始化组件
        self.agent = ReActAgent(
            llm_service=self.llm,
            memory_bank=self.memory_bank,
            config=self.agent_config,
        )

        self.extractor = MemoryExtractor(
            llm_service=self.llm,
            use_llm_judge=False,  # 使用环境评判
        )

        # 统计
        self.stats = EvolutionStats()
        self.results: List[EpisodeResult] = []

        # 确保结果目录存在
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def run_episode(self, task_id: Optional[str] = None) -> EpisodeResult:
        """运行单个任务

        Args:
            task_id: 任务 ID（可选）

        Returns:
            EpisodeResult
        """
        # 执行任务
        agent_result = self.agent.run(self.env, task_id)

        # 获取标准答案
        ground_truth = self.env.get_ground_truth()

        # 提取记忆
        memories_extracted = 0
        if self.extract_memories and self.memory_bank is not None:
            items = self.extractor.extract(agent_result, ground_truth)
            if items:
                self.memory_bank.add(
                    query=agent_result.query,
                    items=items,
                    is_success=agent_result.is_success,
                    trajectory_id=agent_result.task_id,
                )
                memories_extracted = len(items)
                self.stats.memories_added += memories_extracted

        # 构建结果
        episode_result = EpisodeResult(
            task_id=agent_result.task_id,
            query=agent_result.query,
            answer=agent_result.answer,
            ground_truth=ground_truth,
            is_success=agent_result.is_success,
            steps=agent_result.steps,
            memories_retrieved=len(agent_result.memories_used),
            memories_extracted=memories_extracted,
            trajectory=agent_result.trajectory,
        )

        # 更新统计
        self.stats.total_tasks += 1
        self.stats.total_steps += agent_result.steps
        if agent_result.is_success:
            self.stats.success_count += 1
        else:
            self.stats.failure_count += 1

        self.results.append(episode_result)

        return episode_result

    def run(
        self,
        num_tasks: Optional[int] = None,
        task_ids: Optional[List[str]] = None,
        show_progress: bool = True,
    ) -> EvolutionStats:
        """运行进化循环

        Args:
            num_tasks: 运行的任务数量（默认全部）
            task_ids: 指定的任务 ID 列表
            show_progress: 是否显示进度条

        Returns:
            EvolutionStats
        """
        # 确定要运行的任务
        if task_ids:
            tasks_to_run = task_ids
        else:
            tasks_to_run = list(self.env)
            if num_tasks and num_tasks < len(tasks_to_run):
                tasks_to_run = tasks_to_run[:num_tasks]

        total = len(tasks_to_run)
        logger.info(f"开始进化循环，共 {total} 个任务")

        # 进度条
        iterator = tqdm(
            tasks_to_run, desc="Evolution") if show_progress else tasks_to_run

        for task_id in iterator:
            try:
                result = self.run_episode(task_id)

                if show_progress:
                    iterator.set_postfix({
                        "success_rate": f"{self.stats.success_rate:.2%}",
                        "memories": self.stats.memories_added,
                    })

            except Exception as e:
                logger.error(f"任务 {task_id} 执行失败: {e}")
                self.stats.total_tasks += 1
                self.stats.failure_count += 1

        # 保存结果
        if self.save_results:
            self._save_results()

        # 打印统计
        self._print_stats()

        return self.stats

    def _save_results(self):
        """保存结果到文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = self.results_dir / \
            f"evolution_{self.env.name}_{timestamp}.json"

        data = {
            "env_name": self.env.name,
            "timestamp": timestamp,
            "stats": self.stats.to_dict(),
            "memory_bank_stats": self.memory_bank.get_stats() if self.memory_bank else {},
            "results": [
                {
                    "task_id": r.task_id,
                    "query": r.query[:200] + "..." if len(r.query) > 200 else r.query,
                    "answer": r.answer,
                    "ground_truth": r.ground_truth,
                    "is_success": r.is_success,
                    "steps": r.steps,
                    "memories_retrieved": r.memories_retrieved,
                    "memories_extracted": r.memories_extracted,
                }
                for r in self.results
            ],
        }

        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"结果已保存到: {result_file}")

    def _print_stats(self):
        """打印统计信息"""
        logger.info("=" * 50)
        logger.info("进化循环统计")
        logger.info("=" * 50)
        logger.info(f"总任务数: {self.stats.total_tasks}")
        logger.info(f"成功数: {self.stats.success_count}")
        logger.info(f"失败数: {self.stats.failure_count}")
        logger.info(f"成功率: {self.stats.success_rate:.2%}")
        logger.info(f"平均步数: {self.stats.avg_steps:.2f}")
        logger.info(f"新增记忆: {self.stats.memories_added}")

        if self.memory_bank:
            logger.info(f"记忆库大小: {len(self.memory_bank)}")

        logger.info("=" * 50)

    def get_results(self) -> List[EpisodeResult]:
        """获取所有结果"""
        return self.results

    def reset_stats(self):
        """重置统计"""
        self.stats = EvolutionStats()
        self.results = []


def run_benchmark(
    dataset_name: str,
    num_tasks: Optional[int] = None,
    use_memory: bool = True,
    extract_memories: bool = True,
    model: Optional[str] = None,
    **kwargs,
) -> EvolutionStats:
    """便捷函数：运行基准测试

    Args:
        dataset_name: 数据集名称（如 "math500", "gpqa"）
        num_tasks: 任务数量
        use_memory: 是否使用记忆库
        extract_memories: 是否提取记忆
        model: LLM 模型名称
        **kwargs: 其他参数

    Returns:
        EvolutionStats
    """
    from reasoning_bank.envs.single_turn import SingleTurnEnvRegistry

    # 创建环境
    env = SingleTurnEnvRegistry.create(dataset_name, **kwargs)

    # 创建记忆库
    memory_bank = None
    if use_memory:
        memory_bank = MemoryBank(bank_name=dataset_name)

    # 创建 LLM 服务
    llm_service = None
    if model:
        llm_service = LLMService(model=model)

    # 运行
    loop = EvolutionLoop(
        env=env,
        memory_bank=memory_bank,
        llm_service=llm_service,
        extract_memories=extract_memories,
    )

    return loop.run(num_tasks=num_tasks)
