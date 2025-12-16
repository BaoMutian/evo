#!/usr/bin/env python
"""
ReasoningBank 主入口

用法:
    # 运行单轮QA测试
    python -m reasoning_bank.main --dataset math500 --num-tasks 10
    
    # 使用记忆库运行
    python -m reasoning_bank.main --dataset gpqa --use-memory --num-tasks 50
    
    # 指定模型
    python -m reasoning_bank.main --dataset aime24 --model "qwen/qwen3-32b" --num-tasks 5
    
    # 运行 MaTTS 并行扩展
    python -m reasoning_bank.main --dataset math500 --matts parallel --num-tasks 5
"""

from reasoning_bank.workflows.evolution import EvolutionLoop, run_benchmark
from reasoning_bank.envs.single_turn import SingleTurnEnvRegistry
from reasoning_bank.core.memory import MemoryBank
from reasoning_bank.core.llm_service import LLMService
from reasoning_bank.utils.logger import setup_logger, get_logger
from reasoning_bank.utils.config import load_config, get_config
import argparse
import sys
from pathlib import Path

# 确保项目根目录在路径中
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="ReasoningBank - 自我进化智能体框架",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # 数据集参数
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default="math500",
        choices=SingleTurnEnvRegistry.list_datasets(),
        help="数据集名称",
    )

    parser.add_argument(
        "--num-tasks", "-n",
        type=int,
        default=None,
        help="运行的任务数量（默认全部）",
    )

    # 记忆参数
    parser.add_argument(
        "--use-memory",
        action="store_true",
        help="是否使用记忆库",
    )

    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="不提取新记忆",
    )

    parser.add_argument(
        "--clear-memory",
        action="store_true",
        help="清空记忆库后运行",
    )

    # LLM 参数
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="LLM 模型名称（覆盖配置文件）",
    )

    parser.add_argument(
        "--temperature", "-t",
        type=float,
        default=0.3,
        help="生成温度",
    )

    # MaTTS 参数
    parser.add_argument(
        "--matts",
        type=str,
        choices=["none", "parallel", "sequential", "combined"],
        default="none",
        help="MaTTS 模式",
    )

    parser.add_argument(
        "--parallel-n",
        type=int,
        default=5,
        help="MaTTS 并行轨迹数量",
    )

    # 其他参数
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="配置文件路径",
    )

    parser.add_argument(
        "--results-dir",
        type=str,
        default="./results",
        help="结果保存目录",
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="详细输出",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子",
    )

    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="是否打乱数据集顺序",
    )

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 加载配置
    if args.config:
        load_config(args.config)
    else:
        load_config()

    # 设置日志
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logger(
        name="reasoning_bank",
        level=log_level,
        log_dir=get_config("logging.log_dir", "./logs"),
        log_to_file=True,
        log_to_console=True,
    )

    logger.info("=" * 60)
    logger.info("ReasoningBank 自我进化智能体")
    logger.info("=" * 60)
    logger.info(f"数据集: {args.dataset}")
    logger.info(f"任务数量: {args.num_tasks or '全部'}")
    logger.info(f"使用记忆: {args.use_memory}")
    logger.info(f"MaTTS 模式: {args.matts}")
    logger.info(f"模型: {args.model or get_config('llm.default_model')}")
    logger.info("=" * 60)

    # 创建环境
    env = SingleTurnEnvRegistry.create(
        args.dataset,
        shuffle=args.shuffle,
        seed=args.seed,
    )
    logger.info(f"加载数据集: {len(env)} 条数据")

    # 创建记忆库
    memory_bank = None
    if args.use_memory:
        memory_bank = MemoryBank(bank_name=args.dataset)
        if args.clear_memory:
            logger.info("清空记忆库")
            memory_bank.clear()
        logger.info(f"记忆库: {memory_bank}")

    # 创建 LLM 服务
    llm_service = LLMService(
        model=args.model,
        temperature=args.temperature,
    )

    # 根据 MaTTS 模式选择运行方式
    if args.matts == "none":
        # 普通进化循环
        from reasoning_bank.core.agent import AgentConfig

        agent_config = AgentConfig(
            temperature=args.temperature,
            verbose=args.verbose,
        )

        loop = EvolutionLoop(
            env=env,
            memory_bank=memory_bank,
            llm_service=llm_service,
            agent_config=agent_config,
            extract_memories=not args.no_extract,
            save_results=True,
            results_dir=args.results_dir,
        )

        stats = loop.run(num_tasks=args.num_tasks)

    else:
        # MaTTS 模式
        from reasoning_bank.workflows.matts import MaTTSRunner, MaTTSConfig
        from tqdm import tqdm

        matts_config = MaTTSConfig(
            parallel_n=args.parallel_n,
            parallel_temperature=0.7,
        )

        runner = MaTTSRunner(
            env=env,
            memory_bank=memory_bank,
            llm_service=llm_service,
            config=matts_config,
        )

        # 确定任务列表
        task_ids = list(env)
        if args.num_tasks and args.num_tasks < len(task_ids):
            task_ids = task_ids[:args.num_tasks]

        success_count = 0
        total_memories = 0

        for task_id in tqdm(task_ids, desc=f"MaTTS {args.matts}"):
            try:
                if args.matts == "parallel":
                    result, memories = runner.run_parallel(task_id)
                elif args.matts == "sequential":
                    result, memories = runner.run_sequential(task_id)
                else:  # combined
                    result, memories = runner.run_combined(task_id)

                if result.is_success:
                    success_count += 1

                # 添加记忆到记忆库
                if memory_bank and memories:
                    memory_bank.add(
                        query=result.query,
                        items=memories,
                        is_success=result.is_success,
                        trajectory_id=result.task_id,
                    )
                    total_memories += len(memories)

            except Exception as e:
                logger.error(f"任务 {task_id} 失败: {e}")

        # 打印统计
        logger.info("=" * 50)
        logger.info(f"MaTTS {args.matts} 完成")
        logger.info(
            f"成功率: {success_count}/{len(task_ids)} ({success_count/len(task_ids):.2%})")
        logger.info(f"新增记忆: {total_memories}")
        logger.info("=" * 50)

    logger.info("运行完成!")


if __name__ == "__main__":
    main()
