#!/usr/bin/env python
"""
ReasoningBank 主入口

支持单轮 QA（MATH, GPQA, MMLU-Pro）和多轮交互（ALFWorld, ScienceWorld）

用法:
    # 单轮 QA 任务
    python -m reasoning_bank.main --dataset math500 --num-tasks 10
    python -m reasoning_bank.main --dataset gpqa --use-memory --num-tasks 50
    
    # 多轮交互任务
    python -m reasoning_bank.main --env alfworld --num-tasks 5
    python -m reasoning_bank.main --env scienceworld --num-tasks 10
    
    # MaTTS 扩展
    python -m reasoning_bank.main --dataset math500 --matts parallel --num-tasks 5
"""

from reasoning_bank.utils.config import load_config, get_config
from reasoning_bank.utils.logger import setup_logger, get_logger
from reasoning_bank.core.llm_service import LLMService
from reasoning_bank.core.memory import MemoryBank
from reasoning_bank.envs.single_turn import SingleTurnEnvRegistry
from reasoning_bank.workflows.evolution import EvolutionLoop
import argparse
import sys
from pathlib import Path

# 确保项目根目录在路径中
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


# 支持的环境类型
SINGLE_TURN_DATASETS = SingleTurnEnvRegistry.list_datasets()
MULTI_TURN_ENVS = ["alfworld", "scienceworld"]


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="ReasoningBank - 自我进化智能体框架",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 单轮 QA
  python -m reasoning_bank.main --dataset math500 --num-tasks 10
  python -m reasoning_bank.main --dataset gpqa --use-memory -n 50
  
  # 多轮交互
  python -m reasoning_bank.main --env alfworld --num-tasks 5
  python -m reasoning_bank.main --env scienceworld --task-ids "1-1,4-1" -n 10
  
  # MaTTS 扩展
  python -m reasoning_bank.main --dataset math500 --matts parallel -n 5
        """
    )

    # ============ 环境选择（互斥） ============
    env_group = parser.add_mutually_exclusive_group()

    env_group.add_argument(
        "--dataset", "-d",
        type=str,
        choices=SINGLE_TURN_DATASETS,
        help=f"单轮 QA 数据集: {', '.join(SINGLE_TURN_DATASETS)}",
    )

    env_group.add_argument(
        "--env", "-e",
        type=str,
        choices=MULTI_TURN_ENVS,
        help=f"多轮交互环境: {', '.join(MULTI_TURN_ENVS)}",
    )

    # ============ 通用参数 ============
    parser.add_argument(
        "--num-tasks", "-n",
        type=int,
        default=None,
        help="运行的任务数量（默认全部）",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=30,
        help="多轮任务每个 episode 的最大步数（默认 30）",
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
        help="MaTTS 模式（仅单轮任务支持）",
    )

    parser.add_argument(
        "--parallel-n",
        type=int,
        default=5,
        help="MaTTS 并行轨迹数量",
    )

    # ============ 多轮环境专用参数 ============
    # ALFWorld
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="ALFWorld 数据路径",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="valid_seen",
        choices=["train", "valid_seen", "valid_unseen", "dev", "test"],
        help="数据集划分",
    )

    # ScienceWorld
    parser.add_argument(
        "--task-ids",
        type=str,
        default=None,
        help="ScienceWorld 任务 ID 列表（逗号分隔，如 '1-1,1-2,4-1'）",
    )

    parser.add_argument(
        "--simplifications",
        type=str,
        default="easy",
        help="ScienceWorld 简化设置",
    )

    # ============ 其他参数 ============
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

    args = parser.parse_args()

    # 默认使用 math500
    if not args.dataset and not args.env:
        args.dataset = "math500"

    return args


def run_single_turn(args, logger):
    """运行单轮 QA 任务"""
    from reasoning_bank.core.agent import AgentConfig

    # 创建环境
    env = SingleTurnEnvRegistry.create(
        args.dataset,
        shuffle=args.shuffle,
        seed=args.seed,
    )
    logger.info(f"加载数据集 {args.dataset}: {len(env)} 条数据")

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


def run_multi_turn(args, logger):
    """运行多轮交互任务"""
    import os
    from tqdm import tqdm
    from reasoning_bank.core.agent import ReActAgent, AgentConfig, AgentResult
    from reasoning_bank.core.extractor import MemoryExtractor
    from reasoning_bank.prompts.registry import PromptRegistry

    env_name = args.env

    # 创建环境
    if env_name == "alfworld":
        try:
            from reasoning_bank.envs.alfworld_env import AlfWorldEnv
        except ImportError as e:
            logger.error(f"ALFWorld 依赖未安装: {e}")
            logger.info("请先安装: cd bench/alfworld && pip install -e .")
            logger.info("并设置环境变量: export ALFWORLD_DATA=/path/to/alfworld/data")
            return

        # 设置环境变量
        if not os.getenv("ALFWORLD_DATA") and args.data_path:
            os.environ["ALFWORLD_DATA"] = args.data_path

        env = AlfWorldEnv(
            data_path=args.data_path,
            split=args.split,
            max_games=args.num_tasks if args.num_tasks and args.num_tasks > 0 else -1,
            max_steps=args.max_steps,
            seed=args.seed,
        )

    elif env_name == "scienceworld":
        try:
            from reasoning_bank.envs.scienceworld_env import ScienceWorldEnv
        except ImportError as e:
            logger.error(f"ScienceWorld 依赖未安装: {e}")
            logger.info("请先安装: pip install scienceworld")
            return

        task_ids = None
        if args.task_ids:
            task_ids = args.task_ids.split(",")

        env = ScienceWorldEnv(
            task_ids=task_ids,
            simplifications=args.simplifications,
            max_steps=args.max_steps,
            split=args.split if args.split in [
                "train", "dev", "test"] else "dev",
            seed=args.seed,
        )

    if len(env) == 0:
        logger.error("没有找到可用的任务")
        return

    logger.info(f"加载 {env_name} 环境: {len(env)} 个任务")

    # 创建记忆库
    memory_bank = None
    if args.use_memory:
        memory_bank = MemoryBank(bank_name=env_name)
        if args.clear_memory:
            logger.info("清空记忆库")
            memory_bank.clear()
        logger.info(f"记忆库: {memory_bank}")

    # 创建 LLM 服务
    llm = LLMService(
        model=args.model,
        temperature=args.temperature,
    )

    # 创建 Extractor
    extractor = MemoryExtractor(
        llm_service=llm) if args.use_memory and not args.no_extract else None

    # 运行统计
    total = args.num_tasks if args.num_tasks and args.num_tasks > 0 else len(
        env)
    total = min(total, len(env))

    success_count = 0
    total_steps = 0
    memories_added = 0

    # 进度条
    task_ids = list(env)[:total]
    iterator = tqdm(task_ids, desc=f"Running {env_name}")

    for task_id in iterator:
        try:
            # 运行单个多轮任务
            result = _run_multiturn_episode(
                env, llm, memory_bank, task_id,
                args.max_steps, args.temperature, args.verbose
            )

            total_steps += result["steps"]

            if result["success"]:
                success_count += 1

            # 提取记忆
            if extractor and memory_bank:
                agent_result = AgentResult(
                    task_id=task_id,
                    query=result["task_desc"],
                    answer="TASK_COMPLETED" if result["success"] else "TASK_FAILED",
                    is_success=result["success"],
                    trajectory=result["trajectory"],
                    steps=result["steps"],
                )

                items = extractor.extract(agent_result, "")
                if items:
                    memory_bank.add(
                        query=result["task_desc"],
                        items=items,
                        is_success=result["success"],
                        trajectory_id=task_id,
                    )
                    memories_added += len(items)

            # 更新进度条
            current = iterator.n + 1
            iterator.set_postfix({
                "success": f"{success_count}/{current}",
                "rate": f"{success_count/current*100:.1f}%",
            })

        except Exception as e:
            logger.error(f"任务 {task_id} 执行失败: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()

    # 关闭环境
    env.close()

    # 打印统计
    logger.info("=" * 50)
    logger.info(f"多轮交互任务统计 ({env_name})")
    logger.info("=" * 50)
    logger.info(f"总任务数: {total}")
    logger.info(f"成功数: {success_count}")
    logger.info(f"成功率: {success_count/total*100:.2f}%")
    logger.info(f"平均步数: {total_steps/total:.2f}")
    if args.use_memory:
        logger.info(f"新增记忆: {memories_added}")
        if memory_bank:
            logger.info(f"记忆库大小: {len(memory_bank)}")
    logger.info("=" * 50)


def _run_multiturn_episode(env, llm, memory_bank, task_id, max_steps, temperature, verbose):
    """运行单个多轮 episode"""
    import re
    from reasoning_bank.prompts.registry import PromptRegistry

    # 重置环境
    observation = env.reset(task_id)
    task_info = env.get_task_info()
    task_desc = task_info.query if task_info else "Unknown task"

    if verbose:
        print(f"\n{'='*60}")
        print(f"任务: {task_id}")
        print(f"描述: {task_desc[:200]}...")
        print(f"{'='*60}")

    trajectory = []
    done = False
    success = False
    step = 0

    # 检索记忆
    memories = []
    if memory_bank:
        memories = memory_bank.retrieve(task_desc, top_k=1)
        if memories and verbose:
            print(f"[记忆] 检索到 {len(memories)} 条相关经验")

    while not done and step < max_steps:
        step += 1

        # 构建提示词
        if step == 1:
            prompt = f"Task: {task_desc}\n\nObservation:\n{observation}"
        else:
            prompt = f"Observation:\n{observation}"

        # 获取系统提示词
        system_prompt = PromptRegistry.get_system_prompt(
            prompt_type=env.name if env.name in [
                "alfworld", "scienceworld"] else "react",
            memories=memories,
            memory_formatter=lambda mems: "\n".join(
                m.format_for_prompt() for m in mems) if mems else "",
        )

        # 调用 LLM
        result = llm.call(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
        )

        if result.status != "success":
            break

        # 解析响应
        response = result.content
        thought, action = _parse_react_response(response)

        if verbose:
            print(f"\n[Step {step}]")
            print(f"  Thought: {thought[:100]}..." if len(
                thought) > 100 else f"  Thought: {thought}")
            print(f"  Action: {action}")

        # 执行动作
        step_result = env.step(action)
        observation = step_result.observation
        done = step_result.done

        if verbose:
            obs_display = observation[:100] + \
                "..." if len(observation) > 100 else observation
            print(f"  Observation: {obs_display}")

        # 检查是否成功
        if step_result.info.get("won", False):
            success = True
            done = True

        # 记录轨迹
        trajectory.append({
            "step": step,
            "observation": observation,
            "thought": thought,
            "action": action,
        })

    if verbose:
        print(f"\n结果: {'✅ SUCCESS' if success else '❌ FAILURE'} (步数: {step})")

    return {
        "task_id": task_id,
        "task_desc": task_desc,
        "success": success,
        "steps": step,
        "trajectory": trajectory,
    }


def _parse_react_response(response: str) -> tuple:
    """解析 ReAct 格式响应"""
    import re

    thought = ""
    action = ""

    # 匹配 THOUGHT/Thought 和 ACTION/Action
    thought_match = re.search(
        r'(?:THOUGHT|Thought|thought)[:\s]*(.+?)(?=(?:ACTION|Action|action)[:\s]|$)',
        response, re.DOTALL | re.IGNORECASE
    )
    action_match = re.search(
        r'(?:ACTION|Action|action)[:\s]*(.+?)(?:\n|$)',
        response, re.IGNORECASE
    )

    if thought_match:
        thought = thought_match.group(1).strip()

    if action_match:
        action = action_match.group(1).strip()
    else:
        # 取最后一行非空内容作为动作
        lines = [l.strip() for l in response.strip().split('\n') if l.strip()]
        if lines:
            action = lines[-1]

    return thought, action


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

    # 确定任务类型
    if args.dataset:
        task_type = "单轮 QA"
        task_name = args.dataset
    else:
        task_type = "多轮交互"
        task_name = args.env

    logger.info("=" * 60)
    logger.info("ReasoningBank 自我进化智能体")
    logger.info("=" * 60)
    logger.info(f"任务类型: {task_type}")
    logger.info(f"环境/数据集: {task_name}")
    logger.info(f"任务数量: {args.num_tasks or '全部'}")
    logger.info(f"使用记忆: {args.use_memory}")
    if args.dataset:
        logger.info(f"MaTTS 模式: {args.matts}")
    logger.info(f"模型: {args.model or get_config('llm.default_model')}")
    logger.info("=" * 60)

    # 运行
    if args.dataset:
        run_single_turn(args, logger)
    else:
        run_multi_turn(args, logger)

    logger.info("运行完成!")


if __name__ == "__main__":
    main()
