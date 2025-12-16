#!/usr/bin/env python
"""
多轮交互任务运行脚本

支持 ALFWorld 和 ScienceWorld 环境
"""

import argparse
import os
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

from reasoning_bank.utils.config import load_config
from reasoning_bank.utils.logger import get_logger, setup_logging
from reasoning_bank.core.llm_service import LLMService
from reasoning_bank.core.memory import MemoryBank
from reasoning_bank.core.agent import ReActAgent, AgentConfig
from reasoning_bank.core.extractor import MemoryExtractor

logger = get_logger("run_multiturn")


def run_alfworld(args):
    """运行 ALFWorld 多轮任务"""
    try:
        from reasoning_bank.envs.alfworld_env import AlfWorldEnv
    except ImportError as e:
        logger.error(f"ALFWorld 依赖未安装: {e}")
        logger.info("请先安装: cd bench/alfworld && pip install -e .")
        logger.info("并设置环境变量: export ALFWORLD_DATA=/path/to/alfworld/data")
        return
    
    # 检查环境变量
    if not os.getenv("ALFWORLD_DATA"):
        data_path = args.data_path or "./bench/alfworld/data"
        os.environ["ALFWORLD_DATA"] = data_path
        logger.info(f"设置 ALFWORLD_DATA={data_path}")
    
    # 创建环境
    env = AlfWorldEnv(
        data_path=args.data_path,
        split=args.split,
        max_games=args.num_tasks if args.num_tasks > 0 else -1,
        max_steps=args.max_steps,
        seed=args.seed,
    )
    
    if len(env) == 0:
        logger.error("没有找到可用的游戏文件，请检查数据路径")
        return
    
    logger.info(f"加载 {len(env)} 个 ALFWorld 游戏")
    
    # 运行任务
    run_multiturn_tasks(env, args, "alfworld")


def run_scienceworld(args):
    """运行 ScienceWorld 多轮任务"""
    try:
        from reasoning_bank.envs.scienceworld_env import ScienceWorldEnv
    except ImportError as e:
        logger.error(f"ScienceWorld 依赖未安装: {e}")
        logger.info("请先安装: pip install scienceworld")
        return
    
    # 解析任务 ID
    task_ids = None
    if args.task_ids:
        task_ids = args.task_ids.split(",")
    
    # 创建环境
    env = ScienceWorldEnv(
        task_ids=task_ids,
        simplifications=args.simplifications,
        max_steps=args.max_steps,
        episodes_per_task=args.episodes_per_task,
        split=args.split,
        seed=args.seed,
    )
    
    if len(env) == 0:
        logger.error("任务队列为空")
        return
    
    logger.info(f"加载 {len(env)} 个 ScienceWorld 任务")
    
    # 运行任务
    run_multiturn_tasks(env, args, "scienceworld")


def run_multiturn_tasks(env, args, env_name: str):
    """通用多轮任务运行函数"""
    from tqdm import tqdm
    
    # 创建 LLM 服务
    llm = LLMService(
        model=args.model,
        temperature=args.temperature,
    )
    
    # 创建记忆库
    memory_bank = None
    if args.use_memory:
        memory_bank = MemoryBank(bank_name=env_name)
        if args.clear_memory:
            memory_bank.clear()
            logger.info("已清空记忆库")
        logger.info(f"记忆库: {memory_bank.get_stats()}")
    
    # 创建 Agent
    agent_config = AgentConfig(
        max_steps=args.max_steps,
        use_react=True,
        temperature=args.temperature,
        verbose=args.verbose,
    )
    
    agent = ReActAgent(
        llm_service=llm,
        memory_bank=memory_bank,
        config=agent_config,
    )
    
    # 创建 Extractor
    extractor = MemoryExtractor(llm_service=llm) if args.use_memory else None
    
    # 运行统计
    total = args.num_tasks if args.num_tasks > 0 else len(env)
    total = min(total, len(env))
    
    success_count = 0
    total_steps = 0
    memories_added = 0
    
    # 进度条
    task_ids = list(env)[:total]
    iterator = tqdm(task_ids, desc=f"Running {env_name}")
    
    for task_id in iterator:
        try:
            # 运行单个任务
            result = run_single_task(env, agent, task_id, args.max_steps, args.verbose)
            
            total_steps += result["steps"]
            
            if result["success"]:
                success_count += 1
            
            # 提取记忆
            if extractor and memory_bank:
                # 构造 AgentResult 格式
                from reasoning_bank.core.agent import AgentResult
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
            iterator.set_postfix({
                "success": f"{success_count}/{len(list(iterator)[:iterator.n+1])}",
                "rate": f"{success_count/(iterator.n+1)*100:.1f}%",
            })
            
        except Exception as e:
            logger.error(f"任务 {task_id} 执行失败: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
    
    # 关闭环境
    env.close()
    
    # 打印统计
    print("\n" + "=" * 50)
    print(f"多轮交互任务统计 ({env_name})")
    print("=" * 50)
    print(f"总任务数: {total}")
    print(f"成功数: {success_count}")
    print(f"成功率: {success_count/total*100:.2f}%")
    print(f"平均步数: {total_steps/total:.2f}")
    if args.use_memory:
        print(f"新增记忆: {memories_added}")
        print(f"记忆库大小: {len(memory_bank)}")
    print("=" * 50)


def run_single_task(env, agent, task_id: str, max_steps: int, verbose: bool) -> dict:
    """运行单个多轮任务"""
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
    if agent.memory_bank:
        memories = agent.memory_bank.retrieve(task_desc, top_k=1)
        if memories and verbose:
            print(f"[记忆] 检索到 {len(memories)} 条相关经验")
    
    while not done and step < max_steps:
        step += 1
        
        # 构建提示词
        if step == 1:
            prompt = f"Task: {task_desc}\n\nObservation:\n{observation}"
        else:
            prompt = f"Observation:\n{observation}"
        
        # Agent 生成动作
        from reasoning_bank.prompts.registry import PromptRegistry
        
        # 格式化记忆
        memory_text = ""
        if memories:
            memory_text = "\n".join(m.format_for_prompt() for m in memories)
        
        system_prompt = PromptRegistry.get_system_prompt(
            prompt_type=env.name if env.name in ["alfworld", "scienceworld"] else "react",
            memories=memories,
            memory_formatter=lambda mems: "\n".join(m.format_for_prompt() for m in mems) if mems else "",
        )
        
        # 调用 LLM
        result = agent.llm.call(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=agent.config.temperature,
        )
        
        if result.status != "success":
            logger.error(f"LLM 调用失败: {result.error}")
            break
        
        # 解析响应
        response = result.content
        thought, action = parse_react_response(response)
        
        if verbose:
            print(f"\n[Step {step}]")
            print(f"  Thought: {thought[:100]}..." if len(thought) > 100 else f"  Thought: {thought}")
            print(f"  Action: {action}")
        
        # 执行动作
        step_result = env.step(action)
        observation = step_result.observation
        done = step_result.done
        
        if verbose:
            print(f"  Observation: {observation[:100]}..." if len(observation) > 100 else f"  Observation: {observation}")
        
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


def parse_react_response(response: str) -> tuple:
    """解析 ReAct 格式响应"""
    import re
    
    thought = ""
    action = ""
    
    # 尝试匹配 THOUGHT: 和 ACTION:
    thought_match = re.search(r'(?:THOUGHT|Thought|thought)[:\s]*(.+?)(?=(?:ACTION|Action|action)[:\s]|$)', response, re.DOTALL | re.IGNORECASE)
    action_match = re.search(r'(?:ACTION|Action|action)[:\s]*(.+?)(?:\n|$)', response, re.IGNORECASE)
    
    if thought_match:
        thought = thought_match.group(1).strip()
    
    if action_match:
        action = action_match.group(1).strip()
    else:
        # 如果没有找到 ACTION，取最后一行非空内容作为动作
        lines = [l.strip() for l in response.strip().split('\n') if l.strip()]
        if lines:
            action = lines[-1]
    
    return thought, action


def main():
    parser = argparse.ArgumentParser(
        description="ReasoningBank 多轮交互任务运行器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 运行 ALFWorld
  python -m reasoning_bank.run_multiturn --env alfworld --num-tasks 5

  # 运行 ScienceWorld  
  python -m reasoning_bank.run_multiturn --env scienceworld --task-ids "1-1,1-2,4-1" --num-tasks 10

  # 使用记忆库
  python -m reasoning_bank.run_multiturn --env alfworld --use-memory --num-tasks 20

  # 指定模型
  python -m reasoning_bank.run_multiturn --env scienceworld -m "qwen/qwen3-32b" --num-tasks 5
        """
    )
    
    # 环境选择
    parser.add_argument("--env", "-e", type=str, required=True, 
                        choices=["alfworld", "scienceworld"],
                        help="环境类型")
    
    # 通用参数
    parser.add_argument("--num-tasks", "-n", type=int, default=5,
                        help="运行的任务数量（默认 5）")
    parser.add_argument("--max-steps", type=int, default=30,
                        help="每个任务的最大步数（默认 30）")
    parser.add_argument("--model", "-m", type=str, default=None,
                        help="LLM 模型名称")
    parser.add_argument("--temperature", "-t", type=float, default=0.3,
                        help="生成温度")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="详细输出")
    
    # 记忆库参数
    parser.add_argument("--use-memory", action="store_true",
                        help="使用记忆库")
    parser.add_argument("--clear-memory", action="store_true",
                        help="清空记忆库后运行")
    
    # ALFWorld 特定参数
    parser.add_argument("--data-path", type=str, default=None,
                        help="ALFWorld 数据路径")
    parser.add_argument("--split", type=str, default="valid_seen",
                        choices=["train", "valid_seen", "valid_unseen", "dev", "test"],
                        help="数据集划分")
    
    # ScienceWorld 特定参数
    parser.add_argument("--task-ids", type=str, default=None,
                        help="ScienceWorld 任务 ID 列表（逗号分隔，如 '1-1,1-2,4-1'）")
    parser.add_argument("--simplifications", type=str, default="easy",
                        help="ScienceWorld 简化设置")
    parser.add_argument("--episodes-per-task", type=int, default=1,
                        help="每个任务的变体数量")
    
    args = parser.parse_args()
    
    # 加载配置
    load_config()
    setup_logging()
    
    # 运行
    if args.env == "alfworld":
        run_alfworld(args)
    elif args.env == "scienceworld":
        run_scienceworld(args)


if __name__ == "__main__":
    main()

