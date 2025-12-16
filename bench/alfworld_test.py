#!/usr/bin/env python
"""
ALFWorld LLM Agent 测试脚本

测试LLM在ALFWorld家居任务环境中的多轮交互能力。
采用ReAct风格的prompt，让LLM根据环境观察进行思考并选择动作。

使用方法:
    1. 确保安装依赖: cd alfworld && pip install -e .
    2. 设置环境变量: export ALFWORLD_DATA=/home/bmt/evo/bench/alfworld/data
    3. 运行: python alfworld_test.py

设定SEED和NUM_GAMES后确保结果可复现。
"""

from llms_api.call_openrouter_llm import call_openrouter_llm_with_retry
from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos, AlfredExpert, AlfredExpertType
import textworld.gym
import textworld
import yaml
import os
import sys
import json
import random
import re
from typing import Optional, Tuple, List, Dict
from datetime import datetime
from dotenv import load_dotenv

# 设置环境变量 (必须在导入alfworld之前)
load_dotenv()
ALFWORLD_DATA = os.getenv("ALFWORLD_DATA")
sys.path.insert(0, ALFWORLD_DATA)


# 导入LLM调用模块

# ============= 配置 =============
DEFAULT_MODEL = "qwen/qwen3-30b-a3b-instruct-2507"  # 默认测试的模型
MAX_STEPS = 30  # 每个任务的最大步数
NUM_GAMES = 5  # 测试的游戏数量
TEMPERATURE = 0.3  # LLM温度参数 (较低温度使输出更确定)
DEFAULT_SEED = 42  # 默认随机种子 (用于可复现的测试，设为None则随机)

# ============= Prompt 模板 =============

# 系统提示 - 介绍任务和可用动作
SYSTEM_PROMPT = """You are an intelligent agent operating in a text-based household environment. Your goal is to complete household tasks by interacting with objects in the environment.

Available actions:
- look: look around your current location
- inventory: check what you're carrying
- go to [receptacle]: move to a receptacle (e.g., "go to dresser 1", "go to fridge 1")
- open [receptacle]: open a receptacle (e.g., "open drawer 1")
- close [receptacle]: close a receptacle
- take [object] from [receptacle]: pick up an object (e.g., "take apple 1 from countertop 1")
- move [object] to [receptacle]: put an object down (e.g., "move apple 1 to fridge 1")
- examine [something]: examine an object or receptacle
- use [object]: turn on/off an object (e.g., "use desklamp 1")
- heat [object] with [receptacle]: heat an object using microwave
- clean [object] with [receptacle]: clean an object using sink
- cool [object] with [receptacle]: cool an object using fridge

You must respond in the following format:
THINK: [Your reasoning about what to do next based on the current observation]
ACTION: [The exact action to take, must be one of the available actions]

Only output one action at a time. Do NOT add any other text after the ACTION line."""

# 初始任务提示模板
TASK_PROMPT_TEMPLATE = """=== NEW TASK ===
{observation}

Remember to think step by step and output your response in the format:
THINK: [your reasoning]
ACTION: [your action]"""

# 每步交互的提示模板
STEP_PROMPT_TEMPLATE = """=== STEP RESULT ===
Your action: {action}
Observation: {observation}

What's your next action?
THINK: [your reasoning]
ACTION: [your action]"""

# Few-shot 示例 (可选，帮助模型理解任务格式)
FEW_SHOT_EXAMPLES = """
Here are some examples of how to complete tasks:

Example 1 - Put object task:
Task: put some book on desk
THINK: I need to find a book first. Let me look around to see what's available.
ACTION: look

Observation: You are in the middle of a room. Looking quickly around you, you see a bed 1, a desk 1, a drawer 1, a shelf 1.
THINK: I see there's a bed and desk. Books are often on shelves or beds. Let me check the bed first.
ACTION: go to bed 1

Observation: On the bed 1, you see a book 1, a pillow 1.
THINK: I found a book on the bed. Let me take it.
ACTION: take book 1 from bed 1

Observation: You pick up the book 1 from the bed 1.
THINK: Now I have the book. I need to put it on the desk. Let me go there.
ACTION: go to desk 1

Observation: On the desk 1, you see a lamp 1, a pencil 1.
THINK: I'm at the desk now. Let me put the book here to complete the task.
ACTION: move book 1 to desk 1

Observation: You move the book 1 to the desk 1.
Task completed!

Example 2 - Examine in light task:
Task: examine cd under the desklamp
THINK: I need to find a CD and then examine it under a desklamp. First, let me look around.
ACTION: look

Observation: You see a desk 1, a drawer 1, a sidetable 1.
THINK: Let me check the desk for the CD.
ACTION: go to desk 1

Observation: On the desk 1, you see a cd 1, a desklamp 1.
THINK: I found both the CD and desklamp. Let me take the CD first.
ACTION: take cd 1 from desk 1

Observation: You pick up the cd 1 from the desk 1.
THINK: Now I need to turn on the desklamp to examine the CD.
ACTION: use desklamp 1

Observation: You turn on the desklamp 1.
Task completed!

Now complete the following task:
"""


def load_game_files(data_path: str, split: str = "valid_seen", task_types: List[int] = None, max_games: int = -1, seed: int = None) -> List[str]:
    """加载游戏文件列表

    Args:
        data_path: 数据目录路径
        split: 数据集划分 (train/valid_seen/valid_unseen)
        task_types: 任务类型ID列表
        max_games: 最大游戏数量
        seed: 随机种子 (用于可复现的游戏选择)
    """
    from tqdm import tqdm

    TASK_TYPE_MAP = {
        1: "pick_and_place_simple",
        2: "look_at_obj_in_light",
        3: "pick_clean_then_place_in_recep",
        4: "pick_heat_then_place_in_recep",
        5: "pick_cool_then_place_in_recep",
        6: "pick_two_obj_and_place"
    }

    if task_types is None:
        task_types = [1, 2, 3, 4, 5, 6]

    task_type_names = [TASK_TYPE_MAP[t]
                       for t in task_types if t in TASK_TYPE_MAP]

    split_path = os.path.join(data_path, "json_2.1.1", split)
    game_files = []

    print(f"正在从 {split_path} 加载游戏文件...")

    for root, dirs, files in os.walk(split_path):
        if 'game.tw-pddl' in files:
            # 检查任务类型
            traj_path = os.path.join(root, 'traj_data.json')
            if os.path.exists(traj_path):
                with open(traj_path, 'r') as f:
                    traj_data = json.load(f)

                if traj_data['task_type'] in task_type_names:
                    game_file = os.path.join(root, 'game.tw-pddl')

                    # 检查是否可解
                    with open(game_file, 'r') as f:
                        game_data = json.load(f)

                    if game_data.get('solvable', False):
                        # 排除movable receptacle任务
                        if 'movable' not in root and 'Sliced' not in root:
                            game_files.append(game_file)

    # 先排序确保顺序一致，再用种子随机打乱
    game_files.sort()

    if seed is not None:
        rng = random.Random(seed)
        rng.shuffle(game_files)
        print(f"使用随机种子: {seed} (结果可复现)")
    else:
        random.shuffle(game_files)
        print("未设置随机种子 (结果不可复现)")

    if max_games > 0:
        game_files = game_files[:max_games]

    print(f"找到 {len(game_files)} 个可用游戏")
    return game_files


def create_environment(game_file: str, max_steps: int = 50):
    """创建单个游戏环境"""

    # 设置环境wrapper
    alfred_demangler = AlfredDemangler(shuffle=False)
    wrappers = [alfred_demangler, AlfredInfos]

    # 注册环境
    request_infos = textworld.EnvInfos(
        won=True,
        admissible_commands=True,
        extras=["gamefile"]
    )

    env_id = textworld.gym.register_game(
        game_file,
        request_infos,
        max_episode_steps=max_steps,
        wrappers=wrappers
    )

    env = textworld.gym.make(env_id)
    return env


def parse_llm_response(response: str) -> Tuple[str, str]:
    """解析LLM响应，提取思考和动作"""
    think = ""
    action = ""

    # 处理 /no_think 标记（如果有的话，去除它）
    response = response.replace("/no_think", "").strip()

    # 尝试解析 THINK 和 ACTION
    lines = response.strip().split('\n')

    for i, line in enumerate(lines):
        line_stripped = line.strip()

        # 解析 THINK
        if line_stripped.upper().startswith('THINK:'):
            think = line_stripped[6:].strip()
        elif line_stripped.upper().startswith('THINK '):
            think = line_stripped[6:].strip()

        # 解析 ACTION
        if line_stripped.upper().startswith('ACTION:'):
            action = line_stripped[7:].strip()
            break
        elif line_stripped.upper().startswith('ACTION '):
            action = line_stripped[7:].strip()
            break

    # 如果没找到明确的ACTION标签，尝试提取最后一行作为动作
    if not action:
        for line in reversed(lines):
            line = line.strip()
            if line and not line.upper().startswith('THINK'):
                action = line
                break

    # 清理action中可能的引号
    action = action.strip('"\'')

    return think, action


def run_episode(env, model: str, use_few_shot: bool = True, verbose: bool = True) -> Dict:
    """运行单个episode，使用LLM作为agent"""

    # 重置环境
    obs, info = env.reset()

    # 构建历史记录
    conversation_history = []
    actions_taken = []
    observations = [obs]

    done = False
    step = 0
    total_reward = 0

    # 构建初始prompt
    if use_few_shot:
        initial_prompt = FEW_SHOT_EXAMPLES + \
            "\n" + TASK_PROMPT_TEMPLATE.format(observation=obs)
    else:
        initial_prompt = TASK_PROMPT_TEMPLATE.format(observation=obs)

    if verbose:
        print("\n" + "="*60)
        print("初始观察:")
        print(obs)
        print("="*60)

    while not done and step < MAX_STEPS:
        step += 1

        # 构建当前prompt
        if step == 1:
            current_prompt = initial_prompt
        else:
            current_prompt = initial_prompt + "\n\n" + \
                "\n\n".join(conversation_history)

        # 调用LLM (系统提示词通过 system_prompt 参数传递)
        try:
            response = call_openrouter_llm_with_retry(
                current_prompt,
                model=model,
                stream=False,
                temperature=TEMPERATURE,
                max_tokens=512,
                system_prompt=SYSTEM_PROMPT
            )
        except Exception as e:
            print(f"LLM调用失败: {e}")
            break

        # 解析响应
        think, action = parse_llm_response(response)

        if verbose:
            print(f"\n--- Step {step} ---")
            print(f"LLM Think: {think}")
            print(f"LLM Action: {action}")

        if not action:
            if verbose:
                print("警告: 无法解析有效的动作")
            action = "look"  # 默认动作

        # 检查动作是否在可用命令中
        admissible = info.get('admissible_commands', [])
        if action not in admissible and admissible:
            # 尝试模糊匹配
            matched = False
            action_lower = action.lower()
            for cmd in admissible:
                if action_lower == cmd.lower():
                    action = cmd
                    matched = True
                    break

            if not matched and verbose:
                print(f"注意: 动作 '{action}' 不在可用命令中")
                print(f"可用命令: {admissible[:10]}..." if len(
                    admissible) > 10 else f"可用命令: {admissible}")

        # 执行动作
        obs, reward, done, info = env.step(action)
        total_reward += reward

        actions_taken.append(action)
        observations.append(obs)

        if verbose:
            print(f"环境响应: {obs[:200]}..." if len(
                obs) > 200 else f"环境响应: {obs}")

        # 更新对话历史
        step_record = STEP_PROMPT_TEMPLATE.format(
            action=action, observation=obs)
        conversation_history.append(
            f"THINK: {think}\nACTION: {action}\n{step_record}")

        # 检查是否完成
        if info.get('won', False):
            if verbose:
                print(f"\n🎉 任务完成! 步数: {step}")
            break
        elif done:
            # done=True 但 won=False 表示超时或其他原因结束
            if verbose:
                print(f"\n⏱️ 环境结束 (未完成任务), 步数: {step}")
            break

    # 循环因达到 MAX_STEPS 自然结束
    else:
        if verbose and not info.get('won', False):
            print(f"\n⏱️ 达到最大步数限制 ({MAX_STEPS}步), 任务未完成")

    return {
        "success": info.get('won', False),  # 只有 won=True 才算成功
        "steps": step,
        "actions": actions_taken,
        "observations": observations,
        "reward": total_reward
    }


def run_benchmark(
    model: str = DEFAULT_MODEL,
    num_games: int = NUM_GAMES,
    task_types: List[int] = None,
    use_few_shot: bool = True,
    verbose: bool = True,
    output_file: str = None,
    seed: int = DEFAULT_SEED
):
    """运行ALFWorld基准测试"""

    data_path = os.environ.get("ALFWORLD_DATA")

    print(f"\n{'='*60}")
    print(f"ALFWorld LLM Agent 测试")
    print(f"{'='*60}")
    print(f"模型: {model}")
    print(f"游戏数量: {num_games}")
    print(f"任务类型: {task_types if task_types else 'all'}")
    print(f"使用Few-shot: {use_few_shot}")
    print(f"随机种子: {seed if seed is not None else '随机'}")
    print(f"数据路径: {data_path}")
    print(f"{'='*60}\n")

    # 加载游戏文件
    game_files = load_game_files(
        data_path,
        split="valid_seen",
        task_types=task_types,
        max_games=num_games,
        seed=seed
    )

    if not game_files:
        print("错误: 没有找到可用的游戏文件!")
        return

    # 运行测试
    results = []
    successes = 0
    total_steps = 0

    for i, game_file in enumerate(game_files):
        print(f"\n{'='*60}")
        print(f"游戏 {i+1}/{len(game_files)}")
        print(f"文件: {os.path.basename(os.path.dirname(game_file))}")
        print(f"{'='*60}")

        try:
            env = create_environment(game_file, max_steps=MAX_STEPS)
            result = run_episode(
                env, model, use_few_shot=use_few_shot, verbose=verbose)
            env.close()

            result['game_file'] = game_file
            results.append(result)

            if result['success']:
                successes += 1
            total_steps += result['steps']

            print(
                f"\n结果: {'✅ 成功' if result['success'] else '❌ 失败'} (步数: {result['steps']})")

        except Exception as e:
            print(f"游戏运行出错: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "success": False,
                "steps": 0,
                "error": str(e),
                "game_file": game_file
            })

    # 统计结果
    print(f"\n{'='*60}")
    print(f"测试结果汇总")
    print(f"{'='*60}")
    print(f"模型: {model}")
    print(f"总游戏数: {len(game_files)}")
    print(f"成功数: {successes}")
    print(f"成功率: {successes/len(game_files)*100:.1f}%")
    print(f"平均步数: {total_steps/len(game_files):.1f}")
    print(f"{'='*60}")

    # 保存结果
    if output_file:
        summary = {
            "model": model,
            "timestamp": datetime.now().isoformat(),
            "config": {
                "num_games": num_games,
                "task_types": task_types,
                "use_few_shot": use_few_shot,
                "max_steps": MAX_STEPS,
                "temperature": TEMPERATURE,
                "seed": seed
            },
            "summary": {
                "total_games": len(game_files),
                "successes": successes,
                "success_rate": successes/len(game_files),
                "avg_steps": total_steps/len(game_files)
            },
            "results": results
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"\n结果已保存到: {output_file}")

    return results


def demo_single_game(model: str = DEFAULT_MODEL, game_idx: int = 0, seed: int = DEFAULT_SEED):
    """运行单个游戏的演示"""

    data_path = os.environ.get("ALFWORLD_DATA")

    # 加载游戏 (使用种子确保可复现)
    game_files = load_game_files(
        data_path, split="valid_seen", max_games=100, seed=seed)

    if game_idx >= len(game_files):
        print(f"游戏索引超出范围，只有 {len(game_files)} 个游戏")
        return

    game_file = game_files[game_idx]

    print(f"\n运行演示游戏: {os.path.basename(os.path.dirname(game_file))}")

    env = create_environment(game_file, max_steps=MAX_STEPS)
    result = run_episode(env, model, use_few_shot=True, verbose=True)
    env.close()

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ALFWorld LLM Agent 测试")
    parser.add_argument("--model", type=str,
                        default=DEFAULT_MODEL, help="LLM模型名称")
    parser.add_argument("--num_games", type=int,
                        default=NUM_GAMES, help="测试的游戏数量")
    parser.add_argument("--task_types", type=int, nargs="+", default=None,
                        help="任务类型 (1-6)")
    parser.add_argument(
        "--no_few_shot", action="store_true", help="不使用few-shot示例")
    parser.add_argument("--quiet", action="store_true", help="减少输出")
    parser.add_argument("--output", type=str, default=None, help="结果输出文件")
    parser.add_argument("--demo", action="store_true", help="运行单个游戏演示")
    parser.add_argument("--game_idx", type=int, default=0, help="演示游戏的索引")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="随机种子 (用于可复现的游戏选择，默认42)")
    parser.add_argument("--no_seed", action="store_true",
                        help="不使用固定种子 (完全随机选择游戏)")

    args = parser.parse_args()

    # 处理种子参数
    seed = None if args.no_seed else args.seed

    if args.demo:
        demo_single_game(model=args.model, game_idx=args.game_idx, seed=seed)
    else:
        # 默认输出文件名
        if args.output is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = args.model.replace("/", "_")
            args.output = f"/home/bmt/evo/bench/alfworld_results_{model_name}_{timestamp}.json"

        run_benchmark(
            model=args.model,
            num_games=args.num_games,
            task_types=args.task_types,
            use_few_shot=not args.no_few_shot,
            verbose=not args.quiet,
            output_file=args.output,
            seed=seed
        )

# 🎮 运行单个游戏演示
# python3 alfworld_test.py --demo --model "qwen/qwen-2.5-7b-instruct"

# 📊 运行完整测试 (5个游戏)
# python3 alfworld_test.py --model "qwen/qwen3-8b" --num_games 5

# 🎯 测试特定任务类型
# python3 alfworld_test.py --model "qwen/qwen3-8b" --task_types 1 2 --num_games 3

# 🔇 安静模式 (减少输出)
# python3 alfworld_test.py --model "qwen/qwen3-8b" --num_games 10 --quiet
