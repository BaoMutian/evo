#!/usr/bin/env python
"""
ScienceWorld LLM Agent 测试脚本

测试LLM在ScienceWorld科学实验环境中的多轮交互能力。
采用ReAct风格的prompt，让LLM根据环境观察进行思考并选择动作。

使用方法:
    1. 确保安装依赖: pip install scienceworld
    2. 确保 Java 1.8+ 已安装
    3. 运行: python scienceworld_test.py

设定SEED和NUM_EPISODES后确保结果可复现。
"""

from llms_api.call_openrouter_llm import call_openrouter_llm_with_retry
from scienceworld import ScienceWorldEnv
import os
import sys
import json
import random
import re
import time
from typing import Optional, Tuple, List, Dict, Any
from datetime import datetime
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 导入 ScienceWorld

# 导入LLM调用模块

# ============= 配置 =============
DEFAULT_MODEL = "qwen/qwen3-30b-a3b-instruct-2507"  # 默认测试的模型
MAX_STEPS = 30    # 每个任务的最大步数
NUM_EPISODES = 5  # 每个任务测试的episode数量
TEMPERATURE = 0.3  # LLM温度参数 (较低温度使输出更确定)
DEFAULT_SEED = 42  # 默认随机种子
DEFAULT_SIMPLIFICATIONS = "easy"  # 默认简化设置

# ============= 任务配置 =============
# 任务ID到任务信息的映射
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

# ============= Prompt 模板 =============

# 系统提示 - 介绍任务和可用动作
SYSTEM_PROMPT = """You are an intelligent agent operating in a text-based science simulation environment called ScienceWorld. Your goal is to complete various science experiment tasks (like boiling water, measuring temperature, finding living things, etc.).

Available action types:
- look around: observe your current location
- inventory: check what you're carrying
- go to [location]: move to a specific location (e.g., "go to kitchen", "go to outside")
- teleport to [location]: instantly move to any location (if teleport is enabled)
- open [container]: open a container or door (e.g., "open fridge", "open cupboard")
- close [container]: close a container or door
- pick up [object]: pick up an object (e.g., "pick up thermometer")
- put down [object]: put down an object you're carrying
- move [object] to [location/container]: move an object somewhere (e.g., "move pot to stove")
- activate [device]: turn on a device (e.g., "activate stove", "activate sink")
- deactivate [device]: turn off a device
- use [object] on [target]: use an object on something (e.g., "use thermometer on water")
- pour [substance] into [container]: pour liquid into container
- focus on [object]: focus your attention on a specific object (required for some tasks)
- connect [obj1] to [obj2]: connect objects (for electrical circuits)
- wait / wait1: wait for time to pass (important for processes like boiling, melting, plant growth)
- read [object]: read something (e.g., read a thermometer)
- examine [object]: examine an object closely
- task: see the current task description

Important tips:
1. Use "wait" command multiple times when processes need time (heating, cooling, growing)
2. Some containers need to be opened before you can access items inside
3. Focus on the target substance/object when required by the task
4. Pay attention to the task description for specific requirements

You must respond in the following format:
THINK: [Your reasoning about what to do next based on the observation and task]
ACTION: [The exact action to take]

Only output one action at a time. Do NOT add any other text after the ACTION line."""

# 初始任务提示模板
TASK_PROMPT_TEMPLATE = """=== NEW TASK ===
Task Description: {task_description}

Current Observation:
{observation}

Valid actions at this step (partial list):
{valid_actions}

Remember to think step by step and output your response in the format:
THINK: [your reasoning]
ACTION: [your action]"""

# 每步交互的提示模板
STEP_PROMPT_TEMPLATE = """=== STEP RESULT ===
Your action: {action}
Observation: {observation}
Current score: {score}/100
Steps taken: {steps}

Valid actions (partial list):
{valid_actions}

What's your next action?
THINK: [your reasoning]
ACTION: [your action]"""

# Few-shot 示例
FEW_SHOT_EXAMPLES = """
Here are some examples of how to complete tasks:

Example 1 - Melting ice:
Task: Your task is to melt ice.
THINK: I need to find ice first. It's likely in the fridge. Let me check.
ACTION: open fridge

Observation: The fridge is now open. In the fridge, you see: ice cube, butter, milk.
THINK: I found ice cube. Let me pick it up.
ACTION: pick up ice cube

Observation: You pick up the ice cube.
THINK: I need to focus on the ice cube for this task.
ACTION: focus on ice cube

Observation: You focus on the ice cube.
THINK: To melt ice, I need heat. I should put it on the stove.
ACTION: move ice cube to stove

Observation: You move the ice cube to the stove.
THINK: Now I need to activate the stove to heat it.
ACTION: activate stove

Observation: The stove is now activated.
THINK: I need to wait for the ice to melt.
ACTION: wait

Observation: The ice cube is melting...
THINK: Let me wait more for it to fully melt.
ACTION: wait

Observation: The ice cube has melted into water!
Task completed! Score: 100/100

Example 2 - Finding a living thing:
Task: Your task is to find a living thing and put it in the red box.
THINK: I need to find something that is alive. Let me look around.
ACTION: look around

Observation: You are outside. You see: a bee on a flower, a tree, a red box, a path leading to the house.
THINK: A bee is a living thing! Let me pick it up.
ACTION: pick up bee

Observation: You carefully pick up the bee.
THINK: Now I need to put the bee in the red box to complete the task.
ACTION: move bee to red box

Observation: You move the bee to the red box.
Task completed! Score: 100/100

Now complete the following task:
"""


def get_task_name_from_id(task_id: str) -> str:
    """从任务ID获取任务名称"""
    if task_id in TASK_INFO:
        return TASK_INFO[task_id]["name"]
    raise ValueError(f"Unknown task ID: {task_id}")


def get_all_task_ids() -> List[str]:
    """获取所有任务ID"""
    return list(TASK_INFO.keys())


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


def format_valid_actions(valid_actions: List[str], max_display: int = 15) -> str:
    """格式化有效动作列表，只显示部分"""
    if len(valid_actions) <= max_display:
        return "\n".join(f"  - {a}" for a in valid_actions)
    else:
        shown = valid_actions[:max_display]
        return "\n".join(f"  - {a}" for a in shown) + f"\n  ... and {len(valid_actions) - max_display} more actions"


def run_episode(
    env: ScienceWorldEnv,
    model: str,
    task_id: str,
    variation_idx: int,
    use_few_shot: bool = True,
    verbose: bool = True,
    max_steps: int = MAX_STEPS
) -> Dict[str, Any]:
    """运行单个episode，使用LLM作为agent"""

    # 重置环境
    initial_obs, initial_info = env.reset()

    # 获取任务描述
    task_description = env.get_task_description()

    # 构建历史记录
    conversation_history = []
    actions_taken = []
    observations = [initial_obs]

    done = False
    step = 0
    score = 0

    # 获取初始有效动作
    valid_actions = env.get_valid_action_object_combinations()
    valid_actions_str = format_valid_actions(valid_actions)

    # 构建初始prompt
    task_prompt = TASK_PROMPT_TEMPLATE.format(
        task_description=task_description,
        observation=initial_obs,
        valid_actions=valid_actions_str
    )

    if use_few_shot:
        initial_prompt = FEW_SHOT_EXAMPLES + "\n" + task_prompt
    else:
        initial_prompt = task_prompt

    if verbose:
        print("\n" + "="*60)
        print(f"任务: {TASK_INFO[task_id]['desc']} ({task_id})")
        print(f"变体: {variation_idx}")
        print(f"任务描述: {task_description}")
        print("-"*60)
        print("初始观察:")
        print(initial_obs[:500] + "..." if len(initial_obs)
              > 500 else initial_obs)
        print("="*60)

    while not done and step < max_steps:
        step += 1

        # 构建当前prompt
        if step == 1:
            current_prompt = initial_prompt
        else:
            current_prompt = initial_prompt + "\n\n" + \
                "\n\n".join(conversation_history)

        # 调用LLM
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
            print(f"LLM Think: {think[:200]}..." if len(
                think) > 200 else f"LLM Think: {think}")
            print(f"LLM Action: {action}")

        if not action:
            if verbose:
                print("警告: 无法解析有效的动作")
            action = "look around"  # 默认动作

        # 检查动作是否在有效动作列表中，尝试模糊匹配
        if action not in valid_actions and valid_actions:
            action_lower = action.lower()
            matched = False
            for cmd in valid_actions:
                if action_lower == cmd.lower():
                    action = cmd
                    matched = True
                    break
            if not matched and verbose:
                print(f"注意: 动作 '{action}' 不在有效命令中")

        # 执行动作
        obs, reward, done, info = env.step(action)
        score = info['score']

        actions_taken.append(action)
        observations.append(obs)

        if verbose:
            obs_display = obs[:300] + "..." if len(obs) > 300 else obs
            print(f"环境响应: {obs_display}")
            print(f"分数: {score}/100")

        # 获取新的有效动作
        valid_actions = info.get('valid', [])
        valid_actions_str = format_valid_actions(valid_actions)

        # 更新对话历史
        step_record = STEP_PROMPT_TEMPLATE.format(
            action=action,
            observation=obs,
            score=score,
            steps=step,
            valid_actions=valid_actions_str
        )
        conversation_history.append(
            f"THINK: {think}\nACTION: {action}\n{step_record}")

        # 检查是否完成（分数达到100）
        if score >= 100:
            if verbose:
                print(f"\n🎉 任务完成! 分数: {score}, 步数: {step}")
            break
        elif done:
            if verbose:
                print(f"\n⏱️ 环境结束，分数: {score}, 步数: {step}")
            break

    # 循环因达到 MAX_STEPS 自然结束
    if step >= max_steps and not done and score < 100:
        if verbose:
            print(f"\n⏱️ 达到最大步数限制 ({max_steps}步), 分数: {score}")

    return {
        "success": score >= 100,
        "score": score,
        "steps": step,
        "actions": actions_taken,
        "observations": observations,
        "task_id": task_id,
        "task_name": TASK_INFO[task_id]["name"],
        "variation_idx": variation_idx
    }


def run_benchmark(
    model: str = DEFAULT_MODEL,
    num_episodes: int = NUM_EPISODES,
    task_ids: List[str] = None,
    simplifications: str = DEFAULT_SIMPLIFICATIONS,
    max_steps: int = MAX_STEPS,
    use_few_shot: bool = True,
    verbose: bool = True,
    output_file: str = None,
    seed: int = DEFAULT_SEED,
    split: str = "dev"
):
    """运行ScienceWorld基准测试"""

    # 如果没有指定任务，使用所有任务
    if task_ids is None:
        task_ids = get_all_task_ids()

    # 验证任务ID
    for tid in task_ids:
        if tid not in TASK_INFO:
            print(f"错误: 未知的任务ID '{tid}'")
            print(f"可用的任务ID: {list(TASK_INFO.keys())}")
            return

    print(f"\n{'='*60}")
    print(f"ScienceWorld LLM Agent 测试")
    print(f"{'='*60}")
    print(f"模型: {model}")
    print(f"任务数量: {len(task_ids)}")
    print(f"每任务Episode数: {num_episodes}")
    print(f"简化设置: {simplifications}")
    print(f"最大步数: {max_steps}")
    print(f"使用Few-shot: {use_few_shot}")
    print(f"随机种子: {seed if seed is not None else '随机'}")
    print(f"数据集划分: {split}")
    print(f"{'='*60}\n")

    # 设置随机种子
    if seed is not None:
        random.seed(seed)

    # 初始化环境
    env = ScienceWorldEnv("", envStepLimit=max_steps + 10)
    task_names = env.get_task_names()

    # 运行测试
    all_results = []
    task_stats = {}

    total_episodes = 0
    total_successes = 0
    total_score = 0
    total_steps = 0

    for task_id in task_ids:
        task_name = TASK_INFO[task_id]["name"]
        task_topic = TASK_INFO[task_id]["topic"]
        task_desc = TASK_INFO[task_id]["desc"]

        print(f"\n{'='*60}")
        print(f"任务 {task_id}: {task_desc} ({task_name})")
        print(f"主题: {task_topic}")
        print(f"{'='*60}")

        # 加载任务获取变体信息
        try:
            env.load(task_name, 0, simplifications)
        except Exception as e:
            print(f"加载任务失败: {e}")
            continue

        # 获取变体列表
        if split == "train":
            variations = env.get_variations_train()
        elif split == "dev":
            variations = env.get_variations_dev()
        else:
            variations = env.get_variations_test()

        if not variations:
            print(f"警告: 任务 {task_id} 没有可用的 {split} 变体")
            continue

        max_variations = env.get_max_variations(task_name)
        print(f"可用变体数: {len(variations)} / {max_variations}")

        # 随机选择变体
        if seed is not None:
            selected_variations = random.sample(
                variations, min(num_episodes, len(variations)))
        else:
            selected_variations = random.sample(
                variations, min(num_episodes, len(variations)))

        task_successes = 0
        task_score = 0
        task_steps = 0
        task_results = []

        for ep_idx, var_idx in enumerate(selected_variations):
            print(
                f"\n--- Episode {ep_idx + 1}/{len(selected_variations)} (变体 {var_idx}) ---")

            try:
                # 加载特定变体
                env.load(task_name, var_idx, simplifications)

                # 运行episode
                result = run_episode(
                    env, model, task_id, var_idx,
                    use_few_shot=use_few_shot,
                    verbose=verbose,
                    max_steps=max_steps
                )

                task_results.append(result)
                all_results.append(result)

                if result['success']:
                    task_successes += 1
                    total_successes += 1
                task_score += result['score']
                total_score += result['score']
                task_steps += result['steps']
                total_steps += result['steps']
                total_episodes += 1

                print(f"结果: {'✅ 成功' if result['success'] else '❌ 失败'} "
                      f"(分数: {result['score']}, 步数: {result['steps']})")

            except Exception as e:
                print(f"Episode运行出错: {e}")
                import traceback
                traceback.print_exc()
                all_results.append({
                    "success": False,
                    "score": 0,
                    "steps": 0,
                    "error": str(e),
                    "task_id": task_id,
                    "task_name": task_name,
                    "variation_idx": var_idx
                })
                total_episodes += 1

        # 统计该任务的结果
        num_task_episodes = len(selected_variations)
        task_stats[task_id] = {
            "task_name": task_name,
            "topic": task_topic,
            "description": task_desc,
            "episodes": num_task_episodes,
            "successes": task_successes,
            "success_rate": task_successes / num_task_episodes if num_task_episodes > 0 else 0,
            "avg_score": task_score / num_task_episodes if num_task_episodes > 0 else 0,
            "avg_steps": task_steps / num_task_episodes if num_task_episodes > 0 else 0
        }

        print(f"\n任务 {task_id} 统计: "
              f"成功率 {task_successes}/{num_task_episodes} ({task_stats[task_id]['success_rate']*100:.1f}%), "
              f"平均分数 {task_stats[task_id]['avg_score']:.1f}")

    # 关闭环境
    env.close()

    # 总体统计
    print(f"\n{'='*60}")
    print(f"测试结果汇总")
    print(f"{'='*60}")
    print(f"模型: {model}")
    print(f"总Episode数: {total_episodes}")
    print(f"成功数: {total_successes}")
    print(
        f"成功率: {total_successes/total_episodes*100:.1f}%" if total_episodes > 0 else "N/A")
    print(
        f"平均分数: {total_score/total_episodes:.1f}" if total_episodes > 0 else "N/A")
    print(
        f"平均步数: {total_steps/total_episodes:.1f}" if total_episodes > 0 else "N/A")
    print(f"{'='*60}")

    # 按任务统计
    print(f"\n分任务统计:")
    print("-"*60)
    for tid, stats in task_stats.items():
        print(f"{tid} ({stats['description']}): "
              f"成功率 {stats['success_rate']*100:.1f}%, "
              f"平均分 {stats['avg_score']:.1f}")

    # 保存结果
    if output_file:
        summary = {
            "model": model,
            "timestamp": datetime.now().isoformat(),
            "config": {
                "num_episodes": num_episodes,
                "task_ids": task_ids,
                "simplifications": simplifications,
                "max_steps": max_steps,
                "use_few_shot": use_few_shot,
                "temperature": TEMPERATURE,
                "seed": seed,
                "split": split
            },
            "summary": {
                "total_episodes": total_episodes,
                "successes": total_successes,
                "success_rate": total_successes / total_episodes if total_episodes > 0 else 0,
                "avg_score": total_score / total_episodes if total_episodes > 0 else 0,
                "avg_steps": total_steps / total_episodes if total_episodes > 0 else 0
            },
            "by_task": task_stats,
            "results": all_results
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"\n结果已保存到: {output_file}")

    return all_results


def demo_single_task(
    model: str = DEFAULT_MODEL,
    task_id: str = "1-2",
    seed: int = DEFAULT_SEED,
    simplifications: str = DEFAULT_SIMPLIFICATIONS
):
    """运行单个任务的演示"""

    if task_id not in TASK_INFO:
        print(f"错误: 未知的任务ID '{task_id}'")
        print(f"可用的任务ID: {list(TASK_INFO.keys())}")
        return

    task_name = TASK_INFO[task_id]["name"]
    print(f"\n运行演示任务: {task_id} - {TASK_INFO[task_id]['desc']} ({task_name})")

    # 设置随机种子
    if seed is not None:
        random.seed(seed)

    # 初始化环境
    env = ScienceWorldEnv("", envStepLimit=MAX_STEPS + 10)

    # 加载任务
    env.load(task_name, 0, simplifications)

    # 获取一个变体
    variations = env.get_variations_dev()
    if not variations:
        variations = env.get_variations_train()

    if not variations:
        print("没有可用的变体")
        return

    var_idx = variations[0]
    env.load(task_name, var_idx, simplifications)

    # 运行
    result = run_episode(
        env, model, task_id, var_idx,
        use_few_shot=True,
        verbose=True,
        max_steps=MAX_STEPS
    )

    env.close()
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ScienceWorld LLM Agent 测试")
    parser.add_argument("--model", type=str,
                        default=DEFAULT_MODEL, help="LLM模型名称")
    parser.add_argument("--num_episodes", type=int,
                        default=NUM_EPISODES, help="每个任务测试的episode数量")
    parser.add_argument("--task_ids", type=str, nargs="+", default=None,
                        help="任务ID列表 (如 1-1 1-2 4-1)")
    parser.add_argument("--simplifications", type=str, default=DEFAULT_SIMPLIFICATIONS,
                        help="简化设置 (easy 或自定义)")
    parser.add_argument("--max_steps", type=int, default=MAX_STEPS,
                        help="每个episode的最大步数")
    parser.add_argument("--no_few_shot", action="store_true",
                        help="不使用few-shot示例")
    parser.add_argument("--quiet", action="store_true", help="减少输出")
    parser.add_argument("--output", type=str, default=None, help="结果输出文件")
    parser.add_argument("--demo", action="store_true", help="运行单个任务演示")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="随机种子 (用于可复现的结果)")
    parser.add_argument("--no_seed", action="store_true",
                        help="不使用固定种子 (完全随机)")
    parser.add_argument("--split", type=str, default="dev",
                        choices=["train", "dev", "test"],
                        help="使用的数据集划分")

    args = parser.parse_args()

    # 处理种子参数
    seed = None if args.no_seed else args.seed

    if args.demo:
        # 演示模式 - 只运行一个任务
        demo_task_id = args.task_ids[0] if args.task_ids else "1-2"
        demo_single_task(
            model=args.model,
            task_id=demo_task_id,
            seed=seed,
            simplifications=args.simplifications
        )
    else:
        # 完整测试模式
        # 默认输出文件名
        if args.output is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = args.model.replace("/", "_")
            args.output = f"/home/bmt/evo/bench/scienceworld_results_{model_name}_{timestamp}.json"

        run_benchmark(
            model=args.model,
            num_episodes=args.num_episodes,
            task_ids=args.task_ids,
            simplifications=args.simplifications,
            max_steps=args.max_steps,
            use_few_shot=not args.no_few_shot,
            verbose=not args.quiet,
            output_file=args.output,
            seed=seed,
            split=args.split
        )


# ============= 使用示例 =============
# 🎮 运行单个任务演示 (融化任务)
# python scienceworld_test.py --demo --task_ids 1-2 --model "qwen/qwen-2.5-7b-instruct"

# 🎮 运行分类任务演示
# python scienceworld_test.py --demo --task_ids 4-1 --model "qwen/qwen3-8b"

# 📊 测试所有任务 (每个任务3个episode)
# python scienceworld_test.py --model "qwen/qwen3-8b" --num_episodes 3

# 📊 测试特定任务
# python scienceworld_test.py --model "qwen/qwen3-8b" --task_ids 1-1 1-2 4-1 4-2 --num_episodes 5

# 📊 只测试物态变化任务
# python scienceworld_test.py --model "qwen/qwen3-8b" --task_ids 1-1 1-2 1-3 1-4 --num_episodes 3

# 🔇 安静模式
# python scienceworld_test.py --model "qwen/qwen3-8b" --num_episodes 2 --quiet

# 📝 指定输出文件
# python scienceworld_test.py --model "qwen/qwen3-8b" --output my_results.json
