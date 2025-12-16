"""
Prompt 模板注册表

集中管理所有 Prompt 模板，支持动态注入记忆
"""

from typing import Dict, Optional, List
from string import Template


class PromptRegistry:
    """Prompt 模板注册表"""
    
    # ==================== Agent System Prompts ====================
    
    SYSTEM_BASE = """You are a highly capable reasoning agent. Your goal is to solve the given task step by step.

When answering, think carefully and show your reasoning process. For math problems, show your calculations. For multiple choice questions, explain why each option is correct or incorrect before giving your final answer.

{memory_block}

Output your final answer clearly. For math problems, put your answer in \\boxed{{}}. For multiple choice, clearly state the letter of your chosen answer."""

    SYSTEM_REACT = """You are a highly capable reasoning agent operating in a step-by-step manner.

For each step, you must output in the following format:
THOUGHT: [Your reasoning about the current situation and what to do next]
ACTION: [The action to take or answer to give]

{memory_block}

Think carefully at each step. When you have the final answer, output it clearly in the ACTION field."""

    SYSTEM_SINGLE_TURN = """You are an expert problem solver. Analyze the given problem carefully and provide a clear solution.

{memory_block}

Guidelines:
1. Read the problem carefully and identify what is being asked
2. Break down complex problems into smaller steps
3. Show your reasoning clearly
4. For math problems: put your final answer in \\boxed{{}}
5. For multiple choice: clearly state the letter (A, B, C, etc.) of your answer

Now solve the following problem:"""

    # ==================== Memory Block Templates ====================
    
    MEMORY_BLOCK_TEMPLATE = """
=== RELEVANT PAST EXPERIENCES ===
The following are insights from similar tasks you've encountered before. Use them as guidance, but verify their applicability to the current task.

{memories}

Note: Apply these experiences judiciously. If they seem irrelevant to the current task, you may disregard them.
=== END OF EXPERIENCES ===
"""

    MEMORY_BLOCK_EMPTY = ""  # 无记忆时为空

    # ==================== Judge Prompts ====================
    
    JUDGE_PROMPT = """You are a strict judge evaluating whether a task was completed successfully.

Task/Question:
{query}

Agent's Response:
{response}

Ground Truth (if available):
{ground_truth}

Evaluate whether the agent's response correctly answers the question/completes the task.
Consider:
1. Is the final answer correct?
2. Is the reasoning sound (if provided)?
3. For math: Is the numerical answer equivalent to the ground truth?
4. For multiple choice: Does the selected option match?

Output your judgment in the following format:
JUDGMENT: [SUCCESS or FAILURE]
REASON: [Brief explanation of your judgment]"""

    # ==================== Extraction Prompts ====================
    
    EXTRACT_SUCCESS_PROMPT = """You are analyzing a successful problem-solving trajectory to extract reusable strategies.

Original Problem:
{query}

Successful Solution Trajectory:
{trajectory}

Your task is to extract generalizable insights that could help solve similar problems in the future. Focus on:
1. Key problem-solving strategies used
2. Important patterns recognized
3. Effective reasoning techniques
4. Any tricks or shortcuts that worked

Generate 1-3 memory items in the following JSON format:
```json
[
  {{
    "title": "Brief strategy name (e.g., 'Quadratic Equation Pattern Recognition')",
    "description": "One sentence summary",
    "content": "Detailed actionable advice without problem-specific details"
  }}
]
```

Make the advice general enough to apply to similar problems, not tied to specific numbers or values in this problem."""

    EXTRACT_FAILURE_PROMPT = """You are analyzing a failed problem-solving attempt to extract lessons for avoiding similar mistakes.

Original Problem:
{query}

Failed Attempt:
{trajectory}

Correct Answer (for reference):
{ground_truth}

Your task is to extract preventive insights by analyzing:
1. What went wrong in the reasoning?
2. What misconception or error led to the failure?
3. What should be done differently next time?
4. What warning signs should trigger reconsideration?

Generate 1-3 memory items in the following JSON format:
```json
[
  {{
    "title": "Brief lesson name (e.g., 'Avoid Sign Errors in Quadratics')",
    "description": "One sentence describing the pitfall",
    "content": "Detailed preventive advice: what to watch out for and how to avoid this mistake"
  }}
]
```

Focus on generalizable lessons that can prevent similar failures in related problems."""

    # ==================== MaTTS Prompts ====================
    
    MATTS_PARALLEL_CONTRAST_PROMPT = """You are comparing multiple solution attempts to the same problem to identify what distinguishes successful from unsuccessful approaches.

Problem:
{query}

Solution Attempts:
{trajectories}

Analyze the differences between successful and failed attempts:
1. What strategies did successful attempts use that failed ones didn't?
2. What common mistakes appear in failed attempts?
3. What's the key insight that separates success from failure?

Extract the most valuable insights in JSON format:
```json
[
  {{
    "title": "Strategy name",
    "description": "One sentence summary",
    "content": "Detailed advice based on the contrast analysis"
  }}
]
```"""

    MATTS_SEQUENTIAL_CHECK_PROMPT = """Review your previous solution attempt carefully.

Problem:
{query}

Your Previous Attempt:
{trajectory}

Please:
1. Re-examine each step of your reasoning
2. Check for calculation errors, logical mistakes, or missed cases
3. Verify that your final answer actually addresses what was asked
4. If you find any issues, provide a corrected solution

Output format:
REVIEW: [Your analysis of the previous attempt]
ISSUES_FOUND: [List any problems discovered, or "None" if the solution is correct]
CORRECTED_ANSWER: [Your final answer after review, or same as before if no issues]"""

    MATTS_REFINEMENT_EXTRACT_PROMPT = """You are analyzing a self-correction process where an agent caught and fixed its own mistake.

Problem:
{query}

Initial (Incorrect) Attempt:
{initial_trajectory}

Corrected Solution:
{corrected_trajectory}

What was corrected:
{correction_details}

Extract insights from this self-correction process:
1. What was the original error?
2. How was it detected?
3. What should be done to avoid this error initially?

Generate memory items in JSON format:
```json
[
  {{
    "title": "Self-correction lesson",
    "description": "One sentence about the type of error",
    "content": "How to detect and avoid this type of error in the future"
  }}
]
```"""

    # ==================== Multi-turn Environment Prompts ====================
    
    SYSTEM_ALFWORLD = """You are an intelligent agent operating in a text-based household environment. Your goal is to complete household tasks by interacting with objects in the environment.

Available actions:
- look: look around your current location
- inventory: check what you're carrying
- go to [receptacle]: move to a receptacle (e.g., "go to dresser 1", "go to fridge 1")
- open [receptacle]: open a receptacle (e.g., "open drawer 1")
- close [receptacle]: close a receptacle
- take [object] from [receptacle]: pick up an object
- move [object] to [receptacle]: put an object down
- examine [something]: examine an object or receptacle
- use [object]: turn on/off an object (e.g., "use desklamp 1")
- heat [object] with [receptacle]: heat an object using microwave
- clean [object] with [receptacle]: clean an object using sink
- cool [object] with [receptacle]: cool an object using fridge

{memory_block}

You must respond in the following format:
THOUGHT: [Your reasoning about what to do next]
ACTION: [The exact action to take]

Only output one action at a time."""

    SYSTEM_SCIENCEWORLD = """You are an intelligent agent operating in a text-based science simulation environment. Your goal is to complete various science experiment tasks.

Available action types:
- look around: observe your current location
- go to [location]: move to a specific location
- teleport to [location]: instantly move to any location (if enabled)
- open/close [container]: open or close containers
- pick up [object]: pick up an object
- put down [object]: put down an object you're carrying
- move [object] to [location]: move an object somewhere
- activate/deactivate [device]: turn devices on/off
- use [object] on [target]: use an object on something
- pour [substance] into [container]: pour liquid
- focus on [object]: focus attention on an object
- connect [obj1] to [obj2]: connect objects (for circuits)
- wait: wait for time to pass
- read [object]: read something

{memory_block}

You must respond in the following format:
THOUGHT: [Your reasoning about what to do next]
ACTION: [The exact action to take]

Important: Use "wait" when processes need time (heating, cooling, growing)."""

    @classmethod
    def get_system_prompt(
        cls,
        prompt_type: str = "base",
        memories: Optional[List] = None,
        memory_formatter: callable = None,
    ) -> str:
        """获取系统提示词
        
        Args:
            prompt_type: 提示词类型 (base, react, single_turn, alfworld, scienceworld)
            memories: 记忆列表
            memory_formatter: 记忆格式化函数
            
        Returns:
            格式化后的系统提示词
        """
        # 选择基础模板
        template_map = {
            "base": cls.SYSTEM_BASE,
            "react": cls.SYSTEM_REACT,
            "single_turn": cls.SYSTEM_SINGLE_TURN,
            "alfworld": cls.SYSTEM_ALFWORLD,
            "scienceworld": cls.SYSTEM_SCIENCEWORLD,
        }
        
        template = template_map.get(prompt_type, cls.SYSTEM_BASE)
        
        # 格式化记忆块
        if memories and memory_formatter:
            memory_text = memory_formatter(memories)
            memory_block = cls.MEMORY_BLOCK_TEMPLATE.format(memories=memory_text)
        else:
            memory_block = cls.MEMORY_BLOCK_EMPTY
        
        return template.format(memory_block=memory_block)
    
    @classmethod
    def get_extraction_prompt(
        cls,
        is_success: bool,
        query: str,
        trajectory: str,
        ground_truth: str = "",
    ) -> str:
        """获取记忆提取提示词
        
        Args:
            is_success: 是否成功
            query: 原始问题
            trajectory: 解题轨迹
            ground_truth: 标准答案
            
        Returns:
            提取提示词
        """
        if is_success:
            return cls.EXTRACT_SUCCESS_PROMPT.format(
                query=query,
                trajectory=trajectory,
            )
        else:
            return cls.EXTRACT_FAILURE_PROMPT.format(
                query=query,
                trajectory=trajectory,
                ground_truth=ground_truth,
            )
    
    @classmethod
    def get_judge_prompt(
        cls,
        query: str,
        response: str,
        ground_truth: str = "",
    ) -> str:
        """获取评判提示词"""
        return cls.JUDGE_PROMPT.format(
            query=query,
            response=response,
            ground_truth=ground_truth if ground_truth else "Not provided",
        )
    
    @classmethod
    def get_matts_contrast_prompt(
        cls,
        query: str,
        trajectories: str,
    ) -> str:
        """获取 MaTTS 对比提取提示词"""
        return cls.MATTS_PARALLEL_CONTRAST_PROMPT.format(
            query=query,
            trajectories=trajectories,
        )
    
    @classmethod
    def get_matts_check_prompt(
        cls,
        query: str,
        trajectory: str,
    ) -> str:
        """获取 MaTTS 检查提示词"""
        return cls.MATTS_SEQUENTIAL_CHECK_PROMPT.format(
            query=query,
            trajectory=trajectory,
        )

