"""
智能体模块

实现 ReAct 风格的推理智能体
"""

from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field

from reasoning_bank.core.llm_service import LLMService, get_llm_service
from reasoning_bank.core.memory import MemoryBank, MemoryItem
from reasoning_bank.prompts.registry import PromptRegistry
from reasoning_bank.envs.base import BaseEnv, TaskType, StepResult
from reasoning_bank.utils.answer_parser import parse_react_response, extract_final_answer
from reasoning_bank.utils.logger import get_logger, TaskLogger

logger = get_logger("agent")


@dataclass
class AgentConfig:
    """智能体配置"""
    max_steps: int = 30  # 多轮交互最大步数
    use_react: bool = True  # 是否使用 ReAct 格式
    temperature: float = 0.3  # 生成温度
    verbose: bool = True  # 是否输出详细日志


@dataclass
class AgentResult:
    """智能体执行结果"""
    task_id: str
    query: str
    answer: str  # 最终答案
    is_success: bool
    trajectory: List[Dict[str, Any]]  # 完整轨迹
    steps: int  # 步数
    memories_used: List[MemoryItem] = field(default_factory=list)  # 使用的记忆
    raw_response: str = ""  # 原始响应


class ReActAgent:
    """ReAct 风格智能体"""
    
    def __init__(
        self,
        llm_service: Optional[LLMService] = None,
        memory_bank: Optional[MemoryBank] = None,
        config: Optional[AgentConfig] = None,
    ):
        """初始化智能体
        
        Args:
            llm_service: LLM 服务实例
            memory_bank: 记忆库实例
            config: 智能体配置
        """
        self.llm = llm_service or get_llm_service()
        self.memory_bank = memory_bank
        self.config = config or AgentConfig()
    
    def _build_system_prompt(
        self,
        task_type: TaskType,
        memories: Optional[List[MemoryItem]] = None,
    ) -> str:
        """构建系统提示词
        
        Args:
            task_type: 任务类型
            memories: 相关记忆
            
        Returns:
            系统提示词
        """
        # 选择提示词类型
        if task_type == TaskType.SINGLE_TURN:
            prompt_type = "single_turn"
        else:
            prompt_type = "react"
        
        # 记忆格式化器
        def memory_formatter(mems: List[MemoryItem]) -> str:
            if self.memory_bank:
                return self.memory_bank.format_memories_for_prompt(mems)
            return ""
        
        return PromptRegistry.get_system_prompt(
            prompt_type=prompt_type,
            memories=memories,
            memory_formatter=memory_formatter,
        )
    
    def run_single_turn(
        self,
        env: BaseEnv,
        task_id: Optional[str] = None,
    ) -> AgentResult:
        """运行单轮任务
        
        Args:
            env: 环境实例
            task_id: 任务 ID
            
        Returns:
            AgentResult
        """
        # 重置环境
        observation = env.reset(task_id)
        task_info = env.get_task_info()
        
        task_logger = TaskLogger(task_info.task_id)
        
        # 检索相关记忆
        memories = []
        if self.memory_bank:
            memories = self.memory_bank.retrieve(task_info.query)
            task_logger.log_memory_retrieval(task_info.query, 
                [m.to_dict() for m in memories])
        
        # 构建提示词
        system_prompt = self._build_system_prompt(TaskType.SINGLE_TURN, memories)
        
        # 调用 LLM
        response = self.llm.call(
            prompt=observation,
            system_prompt=system_prompt,
            temperature=self.config.temperature,
        )
        
        if response.status != "success":
            logger.error(f"LLM 调用失败: {response.error}")
            return AgentResult(
                task_id=task_info.task_id,
                query=task_info.query,
                answer="",
                is_success=False,
                trajectory=[],
                steps=0,
                memories_used=memories,
                raw_response="",
            )
        
        raw_response = response.content
        
        # 提取答案
        if task_info.options:  # MCQ
            from reasoning_bank.utils.answer_parser import extract_choice_answer
            answer = extract_choice_answer(raw_response) or raw_response
        else:  # Math
            answer = extract_final_answer(raw_response) or raw_response
        
        # 记录轨迹
        env.record_step(
            observation=observation,
            thought=raw_response[:500],  # 截断
            action=answer,
            result="",
        )
        
        # 评估
        step_result = env.step(answer)
        is_success = step_result.info.get("is_correct", False)
        
        task_logger.log_result(is_success, answer, task_info.ground_truth)
        
        return AgentResult(
            task_id=task_info.task_id,
            query=task_info.query,
            answer=answer,
            is_success=is_success,
            trajectory=env.get_trajectory(),
            steps=1,
            memories_used=memories,
            raw_response=raw_response,
        )
    
    def run_multi_turn(
        self,
        env: BaseEnv,
        task_id: Optional[str] = None,
    ) -> AgentResult:
        """运行多轮交互任务
        
        Args:
            env: 环境实例
            task_id: 任务 ID
            
        Returns:
            AgentResult
        """
        # 重置环境
        observation = env.reset(task_id)
        task_info = env.get_task_info()
        
        task_logger = TaskLogger(task_info.task_id)
        
        # 检索相关记忆
        memories = []
        if self.memory_bank:
            memories = self.memory_bank.retrieve(task_info.query)
            task_logger.log_memory_retrieval(task_info.query,
                [m.to_dict() for m in memories])
        
        # 构建系统提示词
        system_prompt = self._build_system_prompt(TaskType.MULTI_TURN, memories)
        
        # 对话历史
        history: List[Dict[str, str]] = []
        
        # 主循环
        done = False
        step = 0
        final_answer = ""
        is_success = False
        
        while not done and step < self.config.max_steps:
            step += 1
            
            # 构建用户消息
            if step == 1:
                user_message = f"Task: {task_info.query}\n\nObservation: {observation}"
            else:
                user_message = f"Observation: {observation}"
            
            # 调用 LLM
            response = self.llm.call(
                prompt=user_message,
                system_prompt=system_prompt,
                history=history,
                temperature=self.config.temperature,
            )
            
            if response.status != "success":
                logger.error(f"Step {step}: LLM 调用失败: {response.error}")
                break
            
            raw_response = response.content
            
            # 解析 ReAct 响应
            thought, action = parse_react_response(raw_response)
            
            if self.config.verbose:
                logger.info(f"Step {step} - Thought: {thought[:100]}...")
                logger.info(f"Step {step} - Action: {action}")
            
            # 记录轨迹
            env.record_step(
                observation=observation,
                thought=thought,
                action=action,
            )
            
            # 更新历史
            history.append({"role": "user", "content": user_message})
            history.append({"role": "assistant", "content": raw_response})
            
            # 执行动作
            step_result = env.step(action)
            observation = step_result.observation
            done = step_result.done
            
            # 检查是否成功
            if step_result.info.get("won", False) or step_result.reward >= 1.0:
                is_success = True
                final_answer = action
                break
            
            if done:
                final_answer = action
                is_success = step_result.info.get("is_correct", False)
        
        task_logger.log_result(is_success, final_answer, task_info.ground_truth)
        
        return AgentResult(
            task_id=task_info.task_id,
            query=task_info.query,
            answer=final_answer,
            is_success=is_success,
            trajectory=env.get_trajectory(),
            steps=step,
            memories_used=memories,
            raw_response="",
        )
    
    def run(
        self,
        env: BaseEnv,
        task_id: Optional[str] = None,
    ) -> AgentResult:
        """运行任务（自动选择单轮/多轮）
        
        Args:
            env: 环境实例
            task_id: 任务 ID
            
        Returns:
            AgentResult
        """
        if env.task_type == TaskType.SINGLE_TURN:
            return self.run_single_turn(env, task_id)
        else:
            return self.run_multi_turn(env, task_id)

