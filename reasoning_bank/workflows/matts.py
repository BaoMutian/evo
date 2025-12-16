"""
MaTTS 模块（Memory-aware Test-Time Scaling）

实现并行扩展（Self-Contrast）和串行扩展（Self-Refine）
"""

import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from reasoning_bank.core.llm_service import LLMService, get_llm_service
from reasoning_bank.core.memory import MemoryBank
from reasoning_bank.core.agent import ReActAgent, AgentConfig, AgentResult
from reasoning_bank.core.extractor import MemoryExtractor
from reasoning_bank.envs.base import BaseEnv
from reasoning_bank.prompts.registry import PromptRegistry
from reasoning_bank.utils.logger import get_logger
from reasoning_bank.utils.answer_parser import parse_react_response

logger = get_logger("matts")


@dataclass
class MaTTSConfig:
    """MaTTS 配置"""
    # 并行扩展配置
    parallel_n: int = 5  # 并行轨迹数量
    parallel_temperature: float = 0.7  # 并行扩展温度
    
    # 串行扩展配置
    sequential_max_refine: int = 3  # 最大修正次数
    sequential_check_temperature: float = 0.3  # 检查温度


class MaTTSParallel:
    """MaTTS 并行扩展模块
    
    对同一问题生成多条轨迹，通过对比提取高质量记忆
    """
    
    def __init__(
        self,
        llm_service: Optional[LLMService] = None,
        memory_bank: Optional[MemoryBank] = None,
        config: Optional[MaTTSConfig] = None,
    ):
        self.llm = llm_service or get_llm_service()
        self.memory_bank = memory_bank
        self.config = config or MaTTSConfig()
        
        self.extractor = MemoryExtractor(llm_service=self.llm)
    
    def run(
        self,
        env: BaseEnv,
        task_id: Optional[str] = None,
    ) -> Tuple[AgentResult, List[Dict[str, str]]]:
        """运行并行扩展
        
        Args:
            env: 环境实例
            task_id: 任务 ID
            
        Returns:
            (best_result, extracted_memories)
        """
        # 获取任务信息
        observation = env.reset(task_id)
        task_info = env.get_task_info()
        
        logger.info(f"MaTTS Parallel: 任务 {task_info.task_id}, 生成 {self.config.parallel_n} 条轨迹")
        
        # 生成多条轨迹
        results: List[AgentResult] = []
        
        for i in range(self.config.parallel_n):
            # 重置环境
            env.reset(task_info.task_id)
            
            # 创建高温度 agent
            agent_config = AgentConfig(
                temperature=self.config.parallel_temperature,
                verbose=False,
            )
            agent = ReActAgent(
                llm_service=self.llm,
                memory_bank=self.memory_bank,
                config=agent_config,
            )
            
            # 运行
            result = agent.run(env, task_info.task_id)
            results.append(result)
            
            logger.debug(f"  轨迹 {i+1}: {'SUCCESS' if result.is_success else 'FAILURE'}")
        
        # 统计
        success_count = sum(1 for r in results if r.is_success)
        logger.info(f"MaTTS Parallel: {success_count}/{self.config.parallel_n} 成功")
        
        # 选择最佳结果
        best_result = self._select_best(results)
        
        # 对比提取记忆
        memories = self.extractor.extract_contrastive(
            query=task_info.query,
            results=results,
        )
        
        return best_result, memories
    
    def _select_best(self, results: List[AgentResult]) -> AgentResult:
        """选择最佳结果
        
        策略：优先选择成功的，其中选择步数最少的
        """
        successful = [r for r in results if r.is_success]
        
        if successful:
            # 选择步数最少的成功结果
            return min(successful, key=lambda r: r.steps)
        else:
            # 都失败时，返回第一个
            return results[0]
    
    async def run_async(
        self,
        env: BaseEnv,
        task_id: Optional[str] = None,
    ) -> Tuple[AgentResult, List[Dict[str, str]]]:
        """异步并行运行（需要环境支持并发）
        
        注意：当前大多数环境不支持真正的并发，
        此方法主要用于 LLM 调用的并发
        """
        # 当前实现与同步版本相同
        # 未来可以优化为真正的并发执行
        return self.run(env, task_id)


class MaTTSSequential:
    """MaTTS 串行扩展模块
    
    生成轨迹后进行自我检查和修正
    """
    
    def __init__(
        self,
        llm_service: Optional[LLMService] = None,
        memory_bank: Optional[MemoryBank] = None,
        config: Optional[MaTTSConfig] = None,
    ):
        self.llm = llm_service or get_llm_service()
        self.memory_bank = memory_bank
        self.config = config or MaTTSConfig()
        
        self.extractor = MemoryExtractor(llm_service=self.llm)
    
    def run(
        self,
        env: BaseEnv,
        task_id: Optional[str] = None,
    ) -> Tuple[AgentResult, List[Dict[str, str]]]:
        """运行串行扩展（自我修正）
        
        Args:
            env: 环境实例
            task_id: 任务 ID
            
        Returns:
            (final_result, extracted_memories)
        """
        # 初始运行
        observation = env.reset(task_id)
        task_info = env.get_task_info()
        
        logger.info(f"MaTTS Sequential: 任务 {task_info.task_id}")
        
        # 第一次尝试
        agent = ReActAgent(
            llm_service=self.llm,
            memory_bank=self.memory_bank,
            config=AgentConfig(verbose=False),
        )
        
        initial_result = agent.run(env, task_info.task_id)
        
        # 如果成功，不需要修正
        if initial_result.is_success:
            logger.info("MaTTS Sequential: 初次尝试成功，无需修正")
            memories = self.extractor.extract(initial_result, env.get_ground_truth())
            return initial_result, memories
        
        # 进入检查-修正循环
        current_result = initial_result
        correction_made = False
        
        for refine_step in range(self.config.sequential_max_refine):
            logger.info(f"MaTTS Sequential: 修正尝试 {refine_step + 1}")
            
            # 自我检查
            check_result = self._self_check(
                task_info.query,
                current_result,
            )
            
            if check_result["no_issues"]:
                logger.info("MaTTS Sequential: 检查未发现问题")
                break
            
            # 重新尝试
            env.reset(task_info.task_id)
            
            # 将检查结果注入到 prompt 中
            agent_with_hint = ReActAgent(
                llm_service=self.llm,
                memory_bank=self.memory_bank,
                config=AgentConfig(
                    verbose=False,
                    temperature=self.config.sequential_check_temperature,
                ),
            )
            
            # 这里可以将 check_result 的 issues 作为额外提示
            # 简化实现：直接重新运行
            new_result = agent_with_hint.run(env, task_info.task_id)
            
            if new_result.is_success:
                logger.info("MaTTS Sequential: 修正后成功!")
                correction_made = True
                
                # 从修正过程中提取记忆
                memories = self.extractor.extract_from_refinement(
                    query=task_info.query,
                    initial_result=initial_result,
                    corrected_result=new_result,
                    correction_details=check_result.get("issues", ""),
                )
                
                return new_result, memories
            
            current_result = new_result
        
        # 所有修正尝试都失败
        logger.info("MaTTS Sequential: 修正尝试均失败")
        memories = self.extractor.extract(current_result, env.get_ground_truth())
        
        return current_result, memories
    
    def _self_check(
        self,
        query: str,
        result: AgentResult,
    ) -> Dict[str, Any]:
        """自我检查
        
        Args:
            query: 原始问题
            result: 当前结果
            
        Returns:
            检查结果 dict
        """
        # 格式化轨迹
        trajectory_text = self._format_trajectory(result)
        
        # 构建检查提示词
        prompt = PromptRegistry.get_matts_check_prompt(
            query=query,
            trajectory=trajectory_text,
        )
        
        # 调用 LLM 检查
        response = self.llm.call(
            prompt=prompt,
            temperature=self.config.sequential_check_temperature,
            max_tokens=1500,
        )
        
        if response.status != "success":
            return {"no_issues": True, "issues": ""}
        
        content = response.content
        
        # 解析检查结果
        no_issues = "none" in content.lower() and "issues_found" in content.lower()
        
        # 提取问题描述
        import re
        issues_match = re.search(r'ISSUES_FOUND:\s*(.+?)(?:\n|CORRECTED)', content, re.DOTALL)
        issues = issues_match.group(1).strip() if issues_match else ""
        
        if "none" in issues.lower():
            no_issues = True
            issues = ""
        
        return {
            "no_issues": no_issues,
            "issues": issues,
            "raw_response": content,
        }
    
    def _format_trajectory(self, result: AgentResult) -> str:
        """格式化轨迹"""
        lines = []
        for step in result.trajectory:
            lines.append(f"Step {step.get('step', '?')}:")
            if step.get('thought'):
                lines.append(f"  Thought: {step['thought']}")
            if step.get('action'):
                lines.append(f"  Action: {step['action']}")
        lines.append(f"\nFinal Answer: {result.answer}")
        return "\n".join(lines)


class MaTTSRunner:
    """MaTTS 统一运行器"""
    
    def __init__(
        self,
        env: BaseEnv,
        memory_bank: Optional[MemoryBank] = None,
        llm_service: Optional[LLMService] = None,
        config: Optional[MaTTSConfig] = None,
    ):
        self.env = env
        self.memory_bank = memory_bank
        self.llm = llm_service or get_llm_service()
        self.config = config or MaTTSConfig()
        
        self.parallel = MaTTSParallel(
            llm_service=self.llm,
            memory_bank=memory_bank,
            config=self.config,
        )
        
        self.sequential = MaTTSSequential(
            llm_service=self.llm,
            memory_bank=memory_bank,
            config=self.config,
        )
    
    def run_parallel(
        self,
        task_id: Optional[str] = None,
    ) -> Tuple[AgentResult, List[Dict[str, str]]]:
        """运行并行扩展"""
        return self.parallel.run(self.env, task_id)
    
    def run_sequential(
        self,
        task_id: Optional[str] = None,
    ) -> Tuple[AgentResult, List[Dict[str, str]]]:
        """运行串行扩展"""
        return self.sequential.run(self.env, task_id)
    
    def run_combined(
        self,
        task_id: Optional[str] = None,
    ) -> Tuple[AgentResult, List[Dict[str, str]]]:
        """组合运行：先并行后串行
        
        1. 并行生成多条轨迹
        2. 选择最佳轨迹
        3. 如果失败，进行串行修正
        4. 合并提取的记忆
        """
        # 并行扩展
        parallel_result, parallel_memories = self.parallel.run(self.env, task_id)
        
        if parallel_result.is_success:
            # 并行成功，直接返回
            return parallel_result, parallel_memories
        
        # 并行失败，进行串行修正
        logger.info("MaTTS Combined: 并行未成功，进入串行修正")
        
        sequential_result, sequential_memories = self.sequential.run(self.env, task_id)
        
        # 合并记忆
        combined_memories = parallel_memories + sequential_memories
        
        return sequential_result, combined_memories

