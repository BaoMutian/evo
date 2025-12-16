"""
记忆提取器模块

负责从任务轨迹中提取可复用的推理经验
"""

import json
import re
from typing import List, Dict, Any, Optional, Tuple

from reasoning_bank.core.llm_service import LLMService, get_llm_service
from reasoning_bank.core.agent import AgentResult
from reasoning_bank.prompts.registry import PromptRegistry
from reasoning_bank.utils.logger import get_logger

logger = get_logger("extractor")


class MemoryExtractor:
    """记忆提取器"""
    
    def __init__(
        self,
        llm_service: Optional[LLMService] = None,
        use_llm_judge: bool = False,  # 是否使用 LLM 判题
    ):
        """初始化提取器
        
        Args:
            llm_service: LLM 服务实例
            use_llm_judge: 是否使用 LLM 作为评判器
        """
        self.llm = llm_service or get_llm_service()
        self.use_llm_judge = use_llm_judge
    
    def judge(
        self,
        query: str,
        response: str,
        ground_truth: str = "",
        oracle_result: Optional[bool] = None,
    ) -> Tuple[bool, str]:
        """判断任务是否成功
        
        Args:
            query: 原始问题
            response: 智能体响应
            ground_truth: 标准答案
            oracle_result: 已知的评判结果（来自环境）
            
        Returns:
            (is_success, reason)
        """
        # 如果有 Oracle 结果，直接使用
        if oracle_result is not None:
            reason = "Evaluated by environment" if oracle_result else "Failed by environment evaluation"
            return oracle_result, reason
        
        # 如果有标准答案但不使用 LLM Judge，进行简单比较
        if ground_truth and not self.use_llm_judge:
            # 简单的字符串匹配
            response_clean = response.strip().lower()
            gt_clean = ground_truth.strip().lower()
            
            is_match = response_clean == gt_clean or gt_clean in response_clean
            reason = "Exact match" if is_match else "No match"
            return is_match, reason
        
        # 使用 LLM 判题
        prompt = PromptRegistry.get_judge_prompt(query, response, ground_truth)
        
        result = self.llm.call(
            prompt=prompt,
            temperature=0.1,  # 低温度保证一致性
            max_tokens=500,
        )
        
        if result.status != "success":
            logger.warning(f"LLM Judge 调用失败: {result.error}")
            return False, "Judge failed"
        
        # 解析判断结果
        response_text = result.content
        
        # 查找 JUDGMENT
        match = re.search(r'JUDGMENT:\s*(SUCCESS|FAILURE)', response_text, re.IGNORECASE)
        if match:
            is_success = match.group(1).upper() == "SUCCESS"
        else:
            is_success = "success" in response_text.lower()
        
        # 查找 REASON
        reason_match = re.search(r'REASON:\s*(.+?)(?:\n|$)', response_text, re.IGNORECASE)
        reason = reason_match.group(1).strip() if reason_match else "Unknown"
        
        return is_success, reason
    
    def extract(
        self,
        agent_result: AgentResult,
        ground_truth: str = "",
    ) -> List[Dict[str, str]]:
        """从智能体结果中提取记忆
        
        Args:
            agent_result: 智能体执行结果
            ground_truth: 标准答案
            
        Returns:
            记忆项列表 [{title, description, content}]
        """
        # 格式化轨迹
        trajectory_text = self._format_trajectory(agent_result)
        
        # 选择提取提示词
        prompt = PromptRegistry.get_extraction_prompt(
            is_success=agent_result.is_success,
            query=agent_result.query,
            trajectory=trajectory_text,
            ground_truth=ground_truth,
        )
        
        # 调用 LLM 提取
        result = self.llm.call(
            prompt=prompt,
            temperature=0.3,
            max_tokens=1500,
        )
        
        if result.status != "success":
            logger.error(f"记忆提取失败: {result.error}")
            return []
        
        # 解析 JSON 输出
        items = self._parse_extraction_result(result.content)
        
        logger.info(f"从任务 {agent_result.task_id} 提取了 {len(items)} 条记忆")
        
        return items
    
    def extract_contrastive(
        self,
        query: str,
        results: List[AgentResult],
    ) -> List[Dict[str, str]]:
        """对比多个轨迹提取记忆（MaTTS 并行模式）
        
        Args:
            query: 原始问题
            results: 多个智能体结果
            
        Returns:
            记忆项列表
        """
        # 格式化所有轨迹
        trajectories_text = []
        for i, r in enumerate(results):
            status = "SUCCESS" if r.is_success else "FAILURE"
            traj = self._format_trajectory(r)
            trajectories_text.append(f"=== Attempt {i+1} ({status}) ===\n{traj}")
        
        combined_trajectories = "\n\n".join(trajectories_text)
        
        # 构建对比提取提示词
        prompt = PromptRegistry.get_matts_contrast_prompt(
            query=query,
            trajectories=combined_trajectories,
        )
        
        # 调用 LLM
        result = self.llm.call(
            prompt=prompt,
            temperature=0.3,
            max_tokens=1500,
        )
        
        if result.status != "success":
            logger.error(f"对比提取失败: {result.error}")
            return []
        
        items = self._parse_extraction_result(result.content)
        
        logger.info(f"对比提取了 {len(items)} 条记忆")
        
        return items
    
    def extract_from_refinement(
        self,
        query: str,
        initial_result: AgentResult,
        corrected_result: AgentResult,
        correction_details: str = "",
    ) -> List[Dict[str, str]]:
        """从自我修正过程中提取记忆（MaTTS 串行模式）
        
        Args:
            query: 原始问题
            initial_result: 初始结果
            corrected_result: 修正后结果
            correction_details: 修正细节
            
        Returns:
            记忆项列表
        """
        initial_traj = self._format_trajectory(initial_result)
        corrected_traj = self._format_trajectory(corrected_result)
        
        prompt = PromptRegistry.MATTS_REFINEMENT_EXTRACT_PROMPT.format(
            query=query,
            initial_trajectory=initial_traj,
            corrected_trajectory=corrected_traj,
            correction_details=correction_details or "Self-correction was applied",
        )
        
        result = self.llm.call(
            prompt=prompt,
            temperature=0.3,
            max_tokens=1500,
        )
        
        if result.status != "success":
            logger.error(f"修正提取失败: {result.error}")
            return []
        
        items = self._parse_extraction_result(result.content)
        
        return items
    
    def _format_trajectory(self, result: AgentResult) -> str:
        """格式化轨迹为文本"""
        lines = []
        
        for step in result.trajectory:
            lines.append(f"Step {step.get('step', '?')}:")
            
            if step.get('observation'):
                obs = step['observation']
                if len(obs) > 300:
                    obs = obs[:300] + "..."
                lines.append(f"  Observation: {obs}")
            
            if step.get('thought'):
                lines.append(f"  Thought: {step['thought']}")
            
            if step.get('action'):
                lines.append(f"  Action: {step['action']}")
            
            lines.append("")
        
        # 添加最终答案
        lines.append(f"Final Answer: {result.answer}")
        lines.append(f"Result: {'SUCCESS' if result.is_success else 'FAILURE'}")
        
        return "\n".join(lines)
    
    def _parse_extraction_result(self, response: str) -> List[Dict[str, str]]:
        """解析 LLM 提取结果
        
        Args:
            response: LLM 响应文本
            
        Returns:
            解析后的记忆项列表
        """
        # 尝试提取 JSON
        try:
            # 查找 JSON 代码块
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response)
            if json_match:
                json_str = json_match.group(1)
            else:
                # 尝试查找 JSON 数组
                json_match = re.search(r'\[\s*\{[\s\S]*\}\s*\]', response)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    logger.warning("未找到 JSON 格式的提取结果")
                    return []
            
            items = json.loads(json_str)
            
            # 验证格式
            valid_items = []
            for item in items:
                if isinstance(item, dict) and "title" in item and "content" in item:
                    valid_items.append({
                        "title": item.get("title", ""),
                        "description": item.get("description", ""),
                        "content": item.get("content", ""),
                    })
            
            return valid_items
            
        except json.JSONDecodeError as e:
            logger.warning(f"JSON 解析失败: {e}")
            return []
        except Exception as e:
            logger.error(f"解析提取结果时出错: {e}")
            return []

