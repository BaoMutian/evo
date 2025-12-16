"""
LLM 服务模块

封装 OpenRouter API 调用，支持同步和异步调用
"""

import os
import time
import asyncio
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass

from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv

from reasoning_bank.utils.config import get_config
from reasoning_bank.utils.logger import get_logger

# 加载环境变量
load_dotenv()

logger = get_logger("llm_service")


@dataclass
class LLMResponse:
    """LLM 响应数据类"""
    status: str  # success, failed
    content: Optional[str] = None
    reasoning: Optional[str] = None  # 推理过程（部分模型支持）
    usage: Optional[Dict] = None
    time_taken: float = 0.0
    error: Optional[str] = None


class LLMService:
    """LLM 服务类，封装 OpenRouter API"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
        timeout: int = 120,
        max_retries: int = 3,
    ):
        """初始化 LLM 服务
        
        Args:
            api_key: API 密钥，默认从环境变量或配置读取
            api_base: API 基础地址
            model: 默认模型名称
            temperature: 温度参数
            max_tokens: 最大 token 数
            timeout: 请求超时时间
            max_retries: 最大重试次数
        """
        # 从配置或环境变量获取参数
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY") or get_config("llm.api_key")
        self.api_base = api_base or os.getenv("OPENROUTER_API_BASE") or get_config("llm.api_base", "https://openrouter.ai/api/v1")
        self.default_model = model or get_config("llm.default_model", "deepseek/deepseek-chat-v3-0324")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        
        if not self.api_key:
            raise ValueError("API Key 未设置，请设置环境变量 OPENROUTER_API_KEY 或在配置中指定")
        
        # 初始化客户端
        self._sync_client: Optional[OpenAI] = None
        self._async_client: Optional[AsyncOpenAI] = None
    
    @property
    def sync_client(self) -> OpenAI:
        """获取同步客户端"""
        if self._sync_client is None:
            self._sync_client = OpenAI(
                api_key=self.api_key,
                base_url=self.api_base,
                timeout=self.timeout,
            )
        return self._sync_client
    
    @property
    def async_client(self) -> AsyncOpenAI:
        """获取异步客户端"""
        if self._async_client is None:
            self._async_client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.api_base,
                timeout=self.timeout,
            )
        return self._async_client
    
    def _build_messages(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> List[Dict[str, str]]:
        """构建消息列表
        
        Args:
            prompt: 用户提示
            system_prompt: 系统提示
            history: 历史对话
            
        Returns:
            消息列表
        """
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        if history:
            messages.extend(history)
        
        messages.append({"role": "user", "content": prompt})
        
        return messages
    
    def call(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        history: Optional[List[Dict[str, str]]] = None,
        stream: bool = False,
    ) -> LLMResponse:
        """同步调用 LLM
        
        Args:
            prompt: 用户提示
            system_prompt: 系统提示
            model: 模型名称
            temperature: 温度参数
            max_tokens: 最大 token 数
            history: 历史对话
            stream: 是否流式输出
            
        Returns:
            LLMResponse 对象
        """
        model = model or self.default_model
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens or self.max_tokens
        
        messages = self._build_messages(prompt, system_prompt, history)
        
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                
                params = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "stream": stream,
                }
                
                if max_tokens and max_tokens > 0:
                    params["max_tokens"] = max_tokens
                
                completion = self.sync_client.chat.completions.create(**params)
                
                if stream:
                    # 流式处理
                    content = ""
                    reasoning = ""
                    for chunk in completion:
                        if hasattr(chunk.choices[0].delta, 'reasoning') and chunk.choices[0].delta.reasoning:
                            reasoning += chunk.choices[0].delta.reasoning
                        if chunk.choices[0].delta.content:
                            content += chunk.choices[0].delta.content
                    
                    return LLMResponse(
                        status="success",
                        content=content,
                        reasoning=reasoning if reasoning else None,
                        time_taken=round(time.time() - start_time, 2),
                    )
                else:
                    return LLMResponse(
                        status="success",
                        content=completion.choices[0].message.content,
                        reasoning=getattr(completion.choices[0].message, 'reasoning', None),
                        usage=completion.usage.model_dump() if completion.usage else None,
                        time_taken=round(time.time() - start_time, 2),
                    )
                    
            except Exception as e:
                logger.warning(f"LLM 调用失败 (尝试 {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # 指数退避
                else:
                    return LLMResponse(
                        status="failed",
                        error=str(e),
                    )
    
    async def acall(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> LLMResponse:
        """异步调用 LLM
        
        Args:
            prompt: 用户提示
            system_prompt: 系统提示
            model: 模型名称
            temperature: 温度参数
            max_tokens: 最大 token 数
            history: 历史对话
            
        Returns:
            LLMResponse 对象
        """
        model = model or self.default_model
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens or self.max_tokens
        
        messages = self._build_messages(prompt, system_prompt, history)
        
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                
                params = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                }
                
                if max_tokens and max_tokens > 0:
                    params["max_tokens"] = max_tokens
                
                completion = await self.async_client.chat.completions.create(**params)
                
                return LLMResponse(
                    status="success",
                    content=completion.choices[0].message.content,
                    reasoning=getattr(completion.choices[0].message, 'reasoning', None),
                    usage=completion.usage.model_dump() if completion.usage else None,
                    time_taken=round(time.time() - start_time, 2),
                )
                
            except Exception as e:
                logger.warning(f"LLM 异步调用失败 (尝试 {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    return LLMResponse(
                        status="failed",
                        error=str(e),
                    )
    
    async def batch_call(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_concurrency: int = 5,
    ) -> List[LLMResponse]:
        """批量异步调用 LLM
        
        Args:
            prompts: 提示列表
            system_prompt: 系统提示
            model: 模型名称
            temperature: 温度参数
            max_tokens: 最大 token 数
            max_concurrency: 最大并发数
            
        Returns:
            LLMResponse 列表
        """
        semaphore = asyncio.Semaphore(max_concurrency)
        
        async def limited_call(prompt: str) -> LLMResponse:
            async with semaphore:
                return await self.acall(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
        
        tasks = [limited_call(prompt) for prompt in prompts]
        return await asyncio.gather(*tasks)
    
    def call_with_retry(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """带重试的同步调用，直接返回内容字符串
        
        Args:
            prompt: 用户提示
            system_prompt: 系统提示
            model: 模型名称
            temperature: 温度参数
            max_tokens: 最大 token 数
            history: 历史对话
            
        Returns:
            响应内容字符串
            
        Raises:
            RuntimeError: 如果所有重试都失败
        """
        response = self.call(
            prompt=prompt,
            system_prompt=system_prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            history=history,
        )
        
        if response.status == "success":
            return response.content
        else:
            raise RuntimeError(f"LLM 调用失败: {response.error}")


# 全局单例
_llm_service: Optional[LLMService] = None


def get_llm_service(**kwargs) -> LLMService:
    """获取 LLM 服务单例
    
    Args:
        **kwargs: 传递给 LLMService 的参数
        
    Returns:
        LLMService 实例
    """
    global _llm_service
    
    if _llm_service is None:
        _llm_service = LLMService(**kwargs)
    
    return _llm_service

