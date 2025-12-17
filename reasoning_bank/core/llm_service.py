"""
LLM 服务模块

封装 OpenRouter API 调用，支持同步和异步调用
"""

import os
import re
import time
import asyncio
from typing import Optional, List, Dict, Any, Union, Tuple
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
        debug: bool = False,
        enable_thinking: Optional[bool] = None,
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
            debug: 是否开启调试模式（打印完整 prompt 和响应）
            enable_thinking: Qwen3 思考模式开关（None=不传递使用模型默认，True=开启，False=关闭）
        """
        # 从配置或环境变量获取参数
        self.api_key = api_key or os.getenv(
            "OPENROUTER_API_KEY") or get_config("llm.api_key")
        self.api_base = api_base or os.getenv("OPENROUTER_API_BASE") or get_config(
            "llm.api_base", "https://openrouter.ai/api/v1")
        self.default_model = model or get_config(
            "llm.default_model", "qwen/qwen-2.5-7b-instruct")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.debug = debug
        self.enable_thinking = enable_thinking

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

    def _strip_thinking_tags(self, content: str) -> Tuple[str, Optional[str]]:
        """从响应中提取并移除 <think> 标签内容
        
        Qwen3 模型在思考模式下会在 content 中返回 <think>...</think> 标签包裹的思考内容。
        此方法将思考内容提取出来，并返回清理后的内容。
        
        Args:
            content: 原始响应内容
            
        Returns:
            (cleaned_content, reasoning): 清理后的内容和提取的思考内容
        """
        if not content:
            return content, None
        
        # 匹配 <think>...</think> 标签（支持多行）
        think_pattern = re.compile(r'<think>(.*?)</think>', re.DOTALL)
        
        # 提取所有思考内容
        think_matches = think_pattern.findall(content)
        reasoning = '\n'.join(match.strip() for match in think_matches) if think_matches else None
        
        # 移除 <think> 标签及其内容
        cleaned_content = think_pattern.sub('', content).strip()
        
        return cleaned_content, reasoning

    def _debug_print_request(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: Optional[int],
    ):
        """打印调试信息：请求"""
        print("\n" + "=" * 80)
        print("🔵 [DEBUG] LLM REQUEST")
        print("=" * 80)
        print(f"📌 Model: {model}")
        print(f"🌡️  Temperature: {temperature}")
        print(f"📊 Max Tokens: {max_tokens}")
        print("-" * 80)
        for i, msg in enumerate(messages):
            role = msg["role"].upper()
            content = msg["content"]
            print(f"\n📝 [{role}] (Message {i+1})")
            print("-" * 40)
            print(content)
        print("\n" + "=" * 80 + "\n")

    def _debug_print_response(self, response: 'LLMResponse'):
        """打印调试信息：响应"""
        print("\n" + "=" * 80)
        print("🟢 [DEBUG] LLM RESPONSE")
        print("=" * 80)
        print(f"✅ Status: {response.status}")
        print(f"⏱️  Time: {response.time_taken}s")
        if response.usage:
            print(f"📊 Usage: {response.usage}")
        if response.reasoning:
            print("-" * 40)
            print("🧠 REASONING:")
            print(response.reasoning)
        print("-" * 40)
        print("💬 CONTENT:")
        print(response.content)
        print("\n" + "=" * 80 + "\n")

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

        # Debug: 打印完整 prompt
        if self.debug:
            self._debug_print_request(model, messages, temperature, max_tokens)

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

                # Qwen3 思考模式控制（通过 extra_body 传递给 vLLM/SGLang）
                if self.enable_thinking is not None:
                    params["extra_body"] = {
                        "enable_thinking": self.enable_thinking}

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

                    # 处理 Qwen3 思考模式的 <think> 标签
                    cleaned_content, extracted_reasoning = self._strip_thinking_tags(content)
                    # 优先使用 API 返回的 reasoning，其次使用从 <think> 标签提取的
                    final_reasoning = reasoning if reasoning else extracted_reasoning

                    response = LLMResponse(
                        status="success",
                        content=cleaned_content,
                        reasoning=final_reasoning if final_reasoning else None,
                        time_taken=round(time.time() - start_time, 2),
                    )
                else:
                    raw_content = completion.choices[0].message.content
                    api_reasoning = getattr(completion.choices[0].message, 'reasoning', None)
                    
                    # 处理 Qwen3 思考模式的 <think> 标签
                    cleaned_content, extracted_reasoning = self._strip_thinking_tags(raw_content)
                    # 优先使用 API 返回的 reasoning，其次使用从 <think> 标签提取的
                    final_reasoning = api_reasoning if api_reasoning else extracted_reasoning

                    response = LLMResponse(
                        status="success",
                        content=cleaned_content,
                        reasoning=final_reasoning,
                        usage=completion.usage.model_dump() if completion.usage else None,
                        time_taken=round(time.time() - start_time, 2),
                    )

                # Debug: 打印完整响应
                if self.debug:
                    self._debug_print_response(response)

                return response

            except Exception as e:
                logger.warning(
                    f"LLM 调用失败 (尝试 {attempt + 1}/{self.max_retries}): {e}")
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

        # Debug: 打印完整 prompt
        if self.debug:
            self._debug_print_request(model, messages, temperature, max_tokens)

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

                # Qwen3 思考模式控制
                if self.enable_thinking is not None:
                    params["extra_body"] = {
                        "enable_thinking": self.enable_thinking}

                completion = await self.async_client.chat.completions.create(**params)

                raw_content = completion.choices[0].message.content
                api_reasoning = getattr(completion.choices[0].message, 'reasoning', None)
                
                # 处理 Qwen3 思考模式的 <think> 标签
                cleaned_content, extracted_reasoning = self._strip_thinking_tags(raw_content)
                # 优先使用 API 返回的 reasoning，其次使用从 <think> 标签提取的
                final_reasoning = api_reasoning if api_reasoning else extracted_reasoning

                response = LLMResponse(
                    status="success",
                    content=cleaned_content,
                    reasoning=final_reasoning,
                    usage=completion.usage.model_dump() if completion.usage else None,
                    time_taken=round(time.time() - start_time, 2),
                )

                # Debug: 打印完整响应
                if self.debug:
                    self._debug_print_response(response)

                return response

            except Exception as e:
                logger.warning(
                    f"LLM 异步调用失败 (尝试 {attempt + 1}/{self.max_retries}): {e}")
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
    elif kwargs:
        # 如果已存在实例但传入了参数，更新部分属性
        for key, value in kwargs.items():
            if hasattr(_llm_service, key):
                setattr(_llm_service, key, value)

    return _llm_service


def set_debug_mode(enabled: bool = True):
    """设置全局 LLM 调试模式

    Args:
        enabled: 是否启用调试模式
    """
    service = get_llm_service()
    service.debug = enabled
    logger.info(f"LLM Debug 模式: {'启用' if enabled else '禁用'}")
