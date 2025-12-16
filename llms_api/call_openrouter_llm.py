from openai import OpenAI
import requests
import json
import time
import os

# 尝试加载 .env 文件中的环境变量（如果安装了 python-dotenv）
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # 如果没有安装 python-dotenv，则只从系统环境变量读取
    pass

# 从环境变量读取配置
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_API_BASE = os.getenv(
    "OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")

if not OPENROUTER_API_KEY:
    raise ValueError("请设置环境变量 OPENROUTER_API_KEY（可在 .env 文件中设置）")


def get_openrouter_key_info(api_key):
    """查询OpenRouter API Key信息"""
    response = requests.get(
        url=f"{OPENROUTER_API_BASE}/auth/key",
        headers={
            "Authorization": f"Bearer {api_key}"
        }
    )
    return response.json()


def get_openrouter_credits(api_key):
    """查询OpenRouter账户余额信息"""
    response = requests.get(
        url=f"{OPENROUTER_API_BASE}/credits",
        headers={
            "Authorization": f"Bearer {api_key}"
        }
    )
    data = response.json()
    if data and "data" in data and "total_credits" in data["data"] and "total_usage" in data["data"]:
        total_credits = data["data"]["total_credits"]
        total_usage = data["data"]["total_usage"]
        return round(total_credits - total_usage, 2)
    else:
        return None


def call_openrouter_llm(prompt_text, model="deepseek/deepseek-chat-v3-0324:free", stream=False, temperature=0.6, max_tokens=None, reason_output=False, system_prompt=None):
    """调用OpenRouter API

    Args:
        prompt_text: 用户提示文本
        model: 模型名称
        stream: 是否流式输出
        temperature: 温度参数
        max_tokens: 最大token数
        reason_output: 是否输出推理过程
        system_prompt: 系统提示词（可选）
    """
    try:
        client = OpenAI(
            base_url=OPENROUTER_API_BASE,
            api_key=OPENROUTER_API_KEY,
        )
        messages = []

        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })

        messages.append({
            "role": "user",
            "content": prompt_text
        })
        params = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "temperature": temperature
        }
        if max_tokens is not None and max_tokens > 0:
            params["max_tokens"] = max_tokens

        start_time = time.time()
        completion = client.chat.completions.create(**params)

        if stream:
            reasoning = ""
            result = ""
            for chunk in completion:
                if hasattr(chunk.choices[0].delta, 'reasoning'):
                    if chunk.choices[0].delta.reasoning:
                        reasoning += chunk.choices[0].delta.reasoning
                        if reason_output:
                            print(
                                chunk.choices[0].delta.reasoning, end="", flush=True)

                if chunk.choices[0].delta.content:
                    # 只在第一次内容出现时打印分隔符
                    if reasoning and not result:
                        print("\n\n--- ANS ---")
                    result += chunk.choices[0].delta.content
                    print(chunk.choices[0].delta.content, end="", flush=True)

            return {
                "status": "success",
                "reasoning": reasoning if reasoning else None,
                "content": result,
                "usage": None,  # 流式输出时无法获取完整的usage信息
                "time": round(time.time() - start_time, 2)
            }
        else:
            return {
                "status": "success",
                "reasoning": completion.choices[0].message.reasoning if hasattr(completion.choices[0].message, 'reasoning') else None,
                "content": completion.choices[0].message.content,
                "usage": completion.usage,
                "time": round(time.time() - start_time, 2)
            }

    except Exception as e:
        return {
            "status": "failed",
            "error": str(e),
            "reasoning": None,
            "content": None
        }


def call_openrouter_llm_with_retry(prompt_text, model="deepseek/deepseek-chat-v3-0324:free", stream=False, temperature=0.6, max_tokens=None, verbose=False, system_prompt=None) -> str:
    """调用OpenRouter API直至成功响应为止, 且只返回响应内容

    Args:
        prompt_text: 用户提示文本
        model: 模型名称
        stream: 是否流式输出
        temperature: 温度参数
        max_tokens: 最大token数
        verbose: 是否输出详细信息
        system_prompt: 系统提示词（可选）
    """
    response = call_openrouter_llm(
        prompt_text, model=model, stream=stream, temperature=temperature, max_tokens=max_tokens, system_prompt=system_prompt)

    while response["status"] != "success":
        print("Error:", response["error"])
        print("Retrying...")
        time.sleep(1)
        response = call_openrouter_llm(
            prompt_text, model=model, stream=stream, temperature=temperature, max_tokens=max_tokens, system_prompt=system_prompt)

    if verbose:
        usage = response["usage"]
        if usage:
            print("Tokens usage:", usage.total_tokens,
                  f"({usage.prompt_tokens} -> {usage.completion_tokens})")
        print("Time taken:", response["time"])
        print("Credits remaining:", get_openrouter_credits(OPENROUTER_API_KEY))

    return response["content"]


# 示例调用
if __name__ == "__main__":
    model = "qwen/qwen3-32b"
    prompt = "如何看待生活"

    response = call_openrouter_llm_with_retry(
        prompt, model, stream=True, temperature=0.6, max_tokens=None)
    print(response)
