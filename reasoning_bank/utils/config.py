"""
配置管理模块

支持从 YAML 文件加载配置，并支持环境变量覆盖
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

# 全局配置存储
_config: Optional[Dict[str, Any]] = None


def _resolve_env_vars(config: Dict[str, Any]) -> Dict[str, Any]:
    """递归解析配置中的环境变量引用
    
    支持格式: ${ENV_VAR:default_value}
    """
    def resolve_value(value: Any) -> Any:
        if isinstance(value, str):
            # 检查是否包含环境变量引用
            if value.startswith("${") and "}" in value:
                # 解析 ${VAR:default} 格式
                var_part = value[2:value.index("}")]
                if ":" in var_part:
                    var_name, default = var_part.split(":", 1)
                else:
                    var_name, default = var_part, ""
                return os.getenv(var_name, default)
            return value
        elif isinstance(value, dict):
            return {k: resolve_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [resolve_value(item) for item in value]
        return value
    
    return resolve_value(config)


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """加载配置文件
    
    Args:
        config_path: 配置文件路径，默认为 reasoning_bank/config/config.yaml
        
    Returns:
        配置字典
    """
    global _config
    
    if config_path is None:
        # 默认配置文件路径
        base_dir = Path(__file__).parent.parent
        config_path = base_dir / "config" / "config.yaml"
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # 解析环境变量
    config = _resolve_env_vars(config)
    
    # 从环境变量覆盖关键配置
    if os.getenv("OPENROUTER_API_KEY"):
        config["llm"]["api_key"] = os.getenv("OPENROUTER_API_KEY")
    if os.getenv("OPENROUTER_API_BASE"):
        config["llm"]["api_base"] = os.getenv("OPENROUTER_API_BASE")
    
    _config = config
    return config


def get_config(key: Optional[str] = None, default: Any = None) -> Any:
    """获取配置项
    
    Args:
        key: 配置键，支持点号分隔的嵌套键（如 "llm.model"）
        default: 默认值
        
    Returns:
        配置值
    """
    global _config
    
    if _config is None:
        load_config()
    
    if key is None:
        return _config
    
    # 支持点号分隔的嵌套键
    keys = key.split(".")
    value = _config
    
    try:
        for k in keys:
            value = value[k]
        return value
    except (KeyError, TypeError):
        return default


def update_config(key: str, value: Any) -> None:
    """动态更新配置项
    
    Args:
        key: 配置键，支持点号分隔的嵌套键
        value: 新值
    """
    global _config
    
    if _config is None:
        load_config()
    
    keys = key.split(".")
    config = _config
    
    for k in keys[:-1]:
        if k not in config:
            config[k] = {}
        config = config[k]
    
    config[keys[-1]] = value

