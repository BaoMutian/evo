"""
记忆库模块

ReasoningBank 的核心实现，负责存储和检索推理经验
"""

import os
import json
import uuid
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional

import numpy as np

from reasoning_bank.utils.embedding import EmbeddingService, get_embedding_service
from reasoning_bank.utils.config import get_config
from reasoning_bank.utils.logger import get_logger

logger = get_logger("memory")


@dataclass
class MemoryContent:
    """记忆内容条目"""
    title: str  # 策略标题
    description: str  # 一句话简介
    content: str  # 核心推理建议/避坑指南


@dataclass
class MemoryItem:
    """记忆项"""
    id: str
    original_query: str  # 原始查询/任务
    items: List[Dict[str, str]]  # 记忆内容列表 [{title, description, content}]
    is_success: bool  # 是否成功经验
    source_trajectory_id: str = ""  # 关联的轨迹 ID
    timestamp: str = ""  # 创建时间
    query_embedding: Optional[List[float]] = None  # 查询向量（内部使用）
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())[:8]
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（不包含 embedding）"""
        return {
            "id": self.id,
            "original_query": self.original_query,
            "items": self.items,
            "is_success": self.is_success,
            "source_trajectory_id": self.source_trajectory_id,
            "timestamp": self.timestamp,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryItem":
        """从字典创建"""
        return cls(
            id=data.get("id", ""),
            original_query=data["original_query"],
            items=data["items"],
            is_success=data["is_success"],
            source_trajectory_id=data.get("source_trajectory_id", ""),
            timestamp=data.get("timestamp", ""),
            query_embedding=data.get("query_embedding"),
        )
    
    def format_for_prompt(self) -> str:
        """格式化为 Prompt 注入格式"""
        lines = []
        for item in self.items:
            lines.append(f"【{item['title']}】")
            if item.get('description'):
                lines.append(f"  简介: {item['description']}")
            lines.append(f"  建议: {item['content']}")
            lines.append("")
        return "\n".join(lines)


class MemoryBank:
    """推理记忆库"""
    
    def __init__(
        self,
        storage_path: Optional[str] = None,
        bank_name: str = "default",
        top_k: int = 1,
        similarity_threshold: float = 0.5,
        embedding_service: Optional[EmbeddingService] = None,
    ):
        """初始化记忆库
        
        Args:
            storage_path: 存储路径
            bank_name: 记忆库名称（用于区分不同数据集）
            top_k: 检索时返回的记忆数量
            similarity_threshold: 相似度阈值
            embedding_service: Embedding 服务实例
        """
        self.storage_path = Path(storage_path or get_config("memory.storage_path", "./data/memory_banks"))
        self.bank_name = bank_name
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        
        # Embedding 服务
        self.embedding_service = embedding_service or get_embedding_service()
        
        # 记忆存储
        self.memories: List[MemoryItem] = []
        self.embeddings: Optional[np.ndarray] = None
        
        # 确保存储目录存在
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # 尝试加载已有记忆
        self._load()
    
    @property
    def bank_file(self) -> Path:
        """记忆库文件路径"""
        return self.storage_path / f"{self.bank_name}.jsonl"
    
    @property
    def embedding_file(self) -> Path:
        """向量文件路径"""
        return self.storage_path / f"{self.bank_name}_embeddings.npy"
    
    def _load(self):
        """从文件加载记忆"""
        if not self.bank_file.exists():
            logger.info(f"记忆库 {self.bank_name} 不存在，创建新库")
            return
        
        try:
            with open(self.bank_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        self.memories.append(MemoryItem.from_dict(data))
            
            # 加载向量
            if self.embedding_file.exists():
                self.embeddings = np.load(self.embedding_file)
            else:
                # 重新计算向量
                self._rebuild_embeddings()
            
            logger.info(f"加载记忆库 {self.bank_name}，共 {len(self.memories)} 条记忆")
            
        except Exception as e:
            logger.error(f"加载记忆库失败: {e}")
            self.memories = []
            self.embeddings = None
    
    def _rebuild_embeddings(self):
        """重建向量索引"""
        if not self.memories:
            self.embeddings = None
            return
        
        queries = [m.original_query for m in self.memories]
        self.embeddings = self.embedding_service.encode(queries)
        
        # 保存向量
        np.save(self.embedding_file, self.embeddings)
    
    def save(self):
        """保存记忆库到文件"""
        try:
            with open(self.bank_file, "w", encoding="utf-8") as f:
                for memory in self.memories:
                    f.write(json.dumps(memory.to_dict(), ensure_ascii=False) + "\n")
            
            if self.embeddings is not None:
                np.save(self.embedding_file, self.embeddings)
            
            logger.info(f"保存记忆库 {self.bank_name}，共 {len(self.memories)} 条记忆")
            
        except Exception as e:
            logger.error(f"保存记忆库失败: {e}")
    
    def add(
        self,
        query: str,
        items: List[Dict[str, str]],
        is_success: bool,
        trajectory_id: str = "",
    ) -> MemoryItem:
        """添加新记忆
        
        Args:
            query: 原始查询
            items: 记忆内容列表 [{title, description, content}]
            is_success: 是否成功
            trajectory_id: 轨迹 ID
            
        Returns:
            创建的 MemoryItem
        """
        memory = MemoryItem(
            id="",
            original_query=query,
            items=items,
            is_success=is_success,
            source_trajectory_id=trajectory_id,
        )
        
        # 计算向量
        query_vec = self.embedding_service.encode(query)
        memory.query_embedding = query_vec.tolist()
        
        # 添加到列表
        self.memories.append(memory)
        
        # 更新向量索引
        if self.embeddings is None:
            self.embeddings = query_vec.reshape(1, -1)
        else:
            self.embeddings = np.vstack([self.embeddings, query_vec])
        
        logger.info(f"添加记忆: {memory.id} (success={is_success})")
        
        # 自动保存
        self.save()
        
        return memory
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> List[MemoryItem]:
        """检索相关记忆
        
        Args:
            query: 查询文本
            top_k: 返回数量
            threshold: 相似度阈值
            
        Returns:
            相关记忆列表
        """
        if not self.memories or self.embeddings is None:
            return []
        
        top_k = top_k or self.top_k
        threshold = threshold if threshold is not None else self.similarity_threshold
        
        # 计算相似度
        query_vec = self.embedding_service.encode(query)
        similarities = self.embedding_service.similarity(query_vec, self.embeddings)
        
        # 获取 top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            sim = similarities[idx]
            if sim >= threshold:
                results.append(self.memories[idx])
                logger.debug(f"检索到记忆 {self.memories[idx].id}, 相似度: {sim:.4f}")
        
        logger.info(f"检索到 {len(results)} 条相关记忆")
        return results
    
    def format_memories_for_prompt(self, memories: List[MemoryItem]) -> str:
        """将记忆格式化为 Prompt 格式
        
        Args:
            memories: 记忆列表
            
        Returns:
            格式化的字符串
        """
        if not memories:
            return ""
        
        lines = ["【相关过往经验】:", ""]
        for i, mem in enumerate(memories, 1):
            if len(memories) > 1:
                lines.append(f"--- 经验 {i} ---")
            lines.append(mem.format_for_prompt())
        
        return "\n".join(lines)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取记忆库统计信息"""
        if not self.memories:
            return {
                "total": 0,
                "success": 0,
                "failure": 0,
            }
        
        success_count = sum(1 for m in self.memories if m.is_success)
        
        return {
            "total": len(self.memories),
            "success": success_count,
            "failure": len(self.memories) - success_count,
            "bank_name": self.bank_name,
        }
    
    def clear(self):
        """清空记忆库"""
        self.memories = []
        self.embeddings = None
        
        # 删除文件
        if self.bank_file.exists():
            self.bank_file.unlink()
        if self.embedding_file.exists():
            self.embedding_file.unlink()
        
        logger.info(f"清空记忆库 {self.bank_name}")
    
    def __len__(self) -> int:
        return len(self.memories)
    
    def __repr__(self) -> str:
        stats = self.get_stats()
        return f"MemoryBank(name={self.bank_name}, total={stats['total']}, success={stats['success']}, failure={stats['failure']})"

