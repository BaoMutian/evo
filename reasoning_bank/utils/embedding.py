"""
Embedding 服务模块

使用 sentence-transformers 进行文本向量化
"""

import numpy as np
from typing import List, Union, Optional
from pathlib import Path

# 延迟导入，避免启动时加载模型
_model = None
_model_name = None


def _get_model(model_name: str, device: str = "cpu"):
    """获取或加载模型（单例模式）"""
    global _model, _model_name
    
    if _model is None or _model_name != model_name:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(model_name, device=device)
        _model_name = model_name
    
    return _model


class EmbeddingService:
    """Embedding 服务类"""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        batch_size: int = 32,
    ):
        """初始化 Embedding 服务
        
        Args:
            model_name: 模型名称
            device: 设备（cpu/cuda）
            batch_size: 批处理大小
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model = None
    
    @property
    def model(self):
        """延迟加载模型"""
        if self._model is None:
            self._model = _get_model(self.model_name, self.device)
        return self._model
    
    def encode(
        self,
        texts: Union[str, List[str]],
        normalize: bool = True,
    ) -> np.ndarray:
        """将文本转换为向量
        
        Args:
            texts: 单个文本或文本列表
            normalize: 是否归一化向量
            
        Returns:
            向量数组，shape=(n, dim)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        )
        
        return embeddings
    
    def similarity(
        self,
        query: Union[str, np.ndarray],
        candidates: Union[List[str], np.ndarray],
    ) -> np.ndarray:
        """计算查询与候选项之间的相似度
        
        Args:
            query: 查询文本或向量
            candidates: 候选文本列表或向量数组
            
        Returns:
            相似度数组
        """
        # 如果是文本，先转换为向量
        if isinstance(query, str):
            query_vec = self.encode(query)
        else:
            query_vec = query
        
        if isinstance(candidates, list) and isinstance(candidates[0], str):
            candidate_vecs = self.encode(candidates)
        else:
            candidate_vecs = candidates
        
        # 确保向量是二维的
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)
        
        # 计算余弦相似度（假设向量已归一化）
        similarities = np.dot(query_vec, candidate_vecs.T).flatten()
        
        return similarities
    
    def get_dimension(self) -> int:
        """获取向量维度"""
        return self.model.get_sentence_embedding_dimension()


# 全局单例
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: str = "cpu",
) -> EmbeddingService:
    """获取 Embedding 服务单例
    
    Args:
        model_name: 模型名称
        device: 设备
        
    Returns:
        EmbeddingService 实例
    """
    global _embedding_service
    
    if _embedding_service is None:
        _embedding_service = EmbeddingService(model_name, device)
    
    return _embedding_service

