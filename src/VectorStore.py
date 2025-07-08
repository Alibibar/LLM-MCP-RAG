from typing import List
import math

class VectorStoreItem:
    """向量存储项，包含嵌入向量和对应的文档内容"""
    def __init__(self, embedding: List[float], document: str):
        """
        初始化向量存储项
        Args:
            embedding: 文档的嵌入向量
            document: 原始文档内容
        """
        self.embedding = embedding
        self.document = document

class VectorStore:
    """
    向量存储类，用于存储文档嵌入向量并提供相似性搜索功能
    使用余弦相似度算法来计算查询向量与存储向量的相似性
    """
    def __init__(self):
        """初始化空的向量存储"""
        self.vector_store: List[VectorStoreItem] = []

    async def add_embedding(self, embedding: List[float], document: str):
        """
        添加文档嵌入向量到存储中
        Args:
            embedding: 文档的嵌入向量（浮点数列表）
            document: 对应的文档内容字符串
        """
        self.vector_store.append(VectorStoreItem(embedding, document))

    async def search(self, query_embedding: List[float], top_k: int = 5) -> List[str]:
        """
        搜索与查询向量最相似的文档
        Args:
            query_embedding: 查询的嵌入向量
            top_k: 返回最相似的前K个文档，默认为3
        Returns:
            按相似度排序的文档内容列表（最相似的在前）
        """
        # 计算查询向量与所有存储向量的相似度分数
        scored = [
            {
                'document': item.document,
                'score': self._cosine_similarity(query_embedding, item.embedding)
            }
            for item in self.vector_store
        ]

        # 按相似度分数降序排序，取前top_k个文档
        top_k_documents = [
            item['document']
            for item in sorted(scored, key=lambda x: x['score'], reverse=True)[:top_k]
        ]
        return top_k_documents
    def _cosine_similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        """
        计算两个向量的余弦相似度
        Args:
            vec_a: 第一个向量
            vec_b: 第二个向量  
        Returns:
            余弦相似度值，范围在[-1, 1]之间
            1表示完全相同，0表示正交，-1表示完全相反
        """
        # 计算点积：对应元素相乘后求和
        dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
        
        # 计算向量A的模长：各元素平方和的平方根
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        
        # 计算向量B的模长：各元素平方和的平方根
        norm_b = math.sqrt(sum(b * b for b in vec_b))
        
        # 返回余弦相似度：点积除以两个向量模长的乘积
        return dot_product / (norm_a * norm_b)