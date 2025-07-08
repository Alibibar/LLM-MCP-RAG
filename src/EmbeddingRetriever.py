import os
import aiohttp
from dotenv import load_dotenv
from VectorStore import VectorStore
from utils import log_title


# 加载环境变量
load_dotenv()

API_KEY = os.getenv('SILICONFLOW_API_KEY')


class EmbeddingRetriever:
    """
    嵌入向量检索器类
    负责将文档和查询转换为嵌入向量，并提供基于向量相似度的检索功能
    """    
    def __init__(self, embedding_model: str):
        self.embedding_model = embedding_model
        self.vector_store = VectorStore()
    
    async def embed_document(self, document: str):
        """
        嵌入文档并存储到向量库
        Args:
            document: 要嵌入的文档内容  
        Returns:
            文档的嵌入向量
        """
        log_title('EMBEDDING DOCUMENT')

        embedding = await self.embed_text(document)
        await self.vector_store.add_embedding(embedding, document)
        return embedding
    
    async def embed_query(self, query: str):
        log_title('EMBEDDING QUERY')
        embedding = await self.embed_text(query)
        return embedding

    async def embed_text(self, text: str):
        """
        调用嵌入API将文本转换为向量
        Args:
            text: 要嵌入的文本
        Returns:
            嵌入向量（浮点数列表）
        """

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url="https://api.siliconflow.cn/v1/embeddings",
                json={
                    "model": self.embedding_model,
                    "input": text
                },
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json"
                }
            ) as response:
                data = await response.json()
                embedding = data['data'][0]['embedding']
                return embedding
    
    async def retrieve(self, query: str, top_k: int = 5):
        """
        根据查询检索最相似的文档
        Args:
            query: 查询文本
            top_k: 返回最相似的前K个文档，默认为5
        Returns:
            按相似度排序的文档列表
        """
        # 获取查询的嵌入向量

        query_embedding = await self.embed_query(query)

        # 在向量存储中搜索最相似的文档
        return await self.vector_store.search(query_embedding, top_k)

