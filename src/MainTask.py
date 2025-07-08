
import os
import asyncio
from pathlib import Path
from MCPClient import MCPClient
from Agent import Agent
from EmbeddingRetriever import EmbeddingRetriever
from utils import log_title, read_paragraphs

# 常量定义
URL = 'https://news.ycombinator.com/'
OUT_PATH = Path.cwd() / 'output'
TASK = f"""
先从我给你的context中找到相关信息，接着找出郭靖干的最多的事情，
把郭靖干的最多的事情保存到{OUT_PATH}/guojing.md,输出一个漂亮md文件
"""

# MCP 客户端初始化
# fetch_mcp = MCPClient("mcp-server-fetch", "uvx", ['mcp-server-fetch'])
file_mcp = MCPClient("mcp-server-file", "npx", ['-y', '@modelcontextprotocol/server-filesystem', str(OUT_PATH)])

async def main():
    """主函数：执行 RAG 检索和 Agent 任务"""
    # RAG 检索上下文
    context = await retrieve_context()

    # 创建并运行 Agent
    # agent = Agent('qwen3-235b-a22b', [fetch_mcp, file_mcp], '', context)
    agent = Agent('qwen3-235b-a22b', [file_mcp], '', context)

    await agent.init()
    await agent.invoke(TASK)
    await agent.close()

async def retrieve_context():
    """
    检索相关上下文信息
    使用 RAG (Retrieval-Augmented Generation) 从知识库中检索相关文档
    """
    # 初始化嵌入检索器
    embedding_retriever = EmbeddingRetriever("BAAI/bge-m3")
    
    # 读取知识库目录中的所有文件
    knowledge_dir = Path.cwd() / 'knowledge'
    files = os.listdir(knowledge_dir)
    
    # 将每个文件的内容添加到嵌入数据库
    for file in files:
        log_title(f'Processing file: {file}')
        paragraphs = read_paragraphs(file)
        for i in range(len(paragraphs)):
            print(f'Embedding paragraph {i + 1}/{len(paragraphs)}...')
            content = paragraphs[i]
            await embedding_retriever.embed_document(content)
    
    # 基于任务检索最相关的文档片段
    retrieved_docs = await embedding_retriever.retrieve(TASK, top_k=5)
    context = '\n'.join(retrieved_docs)
    
    # 显示检索到的上下文
    log_title('CONTEXT')
    print(context)
    
    return context

if __name__ == "__main__":
    asyncio.run(main())