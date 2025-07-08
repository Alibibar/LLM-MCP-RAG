
# LLM-MCP-RAG：从零构建极简 Agent 客户端


从零开始，在不依赖第三方框架的情况下，使用 Python 构建一个功能完备的 Agent 客户端

## 主要特性

  * **从零开始，不依赖框架**: 项目代码纯手工打造，不使用 LangChain、LlamaIndex 。
  * **LLM + MCP + RAG 融合**:
      * **大型语言模型 (LLM)**: 集成 `ChatOpenAI` 类，通过调用大模型 API 实现强大的自然语言理解和生成能力。
      * **模型上下文协议 (MCP)**: 利用 `MCPClient` 类，实现与外部工具的交互，使 Agent 具备调用工具的能力。
      * **检索增强生成 (RAG)**: 通过 `EmbeddingRetriever` 和 `VectorStore`，从知识库中检索相关信息，为 LLM 提供更丰富的上下文，有效缓解模型幻觉问题。
  * **异步编程**: 全面采用 `asyncio` 异步编程模型，提高程序运行效率。

##主要组件

  * `Agent.py`: 核心代理，负责协调 LLM 和 MCP 客户端。
  * `ChatOpenAI.py`: 封装 OpenAI 聊天 API，支持流式输出和工具调用。
  * `MCPClient.py`: 实现 MCP 协议，连接并调用外部工具。
  * `EmbeddingRetriever.py`: 文本嵌入与检索，实现 RAG 的核心功能。
  * `VectorStore.py`: 向量数据库，用于存储和检索文本向量。

## 快速开始

### 1\. 安装依赖

```bash
pip install uv
```
Download Node.js from https://nodejs.org/zh-cn/download
### 2\. 配置环境变量

在项目根目录创建 `.env` 文件，并填入 API Key：

```
ALIYUN_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
SILICONFLOW_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

### 3\. 运行示例

```bash
python src/MainTask.py
```

程序将自动执行以下操作：

1.  读取 `knowledge` 目录下的知识库文件。
2.  对知识库内容进行向量化，并构建向量索引。
3.  根据预设的任务，从知识库中检索相关信息。
4.  调用大模型 API，结合检索到的信息生成分析报告。
5.  将生成的报告保存到 `output` 目录下。

##  项目结构

```
.
├── knowledge         # 知识库目录
│   └── Chapter1.txt
├── output            # 输出目录
│   └── guojing.md
├── src               # 源代码目录
│   ├── Agent.py
│   ├── ChatOpenAI.py
│   ├── EmbeddingRetriever.py
│   ├── MainTask.py
│   ├── MCPClient.py
│   ├── utils.py
│   └── VectorStore.py
└── README.md
```

-----

感谢up主@MiuMiu8802的倾囊相授！
