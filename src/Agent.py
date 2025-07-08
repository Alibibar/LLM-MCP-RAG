import json
from typing import List, Optional
from MCPClient import MCPClient
from ChatOpenAI import ChatOpenAI
from utils import log_title

class Agent:
    """MCP代理类，用于管理多个MCP客户端和LLM交互"""
    def __init__(self, model: str, mcp_clients: List[MCPClient], system_prompt: str='', context: str=''):
        """
        初始化Agent
        Args:
            model: 模型名称
            mcp_clients: MCP客户端列表
            system_prompt: 系统提示词
            context: 上下文信息
        """
        self.model = model
        self.mcp_clients = mcp_clients
        self.system_prompt = system_prompt
        self.context = context
        self.llm: Optional[ChatOpenAI] = None

    async def init(self):
        """初始化代理，连接所有MCP客户端并创建LLM实例"""
        log_title("TOOLS")

        # 初始化所有MCP客户端
        for client in self.mcp_clients:
            await client.init()
        
        # 获取所有工具
        tools = []
        for client in self.mcp_clients:
            tools.extend(client.get_tools())
        
        # 创建LLM实例
        self.llm = ChatOpenAI(
            model=self.model,
            system_prompt=self.system_prompt,
            tools=tools,
            context=self.context
        )
    async def close(self):
        """关闭代理，断开所有MCP客户端连接"""
        print("Closing Agent...")

        # 关闭每个 MCP 客户端
        for i, client in enumerate(self.mcp_clients):
            print(f"  Closing MCP Client {i+1}/{len(self.mcp_clients)}: {client.name}")
            try:
                await client.close()
                print(f" ✔ {client.name} closed")
            except Exception as e:
                print(f" ❌ Error closing {client.name}: {e}")

        print("Agent closed")

    async def invoke(self, prompt: str):
        """
        调用代理执行任务
        Args:
            prompt: 用户输入的提示词
        Returns:
            最终的响应内容 
        Raises:
            Exception: 如果代理未初始化
        """
        # 添加初始化检查
        if self.llm is None:
            raise RuntimeError("Agent not initialized. Call init() first.")
            
        # 获取初始响应
        response = await self.llm.chat(prompt)

        while True:
            #检查是否有工具调用
            if len(response.get('toolCalls', [])) > 0:  # 添加默认值防止KeyError
                #处理每一个工具调用
                for tool_call in response['toolCalls']:
                    # 找到第一个支持 tool_call 中指定函数的 MCP 客户端，将其赋值给 mcp，并停止进一步的循环以提高效率。
                    #查找对应MCP客户端
                    mcp = None
                    for client in self.mcp_clients:
                        client_tools = client.get_tools()
                        if any(tool.name == tool_call['function']['name'] for tool in client_tools):
                            mcp = client
                            break
                    
                    if mcp:
                        log_title('TOOL USE')
                        print(f"Calling tool: {tool_call['function']['name']}")
                        print(f"Arguments: {tool_call['function']['arguments']}")

                        try:
                            # 调用工具
                            result = await mcp.call_tool(
                                tool_call['function']['name'],
                                json.loads(tool_call['function']['arguments'])
                            )

                            # 处理 CallToolResult 对象
                            result_str = self._format_tool_result(result)
                            
                            print(f"Result: {result_str[:500]}...")

                            # 将工具结果添加到对话历史
                            self.llm.append_tool_result(
                                tool_call['id'], 
                                result_str
                            )

                        except Exception as e:
                            # 工具调用失败时的错误处理
                            error_msg = f"Tool execution failed: {str(e)}"
                            print(f"Error: {error_msg}")
                            self.llm.append_tool_result(
                                tool_call['id'],
                                error_msg
                            )

                    else:
                        # 工具未找到
                        self.llm.append_tool_result(
                            tool_call['id'],
                            'Tool not found'
                        )
                # 工具调用后，继续对话
                response = await self.llm.chat()
                continue

            # 如果没有工具调用，结束对话
            return response['content']
        
    def _format_tool_result(self, result):
        """格式化工具调用结果"""
        if hasattr(result, 'content'):
            # 如果 result 有 content 属性（这是 MCP 标准的响应格式）
            result_content = result.content
            if isinstance(result_content, list):
                # 如果 content 是列表，提取文本内容
                text_contents = []
                for item in result_content:
                    if hasattr(item, 'text'):
                        text_contents.append(item.text)
                    elif isinstance(item, dict) and 'text' in item:
                        text_contents.append(item['text'])
                    else:
                        text_contents.append(str(item))
                return '\n'.join(text_contents)
            else:
                return str(result_content)
        elif hasattr(result, '__dict__'):
            # 如果是一个对象，转换为字典
            return json.dumps(result.__dict__)
        else:
            # 否则转换为字符串
            return str(result)
