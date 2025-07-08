from typing import List, Dict, Any, Optional
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class Tool:
    """工具定义类""" 
    def __init__(self, name: str, description: str, input_schema: Dict[str, Any]):
        self.name = name
        self.description = description
        self.input_schema = input_schema
        
    def __repr__(self):
        return f"Tool(name='{self.name}',\n description='{self.description}', \n input_schema={self.input_schema}\n)"

class MCPClient:
    
    def __init__(self, name: str, command: str, args: List[str], version: Optional[str] = None):
        """
        初始化MCP客户端
        Args:
            name: 客户端名称
            command: 服务器命令  
            args: 命令参数列表
            version: 客户端版本号（此参数在当前MCP Python SDK中未使用）
        """
        self.name = name
        self.version = version or "0.0.1"
        self.command = command
        self.args = args
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.tools: List[Tool] = []
        self._initialized = False  # 初始化状态跟踪
    
    async def init(self):
        """初始化客户端连接"""
        if self._initialized:
            print(f"The {self.name} has been initialized. Skip it")
            return
            
        try:
            await self._connect_to_server()
            self._initialized = True
            print(f"✅ {self.name} initialized successfully")
        except Exception as e:
            print(f"❌ {self.name} initialization failed: {e}")
            raise
    
    async def close(self):
        """关闭客户端连接"""
        if not self._initialized:
            return
        try:
            # 先关闭会话
            if self.session:
                self.session = None
            # 关闭资源管理器，并捕获可能的异常
            await self.exit_stack.aclose()
            
        except Exception as e:
            # 忽略关闭时的异常，因为这通常是由于资源已经被清理导致的
            print(f"⚠️ Warning during {self.name} cleanup: {e}")
        finally:
            self._initialized = False
                
    def get_tools(self) -> List[Tool]:
        """
        获取可用工具列表
        """
        return self.tools
    
    async def call_tool(self, name: str, params: Dict[str, Any]):
        """
        调用指定工具
        Args:
            name: 工具名称
            params: 工具参数
        Returns:
            工具调用结果
        """
        if not self.session:
            raise RuntimeError(f"Client {self.name} not initialized. Call init() first.")
        
        if not self._initialized:
            raise RuntimeError(f"Client {self.name} not properly initialized.")
        
        try:
            return await self.session.call_tool(name, params)
        except Exception as e:
            print(f"Tool call failed - Client: {self.name}, Tool: {name}, Error: {e}")
            raise
    
    async def _connect_to_server(self):
        """连接到MCP服务器（私有方法）"""
        try:
            print(f"Connecting to {self.name} server...")
            
            # 创建服务器参数
            server_params = StdioServerParameters(
                command=self.command,
                args=self.args,
                env=None
            )
            
            # 建立stdio连接
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            stdio, write = stdio_transport
            
            # 创建客户端会话
            self.session = await self.exit_stack.enter_async_context(
                ClientSession(stdio, write)
            )
            
            # 初始化会话
            await self.session.initialize()
            
            # 获取工具列表
            tools_result = await self.session.list_tools()
            self.tools = [
                Tool(
                    name=tool.name,
                    description=tool.description or "",
                    input_schema=tool.inputSchema or {}
                )
                for tool in tools_result.tools
            ]
            
            print(f"✅ Connected to {self.name} with tools: {[tool.name for tool in self.tools]}")
            
        except Exception as e:
            print(f"❌ Failed to connect to MCP server {self.name}: {e}")
            raise