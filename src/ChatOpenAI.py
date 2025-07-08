import os
from typing import List, Dict, Any, Optional
from openai import OpenAI
from dotenv import load_dotenv
from utils import log_title
# 加载环境变量
load_dotenv()

class ToolCall:
    """
    工具调用类
    用于封装 OpenAI 返回的工具调用信息
    """
    def __init__(self, id: str='', function: Dict[str, str]=None):
        """  
        Args:
            id: 工具调用的唯一标识符
            function: 包含工具名称和参数的字典
        """
        self.id = id
        self.function = function or {"name": "", "arguments": ""}

class Tool:
    """
    工具定义类
    用于定义可以被 AI 调用的工具
    """
    def __init__(self, name: str, description: str, input_schema: Dict[str, Any]):
        """
        Args:
            name: 工具名称
            description: 工具描述
            input_schema: 工具输入参数的 JSON Schema
        """
        self.name = name
        self.description = description
        self.input_schema = input_schema



class ChatOpenAI:
    """OpenAI聊天客户端类"""
    
    def __init__(self, model: str, system_prompt: str = '', tools: List[Tool] = None, context: str = ''):
        """
        初始化ChatOpenAI实例
        
        Args:
            model: 模型名称
            system_prompt: 系统提示词
            tools: 工具列表
            context: 上下文信息
        """
        self.llm = OpenAI(
            api_key=os.getenv('ALIYUN_API_KEY'),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.model = model
        self.tools = tools or []
        self.messages = []
        
        # 添加系统提示词
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})
        
        # 添加上下文
        if context:
            self.messages.append({"role": "user", "content": context})
    
    async def chat(self, prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        进行聊天对话
        
        Args:
            prompt: 用户输入的提示词
            
        Returns:
            包含content和toolCalls的字典
        """
        log_title('CHAT')
        
        # 如果提供了新的用户消息，添加到消息历史中并显示
        if prompt:
            print(f"User: {prompt}")
            self.messages.append({
                "role": "user", 
                "content": prompt
            })
        else:
            # 如果没有新的用户消息，显示当前状态
            print("Continuing to process tool call results...")
        try:
            # 创建流式聊天完成
            stream = self.llm.chat.completions.create(
                model=self.model,
                messages=self.messages,
                stream=True,
                tools=self._get_tools_definition() if self.tools else None,  # 只有工具存在时才传递
            )
            
            content = ""
            tool_calls = []
            
            log_title('RESPONSE')
            print("AI: ", end="", flush=True)  # 添加AI标识
            
            # 处理流式响应
            for chunk in stream:
                if not chunk.choices:  # 检查choices是否存在
                    continue
                    
                delta = chunk.choices[0].delta
                
                # 处理普通内容
                if delta.content:
                    content_chunk = delta.content
                    content += content_chunk
                    print(content_chunk, end='', flush=True)  

                # 处理工具调用
                if delta.tool_calls:
                    for tool_call_chunk in delta.tool_calls:
                        # 第一次需要创建新的工具调用
                        while len(tool_calls) <= tool_call_chunk.index:
                            tool_calls.append({
                                'id': '', 
                                'function': {'name': '', 'arguments': ''}
                            })
                        
                        current_call = tool_calls[tool_call_chunk.index]
                        
                        if tool_call_chunk.id:
                            current_call['id'] += tool_call_chunk.id
                        if tool_call_chunk.function and tool_call_chunk.function.name:
                            current_call['function']['name'] += tool_call_chunk.function.name
                        if tool_call_chunk.function and tool_call_chunk.function.arguments:
                            current_call['function']['arguments'] += tool_call_chunk.function.arguments
            
            print()  # 添加换行
            
            # 如果有工具调用但没有文本内容，显示工具调用信息
            if tool_calls and not content:
                print("The tool is being invoked...")

            # 构建工具调用格式用于消息历史
            formatted_tool_calls = [
                {
                    "id": call['id'],
                    "type": "function",
                    "function": call['function']
                }
                for call in tool_calls
            ] if tool_calls else None

            # 添加助手响应到消息历史
            assistant_message = {
                "role": "assistant",
                "content": content,
            }
            if formatted_tool_calls:
                assistant_message["tool_calls"] = formatted_tool_calls
                
            self.messages.append(assistant_message)
            
            return {
                "content": content,
                "toolCalls": tool_calls,
            }
            
        except Exception as e:
            print(f"\n❌ API call failed: {e}")
            raise
    
    def append_tool_result(self, tool_call_id: str, tool_output: str):
        """
        添加工具执行结果到消息历史
        
        Args:
            tool_call_id: 工具调用ID
            tool_output: 工具输出结果
        """
        self.messages.append({
            "role": "tool",
            "content": tool_output,
            "tool_call_id": tool_call_id
        })
    
    def _get_tools_definition(self) -> List[Dict[str, Any]]:
        """
        获取工具定义格式
        Returns:
            OpenAI API格式的工具定义列表
        """
        if not self.tools:
            return []
            
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                },
            }
            for tool in self.tools
        ]