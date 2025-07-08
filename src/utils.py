import re
import json
def log_title(title: str):
    """
    Args:
        title: 日志标题
    """
    print(f"\n{'=' * 20} {title} {'=' * 20}\n")



def read_paragraphs(filename, encoding='utf-8'):
    """使用正则表达式识别段落"""
    with open(filename, 'r', encoding=encoding) as file:
        content = file.read()
    
    # 匹配以全角空格开头的段落
    # 这个正则会匹配从"　　"开始到下一个"　　"之前的所有内容
    pattern = r'　　[^　]+(?:(?!　　).)*'
    
    paragraphs = re.findall(pattern, content, re.DOTALL)
    
    # 清理段落
    paragraphs = [para.strip() for para in paragraphs if para.strip()]
    
    return paragraphs

