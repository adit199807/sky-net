from langchain_core.tools import  tool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()
MODEL = os.environ.get('MINI_MODEL', '')
@tool
def triple(nums:float):
    """
    parm  nums:a number to be tripled
    returns: the triple of input nunber
    """
    return nums * 3
    
tools = [triple]
llm = ChatOpenAI(model=MODEL, temperature=0).bind_tools(tools)