from dotenv import load_dotenv
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode
from src.ReAct.React import llm, tools
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage


load_dotenv()

SYSYEM_MESSAGE="""
You are a helpful assistant that can use tools to answer questions.
"""

def runAgentReasoning(state: MessagesState):
    """
    Run the agen reasoninig node
    """
    messages = []
    messages.append(SystemMessage(content=SYSYEM_MESSAGE))
    messages.extend(state['messages'])
    response = llm.invoke(messages)
    return {'messages' : [response]}

toolNode = ToolNode(tools)