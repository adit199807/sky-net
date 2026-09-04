from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, MessagesState, END
from src.Node import runAgentReasoning, toolNode

load_dotenv()
AGENT_REASON = 'agent_reason'
ACT = 'act'
LAST = -1

def shouldContinue(state: MessagesState):
    if not state['messages'][LAST].tool_calls:
        return END
    return ACT
# StateGraph(schema = MessagesState)
flow = StateGraph(MessagesState)
flow.add_node(AGENT_REASON, runAgentReasoning)
flow.set_entry_point(AGENT_REASON)

flow.add_node(ACT, toolNode)
# both pattern are correct for us in this file. Because we already defined the node name ACT and END in our graph. 
# If we wanted cutom output from shouldContinue function, then 3rd parameter add_conditional_edges would be necessary
#  to map our custom value to our graph node name
flow.add_conditional_edges(AGENT_REASON, shouldContinue, { ACT: ACT, END:END})
# flow.add_conditional_edges(AGENT_REASON, shouldContinue)
flow.add_edge(ACT, AGENT_REASON)


graph = flow.compile()
graph.get_graph().draw_mermaid_png(output_file_path="src\\statics\\flow.png")

if __name__ == "__main__":
    print("Hello ReAct LangGraph with Function Calling")
    # StateGraph.invoke(schema)
    res = graph.invoke({"messages": [HumanMessage(content="What is the temperature in Tokyo? List it and then triple it")]})
    print(res["messages"][LAST].content)
