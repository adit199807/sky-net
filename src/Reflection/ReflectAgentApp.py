from dotenv import load_dotenv
from GrapState import GraphState
from GenerationReflectionChains import reflectionChain, generationChain
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage


load_dotenv()
REFLECT = "reflect"
GENERATE = "generate"


def generation_node(state: GraphState):
    return {"messages": [generationChain.invoke({"messageHistory": state["messages"]})]}


def reflection_node(state: GraphState):
    return {"messages": [reflectionChain.invoke({"messageHistory": state["messages"]})]}

def should_continue(state: GraphState):
    if len(state["messages"]) > 6:
        return END
    return REFLECT

graph = StateGraph(GraphState)
graph.add_node(GENERATE, generation_node)
graph.add_node(REFLECT, reflection_node)

graph.set_entry_point(GENERATE)
graph.add_conditional_edges(GENERATE,should_continue)
graph.add_edge(REFLECT, GENERATE)

grapFlow = graph.compile()

if __name__ == "__main__":
    print("Hello ReAct LangGraph with Function Calling")
    response = grapFlow.invoke({'messages' : [HumanMessage( content="""Make this tweet better:"
                                    @LangChainAI
            — newly Tool Calling feature is seriously underrated.

            After a long wait, it's  here- making the implementation of agents across different models with function calling - super easy.

            Made a video covering their newest blog post

                                  """)]})
    print(response)

