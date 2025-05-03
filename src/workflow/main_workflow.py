from typing import Annotated, List, TypedDict
from langgraph.graph import Graph, StateGraph
from langchain_core.messages import HumanMessage, AIMessage
from langchain.callbacks.manager import CallbackManager
from src.agents.data_wrangling_agent import create_data_wrangling_agent
from src.agents.analysis_agent import create_analysis_agent
from src.agents.visualization_agent import create_visualization_agent
from src.config.langsmith_config import get_callback_manager

class AgentState(TypedDict):
    messages: Annotated[List[HumanMessage | AIMessage], "The messages in the conversation"]
    data: Annotated[dict, "The data being processed"]
    current_agent: Annotated[str, "The current agent processing the data"]

def create_workflow():
    # Create the workflow
    workflow = StateGraph(AgentState)
    
    # Get LangSmith callback manager
    callback_manager = get_callback_manager()
    
    # Create agents with LangSmith tracing
    data_wrangling_agent = create_data_wrangling_agent(callback_manager)
    analysis_agent = create_analysis_agent(callback_manager)
    visualization_agent = create_visualization_agent(callback_manager)
    
    # Add nodes
    workflow.add_node("data_wrangling", data_wrangling_agent)
    workflow.add_node("analysis", analysis_agent)
    workflow.add_node("visualization", visualization_agent)
    
    # Set the entry point
    workflow.set_entry_point("data_wrangling")
    
    # Add edges
    workflow.add_edge("data_wrangling", "analysis")
    workflow.add_edge("analysis", "visualization")
    workflow.add_edge("visualization", "data_wrangling")
    
    # Compile the workflow
    return workflow.compile() 