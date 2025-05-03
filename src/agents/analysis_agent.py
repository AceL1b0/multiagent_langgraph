from typing import Annotated, List, TypedDict
from langgraph.graph import Graph, StateGraph
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
import pandas as pd
import numpy as np
from langchain.callbacks.manager import CallbackManager

class AnalysisState(TypedDict):
    messages: Annotated[List[HumanMessage | AIMessage], "The messages in the conversation"]
    data: Annotated[dict, "The data being processed"]
    current_agent: Annotated[str, "The current agent processing the data"]

def create_analysis_agent(callback_manager=None):
    # Create the analysis agent
    llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0.2)
    
    # Enhanced analysis prompt with specific instructions
    analysis_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert data analyst. Your role is to analyze data and extract meaningful insights.

Key Responsibilities:
1. Descriptive Analysis:
   - Calculate summary statistics
   - Identify patterns and trends
   - Analyze distributions
   - Compare groups and categories

2. Statistical Analysis:
   - Perform correlation analysis
   - Conduct hypothesis testing
   - Identify significant relationships
   - Analyze variance and distributions

3. Feature Analysis:
   - Identify important features
   - Analyze feature relationships
   - Detect anomalies and outliers
   - Understand feature importance

4. Insight Generation:
   - Extract meaningful patterns
   - Identify key findings
   - Provide actionable insights
   - Suggest further analysis directions

You will receive cleaned data and should return comprehensive analysis results."""),
        MessagesPlaceholder(variable_name="messages"),
        ("human", """Please analyze the following data:

Data Info:
{data_info}

Please perform a comprehensive analysis and provide insights."""),
    ])
    
    def analysis_agent(state: AnalysisState) -> AnalysisState:
        try:
            # Get the processed data from the state
            df = state["data"]["processed_data"]
            
            # Get data info
            data_info = df.describe().to_string()
            
            # Create analysis chain
            chain = analysis_prompt | llm
            
            # Generate analysis instructions
            response = chain.invoke({
                "messages": state["messages"],
                "data_info": data_info
            })
            
            # Perform analysis
            analysis_results = []
            
            # 1. Basic statistics
            analysis_results.append("Basic Statistics:")
            analysis_results.append(df.describe().to_string())
            
            # 2. Correlation analysis for numerical columns
            numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
            if len(numerical_cols) > 1:
                analysis_results.append("\nCorrelation Analysis:")
                analysis_results.append(df[numerical_cols].corr().to_string())
            
            # 3. Categorical analysis
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            for col in categorical_cols:
                analysis_results.append(f"\nAnalysis of {col}:")
                analysis_results.append(df[col].value_counts().to_string())
            
            # 4. Time series analysis if applicable
            date_cols = df.select_dtypes(include=['datetime64']).columns
            for col in date_cols:
                analysis_results.append(f"\nTime Series Analysis of {col}:")
                analysis_results.append(df[col].dt.year.value_counts().sort_index().to_string())
            
            # Update state with analysis results
            state["data"]["analysis_results"] = "\n".join(analysis_results)
            state["messages"].append(AIMessage(content="Analysis complete. Key insights have been generated."))
            state["current_agent"] = "visualization"
            
        except Exception as e:
            state["messages"].append(AIMessage(content=f"Error during analysis: {str(e)}"))
        
        return state
    
    return analysis_agent

def create_analysis_workflow():
    # Create the workflow
    workflow = StateGraph(AnalysisState)
    
    # Add the analysis agent
    workflow.add_node("analysis_agent", create_analysis_agent())
    
    # Set the entry point
    workflow.set_entry_point("analysis_agent")
    
    # Add edges
    workflow.add_edge("analysis_agent", "analysis_agent")
    
    # Compile the workflow
    return workflow.compile() 