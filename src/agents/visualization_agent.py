from typing import Annotated, List, Tuple, TypedDict
from langgraph.graph import Graph, StateGraph
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

class VisualizationState(TypedDict):
    messages: Annotated[List[HumanMessage | AIMessage], "The messages in the conversation"]
    data: Annotated[pd.DataFrame, "The data to visualize"]
    plots: Annotated[List[str], "List of generated plot filenames"]

def create_visualization_agent():
    # Create the visualization agent
    llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0.2)
    
    # Enhanced visualization prompt with more specific instructions
    visualization_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert data visualization specialist. Your role is to create meaningful and insightful visualizations based on the data and analysis provided.

Key Responsibilities:
1. Understand the data structure and relationships
2. Choose appropriate visualization types based on the data characteristics
3. Create clear, informative, and aesthetically pleasing plots
4. Ensure visualizations tell a story and highlight key insights
5. Follow best practices for data visualization

Visualization Guidelines:
- For numerical data: Use histograms, box plots, or scatter plots
- For categorical data: Use bar charts, pie charts, or count plots
- For time series: Use line plots or area charts
- For correlations: Use heatmaps or scatter plots
- For distributions: Use histograms or density plots
- For comparisons: Use grouped bar charts or box plots

Always ensure:
- Clear titles and labels
- Appropriate color schemes
- Proper scaling and axis formatting
- Legends when necessary
- Readable text sizes
- Meaningful insights in the visualization

You will receive data and analysis results. Create visualizations that best represent the insights and patterns in the data."""),
        MessagesPlaceholder(variable_name="messages"),
        ("human", """Please analyze the following data and create appropriate visualizations:

Data Info:
{data_info}

Analysis Results:
{analysis_results}

Create visualizations that highlight the key insights and patterns in the data. For each visualization:
1. Explain why you chose this type of visualization
2. Describe what insights it reveals
3. Provide the Python code to create it
""")
    ])
    
    def visualization_agent(state: VisualizationState) -> VisualizationState:
        # Get data info and analysis results from messages
        data_info = state["data"].describe().to_string()
        analysis_results = "\n".join([msg.content for msg in state["messages"] if isinstance(msg, AIMessage)])
        
        # Create visualization chain
        chain = visualization_prompt | llm
        
        # Generate visualization instructions
        response = chain.invoke({
            "messages": state["messages"],
            "data_info": data_info,
            "analysis_results": analysis_results
        })
        
        # Parse the response and create visualizations
        plots = []
        try:
            # Create plots directory if it doesn't exist
            if not os.path.exists("plots"):
                os.makedirs("plots")
            
            # Generate timestamp for unique filenames
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create visualizations based on data characteristics
            df = state["data"]
            
            # 1. Numerical columns visualization
            numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
            if len(numerical_cols) > 0:
                # Correlation heatmap
                plt.figure(figsize=(12, 8))
                sns.heatmap(df[numerical_cols].corr(), annot=True, cmap='coolwarm', center=0)
                plt.title('Correlation Heatmap')
                heatmap_path = f"plots/correlation_heatmap_{timestamp}.png"
                plt.savefig(heatmap_path)
                plt.close()
                plots.append(heatmap_path)
                
                # Distribution plots for numerical columns
                for col in numerical_cols[:3]:  # Limit to first 3 numerical columns
                    plt.figure(figsize=(10, 6))
                    sns.histplot(data=df, x=col, kde=True)
                    plt.title(f'Distribution of {col}')
                    dist_path = f"plots/dist_{col}_{timestamp}.png"
                    plt.savefig(dist_path)
                    plt.close()
                    plots.append(dist_path)
            
            # 2. Categorical columns visualization
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            if len(categorical_cols) > 0:
                for col in categorical_cols[:3]:  # Limit to first 3 categorical columns
                    plt.figure(figsize=(10, 6))
                    sns.countplot(data=df, x=col)
                    plt.title(f'Count of {col}')
                    plt.xticks(rotation=45)
                    count_path = f"plots/count_{col}_{timestamp}.png"
                    plt.savefig(count_path)
                    plt.close()
                    plots.append(count_path)
            
            # 3. Time series visualization if date column exists
            date_cols = df.select_dtypes(include=['datetime64']).columns
            if len(date_cols) > 0:
                for col in date_cols:
                    plt.figure(figsize=(12, 6))
                    df.set_index(col).plot()
                    plt.title(f'Time Series Analysis')
                    plt.xticks(rotation=45)
                    time_path = f"plots/time_series_{timestamp}.png"
                    plt.savefig(time_path)
                    plt.close()
                    plots.append(time_path)
            
            # Add visualization results to messages
            state["messages"].append(AIMessage(content=f"Created {len(plots)} visualizations:\n" + "\n".join(plots)))
            state["plots"] = plots
            
        except Exception as e:
            state["messages"].append(AIMessage(content=f"Error creating visualizations: {str(e)}"))
        
        return state
    
    return visualization_agent

def create_visualization_workflow():
    # Create the workflow
    workflow = StateGraph(VisualizationState)
    
    # Add the visualization agent
    workflow.add_node("visualization_agent", create_visualization_agent())
    
    # Set the entry point
    workflow.set_entry_point("visualization_agent")
    
    # Add edges
    workflow.add_edge("visualization_agent", "visualization_agent")
    
    # Compile the workflow
    return workflow.compile() 