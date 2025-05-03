import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import Graph, StateGraph, END
from typing import TypedDict, Annotated, Sequence
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4-turbo-preview")

# Global variable to store the DataFrame
df = None

# Define state schema
class AgentState(TypedDict):
    messages: Annotated[Sequence[HumanMessage | AIMessage], "The messages in the conversation"]
    next: str

# Define tools for each agent
@tool
def load_data(file_path: str) -> str:
    """Load data from a CSV file and return basic information about it."""
    try:
        global df
        logger.info(f"Loading data from {file_path}")
        
        # Try different CSV loading methods
        try:
            # First try with standard settings
            df = pd.read_csv(file_path)
        except Exception as e:
            logger.warning(f"Standard CSV loading failed: {str(e)}")
            try:
                # Try with different separators
                df = pd.read_csv(file_path, sep=';')
            except Exception as e:
                logger.warning(f"Semicolon separator failed: {str(e)}")
                try:
                    # Try with different decimal points
                    df = pd.read_csv(file_path, decimal=',')
                except Exception as e:
                    logger.warning(f"Comma decimal failed: {str(e)}")
                    try:
                        # Try with different encoding
                        df = pd.read_csv(file_path, encoding='latin1')
                    except Exception as e:
                        logger.error(f"All CSV loading attempts failed: {str(e)}")
                        return f"Error loading data: {str(e)}"
        
        # Basic data cleaning
        df = df.dropna(how='all')  # Drop rows that are all NA
        df = df.dropna(axis=1, how='all')  # Drop columns that are all NA
        
        # Convert numeric columns
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    # First try standard numeric conversion
                    df[col] = pd.to_numeric(df[col])
                except (ValueError, TypeError):
                    try:
                        # Try replacing commas with dots for decimal numbers
                        df[col] = pd.to_numeric(df[col].str.replace(',', '.'))
                    except (ValueError, TypeError, AttributeError):
                        # If conversion fails, keep the column as is
                        continue
        
        # Log column information
        logger.info(f"Data loaded with shape: {df.shape}")
        logger.info(f"Columns: {', '.join(df.columns)}")
        logger.info(f"Data types:\n{df.dtypes}")
        
        # Create basic information string
        info = f"Data loaded successfully. Shape: {df.shape}\nColumns: {', '.join(df.columns)}\n"
        info += f"First few rows:\n{df.head().to_string()}\n\n"
        info += f"Data types:\n{df.dtypes}\n\n"
        info += f"Missing values:\n{df.isnull().sum()}\n\n"
        info += f"Sample values for each column:\n"
        
        # Add sample values for each column
        for col in df.columns:
            unique_values = df[col].unique()
            if len(unique_values) > 5:
                info += f"{col}: {', '.join(map(str, unique_values[:5]))}...\n"
            else:
                info += f"{col}: {', '.join(map(str, unique_values))}\n"
        
        logger.info("Data loading completed successfully")
        return info
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        return f"Error loading data: {str(e)}"

@tool
def analyze_data(data_info: str) -> str:
    """Analyze the data and return statistical insights."""
    try:
        global df
        if df is None:
            logger.error("No data loaded")
            return "No data loaded. Please load data first."
        
        logger.info("Starting data analysis")
        analysis_results = []
        
        # Basic statistics for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            analysis_results.append("\nNumeric Column Statistics:")
            analysis_results.append(df[numeric_cols].describe().to_string())
        
        # Correlation analysis
        if len(numeric_cols) > 1:
            analysis_results.append("\nCorrelation Matrix:")
            corr_matrix = df[numeric_cols].corr()
            analysis_results.append(corr_matrix.to_string())
        
        # Categorical analysis
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            analysis_results.append("\nCategorical Column Analysis:")
            for col in categorical_cols:
                value_counts = df[col].value_counts()
                analysis_results.append(f"\n{col}:\n{value_counts}")
        
        # Time series analysis if date column exists
        date_cols = df.select_dtypes(include=['datetime64']).columns
        if len(date_cols) > 0:
            analysis_results.append("\nTime Series Analysis:")
            for col in date_cols:
                analysis_results.append(f"\n{col}:\n{df[col].describe()}")
        
        logger.info("Data analysis completed")
        return "\n".join(analysis_results)
    except Exception as e:
        logger.error(f"Error analyzing data: {str(e)}")
        return f"Error analyzing data: {str(e)}"

@tool
def create_visualizations(analysis_results: str) -> str:
    """Create appropriate visualizations based on the data types and content."""
    global df
    if df is None:
        return "No data loaded. Please load data first."

    plots_dir = os.path.join(os.getcwd(), "plots")
    os.makedirs(plots_dir, exist_ok=True)
    results = []

    # 1. Correlation Heatmap (always keep)
    numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns if col.lower() != "id" and df[col].nunique() > 5]
    if len(numeric_cols) > 1:
        plt.figure(figsize=(12, 8))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Heatmap of Numeric Variables')
        heatmap_path = os.path.join(plots_dir, "correlation_heatmap.png")
        plt.savefig(heatmap_path)
        plt.close()
        results.append("Created correlation heatmap")

    # 2. Most variable numeric column (excluding id-like columns)
    if numeric_cols:
        # Choose the numeric column with the highest variance
        var_col = df[numeric_cols].var().sort_values(ascending=False).index[0]
        plt.figure(figsize=(10, 6))
        sns.histplot(df[var_col], kde=True)
        plt.title(f'Distribution of {var_col}')
        plt.xlabel(var_col)
        plt.ylabel('Count')
        plot_path = os.path.join(plots_dir, f"{var_col}_distribution.png")
        plt.savefig(plot_path)
        plt.close()
        results.append(f"Created distribution plot for {var_col}")

    # 3. Most balanced categorical column (2–10 unique values, not id-like)
    cat_cols = [col for col in df.select_dtypes(include=['object', 'category']).columns if 2 <= df[col].nunique() <= 10 and "id" not in col.lower()]
    if cat_cols:
        # Choose the categorical column with the most balanced value counts
        cat_col = sorted(cat_cols, key=lambda c: -df[c].value_counts().min())[0]
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df, x=cat_col)
        plt.title(f'Distribution of {cat_col}')
        plt.xlabel(cat_col)
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plot_path = os.path.join(plots_dir, f"{cat_col}_distribution.png")
        plt.savefig(plot_path)
        plt.close()
        results.append(f"Created distribution plot for {cat_col}")

    # 4. Boxplot: Most variable numeric vs. most balanced categorical (if both exist)
    if numeric_cols and cat_cols:
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x=cat_col, y=var_col)
        plt.title(f'{var_col} by {cat_col}')
        plt.xlabel(cat_col)
        plt.ylabel(var_col)
        plt.xticks(rotation=45)
        plot_path = os.path.join(plots_dir, f"{var_col}_by_{cat_col}_boxplot.png")
        plt.savefig(plot_path)
        plt.close()
        results.append(f"Created box plot for {var_col} by {cat_col}")

    return "\n".join(results)

# Define agent prompts
data_wrangler_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a data wrangling expert. Your job is to load and prepare data for analysis.
    You should:
    1. Load the data from the provided file
    2. Clean the data by handling missing values and data type conversions
    3. Provide basic information about the dataset
    4. Identify any potential data quality issues
    """),
    MessagesPlaceholder(variable_name="messages"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

analyst_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a data analyst. Your job is to analyze data and extract insights.
    You should:
    1. Perform statistical analysis on numeric columns
    2. Analyze correlations between variables
    3. Examine categorical data distributions
    4. Identify trends and patterns
    5. Highlight any interesting findings or anomalies
    """),
    MessagesPlaceholder(variable_name="messages"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

visualizer_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a data visualization expert. Your job is to create meaningful visualizations based on the data and analysis.
    You should:
    1. Create appropriate visualizations using matplotlib and seaborn
    2. Save all plots to the 'plots' directory
    3. Use proper figure sizes and labels
    4. Ensure plots are informative and well-formatted
    5. Return a summary of the visualizations created
    """),
    MessagesPlaceholder(variable_name="messages"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Create agents
data_wrangler = create_openai_functions_agent(llm, [load_data], data_wrangler_prompt)
analyst = create_openai_functions_agent(llm, [analyze_data], analyst_prompt)
visualizer = create_openai_functions_agent(llm, [create_visualizations], visualizer_prompt)

# Create agent executors
data_wrangler_executor = AgentExecutor(agent=data_wrangler, tools=[load_data])
analyst_executor = AgentExecutor(agent=analyst, tools=[analyze_data])
visualizer_executor = AgentExecutor(agent=visualizer, tools=[create_visualizations])

# Define the graph
def create_workflow():
    logger.info("Creating workflow")
    # Create the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("data_wrangler", data_wrangler_executor)
    workflow.add_node("analyst", analyst_executor)
    workflow.add_node("visualizer", visualizer_executor)
    
    # Add edges
    workflow.add_edge("data_wrangler", "analyst")
    workflow.add_edge("analyst", "visualizer")
    workflow.add_edge("visualizer", END)
    
    # Set entry point
    workflow.set_entry_point("data_wrangler")
    
    # Compile the graph
    compiled_workflow = workflow.compile()
    logger.info("Workflow created and compiled successfully")
    return compiled_workflow

# Streamlit interface
def main():
    st.title("Multi-Agent Data Analysis System")
    
    # File upload
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        # Save the uploaded file
        file_path = "temp_data.csv"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Display plots if they exist
                plots_dir = os.path.join(os.getcwd(), "plots")
                if message["role"] == "assistant" and os.path.exists(plots_dir):
                    plot_files = [f for f in os.listdir(plots_dir) if f.endswith(".png")]
                    if plot_files:
                        st.write("Generated Visualizations:")
                        for plot_file in plot_files:
                            st.image(os.path.join(plots_dir, plot_file))
        
        # Chat input
        if prompt := st.chat_input("Ask a question about the data"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            try:
                # Create workflow
                logger.info("Creating workflow")
                workflow = create_workflow()
                
                # Run workflow
                logger.info("Running workflow")
                initial_state = {
                    "messages": [HumanMessage(content=f"Please analyze this data: {file_path}. {prompt}")],
                    "next": "data_wrangler"
                }
                logger.info(f"Initial state: {initial_state}")
                
                # Run each step manually to ensure proper execution
                logger.info("Running data wrangler")
                wrangler_result = data_wrangler_executor.invoke(initial_state)
                logger.info(f"Data wrangler result: {wrangler_result}")
                
                # Display data wrangler results
                with st.chat_message("assistant"):
                    st.markdown("### Data Overview")
                    st.markdown(wrangler_result["output"])
                
                logger.info("Running analyst")
                analyst_state = {**wrangler_result, "next": "analyst"}
                analyst_result = analyst_executor.invoke(analyst_state)
                logger.info(f"Analyst result: {analyst_result}")
                
                # Display analyst results
                with st.chat_message("assistant"):
                    st.markdown("### Analysis Results")
                    st.markdown(analyst_result["output"])
                
                # Create visualizations
                logger.info("Creating visualizations")
                visualizer_result = create_visualizations(analyst_result["output"])
                logger.info("Visualizations created successfully")
                
                # Display visualizations
                with st.chat_message("assistant"):
                    st.markdown("### Visualizations")
                    plots_dir = os.path.join(os.getcwd(), "plots")
                    if os.path.exists(plots_dir):
                        plot_files = [f for f in os.listdir(plots_dir) if f.endswith(".png")]
                        if plot_files:
                            for plot_file in plot_files:
                                st.image(os.path.join(plots_dir, plot_file))
                        else:
                            st.warning("No visualizations were generated")
                    else:
                        st.warning(f"No plots directory found at: {plots_dir}")
                
                # Add messages to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"### Data Overview\n{wrangler_result['output']}\n\n### Analysis Results\n{analyst_result['output']}\n\n### Visualizations\n{visualizer_result}"
                })
                
                # Clean up
                try:
                    os.remove(file_path)
                    if os.path.exists(plots_dir):
                        for plot_file in os.listdir(plots_dir):
                            os.remove(os.path.join(plots_dir, plot_file))
                        os.rmdir(plots_dir)
                except Exception as e:
                    logger.error(f"Error during cleanup: {str(e)}")
                    st.warning(f"Error during cleanup: {str(e)}")
            
            except Exception as e:
                logger.error(f"Error in workflow execution: {str(e)}")
                st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 