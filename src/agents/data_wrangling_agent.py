from typing import Annotated, List, TypedDict
from langgraph.graph import Graph, StateGraph
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
import pandas as pd
from langchain.callbacks.manager import CallbackManager

class DataWranglingState(TypedDict):
    messages: Annotated[List[HumanMessage | AIMessage], "The messages in the conversation"]
    data: Annotated[dict, "The data being processed"]
    current_agent: Annotated[str, "The current agent processing the data"]

def create_data_wrangling_agent(callback_manager=None):
    # Create the data wrangling agent
    llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0.2)
    
    # Enhanced data wrangling prompt with specific instructions
    wrangling_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert data wrangling specialist. Your role is to clean, preprocess, and prepare data for analysis.

Key Responsibilities:
1. Data Cleaning:
   - Handle missing values appropriately
   - Remove duplicates
   - Fix inconsistent data formats
   - Handle outliers
   - Standardize data types

2. Data Transformation:
   - Convert data types as needed
   - Create derived features
   - Normalize or scale numerical data
   - Encode categorical variables
   - Handle datetime formats

3. Data Validation:
   - Check for data quality issues
   - Verify data consistency
   - Ensure data meets analysis requirements
   - Document data transformations

4. Data Preparation:
   - Split data if needed
   - Create appropriate data structures
   - Prepare data for specific analysis types
   - Ensure data is in the correct format for visualization

You will receive raw data and should return cleaned, processed data ready for analysis."""),
        MessagesPlaceholder(variable_name="messages"),
        ("human", """Please process the following data:

Data Info:
{data_info}

Current Issues:
{current_issues}

Please perform necessary data wrangling steps and explain your decisions."""),
    ])
    
    def data_wrangling_agent(state: DataWranglingState) -> DataWranglingState:
        try:
            # Get the data file from the state
            data_file = state["data"]["file"]
            
            # Read the data based on file type
            if data_file.name.endswith('.csv'):
                df = pd.read_csv(data_file)
            elif data_file.name.endswith('.xlsx'):
                df = pd.read_excel(data_file)
            else:
                raise ValueError("Unsupported file format")
            
            # Get data info
            data_info = df.info()
            current_issues = []
            
            # Check for common data issues
            if df.isnull().sum().sum() > 0:
                current_issues.append("Missing values detected")
            if df.duplicated().sum() > 0:
                current_issues.append("Duplicates detected")
            
            # Create data wrangling chain
            chain = wrangling_prompt | llm
            
            # Generate data wrangling instructions
            response = chain.invoke({
                "messages": state["messages"],
                "data_info": data_info,
                "current_issues": "\n".join(current_issues)
            })
            
            # Perform data wrangling
            # 1. Handle missing values
            for col in df.columns:
                if df[col].dtype in ['float64', 'int64']:
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(df[col].mode()[0])
            
            # 2. Remove duplicates
            df = df.drop_duplicates()
            
            # 3. Convert data types
            for col in df.columns:
                if df[col].dtype == 'object':
                    try:
                        df[col] = pd.to_datetime(df[col])
                    except:
                        pass
            
            # Update state with processed data
            state["data"]["processed_data"] = df
            state["messages"].append(AIMessage(content=f"Data wrangling complete. Processed {len(df)} rows and {len(df.columns)} columns."))
            state["current_agent"] = "analysis"
            
        except Exception as e:
            state["messages"].append(AIMessage(content=f"Error during data wrangling: {str(e)}"))
        
        return state
    
    return data_wrangling_agent

def create_data_wrangling_workflow():
    # Create the workflow
    workflow = StateGraph(DataWranglingState)
    
    # Add the data wrangling agent
    workflow.add_node("data_wrangling_agent", create_data_wrangling_agent())
    
    # Set the entry point
    workflow.set_entry_point("data_wrangling_agent")
    
    # Add edges
    workflow.add_edge("data_wrangling_agent", "data_wrangling_agent")
    
    # Compile the workflow
    return workflow.compile() 