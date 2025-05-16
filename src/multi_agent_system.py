from typing import Annotated, List, Dict, Any
from langgraph.graph import Graph, StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_anthropic import ChatAnthropic  # Changed from OpenAI to Anthropic
import pandas as pd
import numpy as np
import logging
import json
import os
import hashlib
from io import StringIO
from datetime import datetime
from src.tools.visualization_tool import execute_visualization_code

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create a memory directory if it doesn't exist
if not os.path.exists("memory"):
    os.makedirs("memory", exist_ok=True)


# Define the state schema
class AgentState:
    messages: List[Any]
    data: Dict[str, Any]
    current_agent: str  # Which agent is currently active
    memory: Dict[str, Any]  # Memory of previous analyses and visualizations
    thought_process: Dict[str, str]  # Store agent reasoning

    def __init__(self):
        self.messages = []
        self.data = {}
        self.current_agent = "wrangler"
        self.memory = {}
        self.thought_process = {}


class MultiAgentSystem:
    def __init__(self, model_name="claude-3-haiku-20240307"):  # Change to Claude model
        # Initialize Claude LLM instead of OpenAI
        self.llm = ChatAnthropic(
            model=model_name,
            temperature=0.2,
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")  # Use Anthropic API key
        )
        self.memory_file = os.path.join("memory", "agent_memory.json")
        self.load_memory()

    def load_memory(self):
        """Load memory from JSON file"""
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, 'r') as f:
                    self.memory = json.load(f)
            else:
                self.memory = {"datasets": {}, "analyses": {},
                               "visualizations": {}}
        except Exception as e:
            logger.error(f"Error loading memory: {str(e)}")
            self.memory = {"datasets": {}, "analyses": {},
                           "visualizations": {}}

    def save_memory(self):
        """Save memory to JSON file"""
        try:
            with open(self.memory_file, 'w') as f:
                json.dump(self.memory, f, indent=2)
                logger.info("Memory saved successfully")
        except Exception as e:
            logger.error(f"Error saving memory: {str(e)}")

    def create_dataset_fingerprint(self, df):
        """Create a fingerprint for a dataset based on its structure"""
        cols = list(df.columns)
        dtypes = df.dtypes.astype(str).to_dict()
        shape = df.shape

        # Get some basic stats
        stats = {}
        for col in df.select_dtypes(include=['float64', 'int64']).columns[
                   :5]:  # First 5 numeric cols
            try:
                stats[col] = {
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                }
            except:
                pass

        fingerprint_dict = {
            'columns': cols[:10],  # First 10 columns
            'dtypes': dtypes,
            'shape': shape,
            'stats': stats
        }

        # Hash this dictionary to create a unique ID
        fingerprint_str = json.dumps(fingerprint_dict, sort_keys=True)
        return hashlib.md5(fingerprint_str.encode()).hexdigest()

    def find_similar_dataset(self, df, threshold=0.7):
        """Find if a similar dataset has been analyzed before"""
        current_id = self.create_dataset_fingerprint(df)

        # Exact match check
        if current_id in self.memory["datasets"]:
            logger.info(f"Found exact match for dataset: {current_id}")
            return current_id

        return None  # For now, only return exact matches

    def record_dataset(self, df):
        """Record a dataset in memory"""
        dataset_id = self.create_dataset_fingerprint(df)

        # Store basic metadata
        self.memory["datasets"][dataset_id] = {
            'columns': list(df.columns),
            'shape': df.shape,
            'timestamp': datetime.now().isoformat()
        }

        self.save_memory()
        return dataset_id

    def record_analysis(self, dataset_id, analysis_results):
        """Record analysis for a dataset"""
        if dataset_id not in self.memory["analyses"]:
            self.memory["analyses"][dataset_id] = []

        self.memory["analyses"][dataset_id].append({
            'timestamp': datetime.now().isoformat(),
            'results': analysis_results
        })

        self.save_memory()

    def record_visualization(self, dataset_id, viz_code, plot_files):
        """Record visualizations for a dataset"""
        if dataset_id not in self.memory["visualizations"]:
            self.memory["visualizations"][dataset_id] = []

        self.memory["visualizations"][dataset_id].append({
            'timestamp': datetime.now().isoformat(),
            'code': viz_code,
            'files': plot_files
        })

        self.save_memory()

    def get_memory_context(self, dataset_id):
        """Get memory context for a dataset"""
        context = ""

        if dataset_id in self.memory["datasets"]:
            context += f"This dataset has {self.memory['datasets'][dataset_id]['shape'][0]} rows and {self.memory['datasets'][dataset_id]['shape'][1]} columns.\n"

        if dataset_id in self.memory["analyses"]:
            context += "\nPrevious analyses found:\n"
            for analysis in self.memory["analyses"][dataset_id][
                            -1:]:  # Just the most recent
                context += f"- Analysis from {analysis['timestamp']}: {analysis['results'][:300]}...\n"

        if dataset_id in self.memory["visualizations"]:
            context += "\nPrevious visualizations created:\n"
            for viz in self.memory["visualizations"][dataset_id][
                       -3:]:  # Last 3
                context += f"- Visualization from {viz['timestamp']} with {len(viz['files'])} plots\n"

        return context

    def data_wrangling(self, state):
        """Data wrangling agent"""
        logger.info("Running data wrangling agent")

        try:
            # Get the data file from state
            df = None
            file_path = state.data.get("file_path")

            if file_path:
                # Try to load the data with appropriate error handling
                try:
                    df = pd.read_csv(file_path)
                    logger.info(f"Successfully loaded data from {file_path}")
                except Exception as e:
                    logger.error(
                        f"Error loading CSV with default settings: {str(e)}")
                    try:
                        # Try alternate settings
                        df = pd.read_csv(file_path, sep=';', encoding='latin1')
                        logger.info(f"Loaded data with alternate settings")
                    except Exception as e2:
                        logger.error(f"All loading attempts failed: {str(e2)}")
                        state.messages.append(AIMessage(
                            content=f"Error loading data: {str(e2)}"))
                        return state
            else:
                logger.error("No file path provided")
                state.messages.append(AIMessage(
                    content="No file path provided for data wrangling."))
                return state

            # Check memory for similar datasets
            dataset_id = self.find_similar_dataset(df)
            memory_context = ""

            if dataset_id:
                memory_context = self.get_memory_context(dataset_id)
                logger.info(f"Found similar dataset in memory: {dataset_id}")
            else:
                # Record new dataset
                dataset_id = self.record_dataset(df)
                logger.info(f"Recorded new dataset in memory: {dataset_id}")

            state.data["dataset_id"] = dataset_id

            # Create data info summary for the agent
            buffer = StringIO()
            df.info(buf=buffer)
            data_info = buffer.getvalue()

            # Identify data issues
            missing_values = df.isnull().sum().sum()
            duplicates = df.duplicated().sum()

            issues = []
            if missing_values > 0:
                issues.append(f"Missing values detected: {missing_values}")
            if duplicates > 0:
                issues.append(f"Duplicate rows detected: {duplicates}")

            # Create prompt for data wrangling
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert data wrangling specialist. Your job is to clean and prepare data for analysis.

Your task is to:
1. Clean the data by handling missing values, duplicates, and outliers
2. Convert data types appropriately
3. Provide a summary of the data
4. Explain any data quality issues found and how they were addressed

Memory Information:
{memory_context}

First, explain your thought process about how you'll clean this data.
Then perform the necessary data wrangling steps."""),
                MessagesPlaceholder(variable_name="messages"),
                ("human", """Please process the following data:

Data Info:
{data_info}

Issues Detected:
{issues}

Please clean and prepare this data for analysis."""),
            ])

            # Generate wrangling instructions
            chain = prompt | self.llm

            response = chain.invoke({
                "messages": state.messages,
                "data_info": data_info,
                "issues": "\n".join(issues),
                "memory_context": memory_context
            })

            # Save thought process
            state.thought_process["wrangler"] = response.content

            # Perform data wrangling

            # 1. Handle missing values
            for col in df.columns:
                if df[col].dtype in ['float64', 'int64']:
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(df[col].mode()[0] if not df[
                        col].mode().empty else "Unknown")

            # 2. Remove duplicates
            df = df.drop_duplicates()

            # 3. Convert data types intelligently
            for col in df.columns:
                if df[col].dtype == 'object':
                    # Check if column is categorical (few unique values)
                    unique_ratio = df[col].nunique() / len(df)

                    if unique_ratio < 0.05:  # If less than 5% unique values, treat as categorical
                        continue

                    # Try numeric conversion
                    try:
                        df[col] = pd.to_numeric(df[col])
                    except:
                        # Not numeric, leave as is
                        pass

            # Store processed data
            state.data["processed_data"] = df

            # Add wrangling completion message
            state.messages.append(AIMessage(
                content=f"Data wrangling complete. Processed {len(df)} rows and {len(df.columns)} columns. Removed {duplicates} duplicates and handled {missing_values} missing values."))

            # Move to analysis stage
            state.current_agent = "analyst"

            return state

        except Exception as e:
            logger.error(f"Error in data wrangling: {str(e)}")
            state.messages.append(
                AIMessage(content=f"Error during data wrangling: {str(e)}"))
            return state

    def data_analysis(self, state):
        """Data analysis agent"""
        logger.info("Running data analysis agent")

        try:
            # Get processed data
            df = state.data.get("processed_data")
            dataset_id = state.data.get("dataset_id")

            if df is None:
                logger.error("No processed data available for analysis")
                state.messages.append(AIMessage(
                    content="No processed data available for analysis."))
                return state

            # Get memory context
            memory_context = ""
            if dataset_id:
                memory_context = self.get_memory_context(dataset_id)
                logger.info(
                    f"Retrieved memory context for dataset: {dataset_id}")

            # Create data summary
            data_summary = df.describe().to_string()

            # Add correlation matrix if sufficient numeric columns
            num_cols = df.select_dtypes(include=['float64', 'int64']).columns
            if len(num_cols) > 1:
                corr_matrix = df[num_cols].corr()
                data_summary += "\n\nCorrelation Matrix:\n" + corr_matrix.round(
                    2).to_string()

            # Create analysis prompt
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert data analyst. Your job is to analyze data and extract meaningful insights.

Your task is to:
1. Perform statistical analysis on the data
2. Identify patterns, trends, and relationships
3. Extract key insights and findings
4. Suggest areas for further investigation or visualization

Memory Information:
{memory_context}

First, explain your thought process about how you'll analyze this data.
Then perform a comprehensive analysis to extract meaningful insights."""),
                MessagesPlaceholder(variable_name="messages"),
                ("human", """Please analyze the following data:

Data Summary:
{data_summary}

Please provide a comprehensive analysis with key insights."""),
            ])

            # Generate analysis
            chain = prompt | self.llm

            response = chain.invoke({
                "messages": state.messages,
                "data_summary": data_summary,
                "memory_context": memory_context
            })

            # Save thought process
            state.thought_process["analyst"] = response.content

            # Perform comprehensive analysis
            analysis_results = []

            # 1. Basic statistics
            analysis_results.append("# Basic Statistics")
            analysis_results.append(df.describe().to_string())

            # 2. Correlation analysis for numerical columns
            if len(num_cols) > 1:
                analysis_results.append("\n# Correlation Analysis")
                analysis_results.append(
                    df[num_cols].corr().round(2).to_string())

                # Identify strong correlations
                strong_corrs = []
                corr_matrix = df[num_cols].corr()
                for i in range(len(corr_matrix.columns)):
                    for j in range(i + 1, len(corr_matrix.columns)):
                        col1 = corr_matrix.columns[i]
                        col2 = corr_matrix.columns[j]
                        corr = corr_matrix.iloc[i, j]
                        if abs(corr) > 0.7:
                            strong_corrs.append(
                                f"{col1} and {col2}: {corr:.2f}")

                if strong_corrs:
                    analysis_results.append("\n# Strong Correlations")
                    analysis_results.append("\n".join(strong_corrs))

            # 3. Categorical analysis
            cat_cols = df.select_dtypes(include=['object']).columns
            if len(cat_cols) > 0:
                analysis_results.append("\n# Categorical Analysis")
                for col in cat_cols:
                    analysis_results.append(f"\n## {col} Distribution")
                    analysis_results.append(
                        df[col].value_counts().head(10).to_string())

            # Join analysis results
            analysis_text = "\n".join(analysis_results)

            # Store analysis results
            state.data["analysis_results"] = analysis_text

            # Record in memory
            if dataset_id:
                self.record_analysis(dataset_id, analysis_text)
                logger.info(
                    f"Recorded analysis for dataset {dataset_id} in memory")

            # Add analysis completion message
            state.messages.append(AIMessage(
                content="Analysis complete. Key insights have been generated."))

            # Move to visualization stage
            state.current_agent = "visualizer"

            return state

        except Exception as e:
            logger.error(f"Error in data analysis: {str(e)}")
            state.messages.append(
                AIMessage(content=f"Error during data analysis: {str(e)}"))
            return state

    def data_visualization(self, state):
        """Data visualization agent"""
        logger.info("Running data visualization agent")

        try:
            # Get processed data and analysis results
            df = state.data.get("processed_data")
            analysis_results = state.data.get("analysis_results", "")
            dataset_id = state.data.get("dataset_id")

            if df is None:
                logger.error("No processed data available for visualization")
                state.messages.append(AIMessage(
                    content="No processed data available for visualization."))
                return state

            # Get memory context
            memory_context = ""
            if dataset_id:
                memory_context = self.get_memory_context(dataset_id)
                logger.info(
                    f"Retrieved memory context for dataset: {dataset_id}")

            # Create a data overview
            data_overview = f"Data Shape: {df.shape}\n\nColumns: {', '.join(df.columns)}\n\nDatatypes:\n{df.dtypes.to_string()}"

            # Create visualization prompt
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert data visualization specialist. Your job is to create meaningful visualizations that reveal insights in the data.

Your task is to:
1. Identify which aspects of the data would benefit most from visualization
2. Create appropriate visualizations using Plotly Express (px)
3. Ensure visualizations are informative, clear, and properly labeled
4. Explain what insights each visualization reveals

IMPORTANT INSTRUCTIONS FOR CODE GENERATION:
1. Generate Python code using Plotly Express (px) for visualizations
2. Make sure your code is properly indented and has no syntax errors
3. Do not use plt.show() or fig.show() - the code will be executed separately
4. Check if columns you reference actually exist in the dataframe
5. Only select numeric columns for scatter plots, bar charts, etc.
6. Use color_discrete_sequence=px.colors.qualitative.Plotly for colorful bar charts
7. Use color_continuous_scale='RdBu_r' for heatmaps
8. Make visualizations interactive with proper titles, labels, and legends
9. Generate 3-5 different visualizations that best tell the data story

Memory Information:
{memory_context}

First, explain your thought process about what visualizations would be most valuable.
Then provide Python code for each visualization, with an explanation of what it shows."""),
                MessagesPlaceholder(variable_name="messages"),
                ("human", """Please create visualizations for the following data:

Data Overview:
{data_overview}

Analysis Results:
{analysis_results}

Please provide Python code for 3-5 visualizations that best illustrate the key insights in this data."""),
            ])

            # Generate visualization instructions
            chain = prompt | self.llm

            response = chain.invoke({
                "messages": state.messages,
                "data_overview": data_overview,
                "analysis_results": analysis_results,
                "memory_context": memory_context
            })

            # Save thought process
            state.thought_process["visualizer"] = response.content

            # Extract Python code blocks from the response
            import re
            code_blocks = re.findall(r'```python(.*?)```', response.content,
                                     re.DOTALL)

            # Create visualizations
            all_plots = []

            for i, code_block in enumerate(code_blocks):
                logger.info(f"Executing visualization code block {i + 1}")

                # Clean up the code
                code = code_block.strip()

                # Execute the code and get the results
                output, plot_files = execute_visualization_code(df, code)

                if plot_files:
                    all_plots.extend(plot_files)
                    logger.info(
                        f"Successfully created {len(plot_files)} plots")
                else:
                    logger.warning(f"No plots created for code block {i + 1}")

            # Store visualization results
            state.data["plots"] = all_plots

            # Record in memory
            if dataset_id and all_plots:
                self.record_visualization(dataset_id, code_blocks, all_plots)
                logger.info(
                    f"Recorded visualizations for dataset {dataset_id} in memory")

            # Add visualization completion message
            if all_plots:
                state.messages.append(AIMessage(
                    content=f"Visualization complete. Created {len(all_plots)} visualizations."))
            else:
                state.messages.append(AIMessage(
                    content="No visualizations could be created. Please check the data and try again."))

            # End workflow
            state.current_agent = "done"

            return state

        except Exception as e:
            logger.error(f"Error in data visualization: {str(e)}")
            state.messages.append(AIMessage(
                content=f"Error during data visualization: {str(e)}"))
            return state

    def run_workflow(self, query, file_path):
        """Run the complete workflow"""
        # Initialize state
        state = AgentState()
        state.messages = [HumanMessage(content=query)]
        state.data = {"file_path": file_path}

        # Run agents in sequence
        while state.current_agent != "done":
            current_agent = state.current_agent

            if current_agent == "wrangler":
                state = self.data_wrangling(state)
            elif current_agent == "analyst":
                state = self.data_analysis(state)
            elif current_agent == "visualizer":
                state = self.data_visualization(state)
            else:
                logger.error(f"Unknown agent: {current_agent}")
                state.messages.append(AIMessage(
                    content=f"Error: Unknown agent '{current_agent}'"))
                state.current_agent = "done"

        return state