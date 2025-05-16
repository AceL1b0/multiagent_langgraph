import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
import logging
import json
import time
import uuid
from typing import List, Dict, Any
from src.multi_agent_system import MultiAgentSystem

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="Multi-Agent Data Analysis System",
    page_icon="📊",
    layout="wide"
)


def get_file_extension(file):
    """Get the extension of a file."""
    return os.path.splitext(file.name)[1].lower()


def save_uploaded_file(uploaded_file):
    """Save an uploaded file to disk."""
    file_path = os.path.join("uploads", uploaded_file.name)
    os.makedirs("uploads", exist_ok=True)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return file_path


def initialize_session_state():
    """Initialize session state variables if they don't exist."""
    if "agent_system" not in st.session_state:
        st.session_state.agent_system = MultiAgentSystem()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = {}

    if "processing" not in st.session_state:
        st.session_state.processing = False


def main():
    """Main application."""
    st.title("Multi-Agent Data Analysis System with Memory")

    # Initialize session state
    initialize_session_state()

    # Create side panel for file upload and memory status
    with st.sidebar:
        st.header("Upload Data")
        uploaded_file = st.file_uploader("Upload a CSV file for analysis",
                                         type=["csv"])

        st.header("Memory Status")
        memory = st.session_state.agent_system.memory
        st.write(f"Datasets in memory: {len(memory.get('datasets', {}))}")
        st.write(f"Analyses stored: {len(memory.get('analyses', {}))}")
        st.write(
            f"Visualizations stored: {len(memory.get('visualizations', {}))}")

        if st.button("Clear Memory"):
            st.session_state.agent_system.memory = {"datasets": {},
                                                    "analyses": {},
                                                    "visualizations": {}}
            st.session_state.agent_system.save_memory()
            st.success("Memory cleared successfully!")

    # Create tab layout
    tab1, tab2, tab3 = st.tabs(
        ["Data Analysis", "Chat Interface", "Debug View"])

    with tab1:
        col1, col2 = st.columns([2, 1])

        with col1:
            st.header("Data Analysis")

            if uploaded_file:
                file_path = save_uploaded_file(uploaded_file)

                if st.button(
                        "Run Analysis") and not st.session_state.processing:
                    with st.spinner("Running multi-agent analysis..."):
                        st.session_state.processing = True

                        try:
                            # Run the workflow
                            result_state = st.session_state.agent_system.run_workflow(
                                f"Please analyze this data: {uploaded_file.name}",
                                file_path
                            )

                            # Store results
                            st.session_state.analysis_results = {
                                "messages": result_state.messages,
                                "data": result_state.data,
                                "thought_process": result_state.thought_process
                            }

                            # Add final result to chat history
                            for msg in result_state.messages:
                                if msg not in st.session_state.chat_history:
                                    st.session_state.chat_history.append(msg)

                            st.success("Analysis complete!")

                        except Exception as e:
                            st.error(f"Error running analysis: {str(e)}")
                            logger.error(f"Analysis error: {str(e)}")

                        st.session_state.processing = False

                # Display previous analysis results if available
                if "analysis_results" in st.session_state and st.session_state.analysis_results:
                    results = st.session_state.analysis_results

                    # Create tabs for different stages
                    analysis_tab1, analysis_tab2, analysis_tab3 = st.tabs(
                        ["Data Wrangling", "Analysis", "Visualization"])

                    with analysis_tab1:
                        st.subheader("Data Wrangling Results")

                        # Show thought process
                        if "thought_process" in results and "wrangler" in \
                                results["thought_process"]:
                            with st.expander("Data Wrangler's Thought Process",
                                             expanded=False):
                                st.markdown(
                                    results["thought_process"]["wrangler"])

                        # Show data preview
                        if "data" in results and "processed_data" in results[
                            "data"]:
                            processed_data = results["data"]["processed_data"]
                            st.subheader("Processed Data Preview")
                            st.dataframe(processed_data.head(10))

                            # Show data info
                            st.subheader("Data Info")
                            from io import StringIO
                            buffer = StringIO()
                            processed_data.info(buf=buffer)
                            st.text(buffer.getvalue())

                    with analysis_tab2:
                        st.subheader("Analysis Results")

                        # Show thought process
                        if "thought_process" in results and "analyst" in \
                                results["thought_process"]:
                            with st.expander("Data Analyst's Thought Process",
                                             expanded=False):
                                st.markdown(
                                    results["thought_process"]["analyst"])

                        # Show analysis results
                        if "data" in results and "analysis_results" in results[
                            "data"]:
                            st.subheader("Key Insights")
                            st.text(results["data"]["analysis_results"])

                    with analysis_tab3:
                        st.subheader("Visualizations")

                        # Show thought process
                        if "thought_process" in results and "visualizer" in \
                                results["thought_process"]:
                            with st.expander(
                                    "Data Visualizer's Thought Process",
                                    expanded=False):
                                st.markdown(
                                    results["thought_process"]["visualizer"])

                        # Show visualizations
                        if "data" in results and "plots" in results["data"] and \
                                results["data"]["plots"]:
                            plots = results["data"]["plots"]
                            st.subheader(
                                f"Generated Visualizations ({len(plots)})")

                            for i, plot_file in enumerate(plots):
                                plot_path = os.path.join("plots", plot_file)
                                if os.path.exists(
                                        plot_path) and plot_path.endswith(
                                        ".html"):
                                    st.subheader(f"Visualization {i + 1}")
                                    with open(plot_path, 'r') as f:
                                        html_content = f.read()
                                        st.components.v1.html(html_content,
                                                              height=600)
                        else:
                            st.info("No visualizations were generated.")

        with col2:
            st.header("Data Summary")

            if uploaded_file:
                # Show data preview
                try:
                    df = pd.read_csv(file_path)
                    st.subheader("Raw Data Preview")
                    st.dataframe(df.head(5))

                    st.subheader("Basic Stats")
                    st.write(f"Rows: {df.shape[0]}")
                    st.write(f"Columns: {df.shape[1]}")

                    missing_values = df.isnull().sum().sum()
                    st.write(f"Missing Values: {missing_values}")

                    duplicates = df.duplicated().sum()
                    st.write(f"Duplicate Rows: {duplicates}")

                    # Show column types
                    st.subheader("Column Types")
                    type_counts = df.dtypes.value_counts()
                    for dtype, count in type_counts.items():
                        st.write(f"{dtype}: {count} columns")

                except Exception as e:
                    st.error(f"Error loading data: {str(e)}")

    with tab2:
        st.header("Chat Interface")

        # Display chat history
        for message in st.session_state.chat_history:
            if hasattr(message, "content"):
                if message.type == "human":
                    with st.chat_message("human"):
                        st.markdown(message.content)
                else:
                    with st.chat_message("assistant"):
                        st.markdown(message.content)

        # Chat input
        prompt = st.chat_input("Ask a question about your data...")
        if prompt:
            # Add user message to chat history
            from langchain_core.messages import HumanMessage
            user_message = HumanMessage(content=prompt)
            st.session_state.chat_history.append(user_message)

            # Display user message
            with st.chat_message("human"):
                st.markdown(prompt)

            # Get AI response
            with st.spinner("Thinking..."):
                try:
                    # Create context
                    context = "I can help answer questions about your data analysis."

                    if "analysis_results" in st.session_state and st.session_state.analysis_results:
                        # Add analysis context
                        results = st.session_state.analysis_results
                        if "data" in results and "analysis_results" in results[
                            "data"]:
                            context += f"\n\nAnalysis Results:\n{results['data']['analysis_results'][:1000]}..."

                        # Add visualization context
                        if "data" in results and "plots" in results["data"]:
                            context += f"\n\nI've created {len(results['data']['plots'])} visualizations for your data."

                    # Generate response
                    from langchain_openai import ChatOpenAI
                    from langchain_core.messages import AIMessage

                    llm = ChatOpenAI(model="gpt-4-turbo-preview",
                                     temperature=0.3)
                    response = llm.invoke(f"""You are an assistant for a Multi-Agent Data Analysis System.

                    Context about the analysis:
                    {context}

                    User question: {prompt}

                    Provide a helpful and informative response about the data analysis.""")

                    # Add AI response to chat history
                    ai_message = AIMessage(content=response.content)
                    st.session_state.chat_history.append(ai_message)

                    # Display AI response
                    with st.chat_message("assistant"):
                        st.markdown(response.content)

                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")

    with tab3:
        st.header("Debug Information")

        # Show memory contents
        with st.expander("Memory Contents", expanded=False):
            st.json(st.session_state.agent_system.memory)

        # Show agent thought processes
        if "analysis_results" in st.session_state and st.session_state.analysis_results:
            results = st.session_state.analysis_results

            if "thought_process" in results:
                st.subheader("Agent Thought Processes")

                for agent, thought in results["thought_process"].items():
                    with st.expander(f"{agent.capitalize()} Thought Process",
                                     expanded=False):
                        st.markdown(thought)

        # Show system logs
        if os.path.exists("debug.log"):
            with st.expander("System Logs", expanded=False):
                with open("debug.log", "r") as f:
                    st.code(f.read())


if __name__ == "__main__":
    main()