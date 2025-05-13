# Multi-Agent Data Analysis System

This project implements a multi-agent system using LangGraph for data analysis. The system consists of three specialized agents:

1. Data Wrangler Agent: Handles data loading and preparation
2. Data Analyst Agent: Performs statistical analysis and insights extraction
3. Data Visualizer Agent: Creates meaningful visualizations

Notes:
I am about to implement the next steps:
- Change and test prompts in LangSmith playground
- Change model for gpt-4.1
- Add memory
- MCP (Anthropic)
- Enhance visualisation (more charts, meaningul charts)
- Add reasoning (CoT process)

## Setup

1. Clone this repository
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the project root with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Usage

1. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```
2. Upload a CSV file through the web interface
3. Ask questions about your data in the chat interface
4. The system will process your request through the three agents and provide insights and visualizations

## Features

- File upload support for CSV data
- Interactive chat interface
- Multi-agent workflow for data processing
- Automatic data analysis and visualization
- Persistent chat history during the session

## Requirements

- Python 3.8+
- OpenAI API key
- Required Python packages (see requirements.txt)

## Project Structure

- `app.py`: Main application file containing the Streamlit interface and agent definitions
- `requirements.txt`: Project dependencies
- `.env`: Environment variables (create this file with your API key) 