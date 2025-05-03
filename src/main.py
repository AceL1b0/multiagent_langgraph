import streamlit as st
import os
from dotenv import load_dotenv
from src.workflow.main_workflow import create_workflow
from src.config.langsmith_config import get_langsmith_env

# Load environment variables
load_dotenv()

# Set LangSmith environment variables
for key, value in get_langsmith_env().items():
    os.environ[key] = value

def main():
    st.title("Multi-Agent Data Analysis System")
    
    # File upload
    uploaded_file = st.file_uploader("Upload your data file", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        # Initialize workflow
        workflow = create_workflow()
        
        # Process the file
        if st.button("Process Data"):
            with st.spinner("Processing data..."):
                try:
                    # Run the workflow
                    result = workflow.invoke({
                        "messages": [],
                        "data": {"file": uploaded_file},
                        "current_agent": "data_wrangling"
                    })
                    
                    # Display results
                    st.success("Processing complete!")
                    
                    # Display messages
                    for message in result["messages"]:
                        if isinstance(message, AIMessage):
                            st.write(message.content)
                    
                    # Display plots if any were generated
                    if "plots" in result["data"]:
                        for plot_path in result["data"]["plots"]:
                            st.image(plot_path)
                            
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 