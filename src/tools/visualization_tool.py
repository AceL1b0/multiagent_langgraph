import os
import sys
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import logging
import uuid
from io import StringIO
from typing import Tuple, Dict, Any, List

logger = logging.getLogger(__name__)

# Create directories if they don't exist
if not os.path.exists("plots"):
    os.makedirs("plots", exist_ok=True)


def execute_visualization_code(df: pd.DataFrame, code: str) -> Tuple[
    str, List[str]]:
    """
    Execute visualization code and save the resulting plots

    Args:
        df: The dataframe to visualize
        code: Python code to generate visualizations

    Returns:
        Tuple containing output text and list of saved file paths
    """
    # Generate a unique ID for this execution
    exec_id = str(uuid.uuid4())[:8]

    # Make sure the code creates plotly figures
    modified_code = code.strip()

    # Add import statements if they're not present
    imports = """
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
"""
    if not any(imp in modified_code for imp in
               ["import plotly", "import px", "import go"]):
        modified_code = imports + "\n" + modified_code

    # Make sure figures are collected
    if "plotly_figures" not in modified_code:
        modified_code = "plotly_figures = []\n" + modified_code

    # Ensure figures are saved or added to collection
    if "fig =" in modified_code and "plotly_figures.append(fig)" not in modified_code:
        modified_code = modified_code.replace("fig.show()",
                                              "plotly_figures.append(fig)")
        # If there's no fig.show(), add append at the end of blocks with fig =
        lines = modified_code.split("\n")
        for i, line in enumerate(lines):
            if "fig =" in line and i < len(lines) - 1:
                # Find the end of this code block
                indent = len(line) - len(line.lstrip())
                j = i + 1
                while j < len(lines) and (
                        not lines[j].strip() or len(lines[j]) - len(
                        lines[j].lstrip()) > indent):
                    j += 1
                # Insert the append line before the next code block
                lines.insert(j, " " * indent + "plotly_figures.append(fig)")
        modified_code = "\n".join(lines)

    # Capture stdout
    old_stdout = sys.stdout
    sys.stdout = StringIO()

    saved_files = []

    try:
        # Create environment with necessary variables
        exec_globals = {
            'pd': pd,
            'np': np,
            'px': px,
            'go': go,
            'pio': pio,
            'df': df,
            'plotly_figures': []
        }

        # Execute the code
        exec(modified_code, exec_globals)

        # Save any created figures
        for i, fig in enumerate(exec_globals.get('plotly_figures', [])):
            # Generate a unique filename
            filename = f"plot_{exec_id}_{i + 1}.html"
            filepath = os.path.join("plots", filename)

            # Save the figure
            fig.write_html(filepath)
            logger.info(f"Saved figure to {filepath}")
            saved_files.append(filename)

        # Get output
        output = sys.stdout.getvalue()

        # Restore stdout
        sys.stdout = old_stdout

        return output, saved_files

    except Exception as e:
        # Restore stdout
        sys.stdout = old_stdout

        error_msg = f"Error executing visualization code: {str(e)}"
        logger.error(error_msg)
        return error_msg, []