"""
NexusRAG - Document Retrieval & Question-Answering System
Backward-compatible launcher: seamlessly executes app.py when started via:
    streamlit run prj.py
"""

import os
import runpy

if __name__ == "__main__" or __name__ == "builtins":
    app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")
    runpy.run_path(app_path, run_name="__main__")
else:
    # If imported as a module
    from app import *