"""Streamlit Cloud entry point for LLM Research Dashboard.

This file is the main entry point for Streamlit Cloud deployment.
It imports and runs the dashboard from the src directory.
"""

import sys
import os
from pathlib import Path

# Add current directory to Python path
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Add src directory to Python path
src_dir = current_dir / 'src'
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Now run the main dashboard app
exec(open('src/dashboard/app.py').read())