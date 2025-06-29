#!/usr/bin/env python3
"""
Entry point for Hugging Face Spaces deployment
"""

import os
import sys

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and run the main app
from app import demo

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860) 