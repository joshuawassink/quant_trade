#!/bin/bash
# Start Jupyter Lab for interactive data exploration

echo "Starting Jupyter Lab..."
echo "Notebook directory: notebooks/"
echo ""
echo "Opening browser at: http://localhost:8888"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Activate virtual environment
source .venv/bin/activate

# Start Jupyter Lab in notebooks directory
cd notebooks
jupyter lab --no-browser
