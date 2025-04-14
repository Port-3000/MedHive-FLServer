# app.py - Main entry point for the FastAPI application
import os
import sys

# Add the current directory to the Python path to allow imports from fl_server
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the FastAPI app from fl_server.main
from fl_server.main import app

# This allows running with: uvicorn app:app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8082, reload=True)