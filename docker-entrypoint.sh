#!/bin/bash
set -e

# Start FastAPI server in the background
uvicorn resume_video_creator.api.routes:app --host 0.0.0.0 --port 8000 &

# Start Streamlit app
streamlit run streamlit_app/app.py --server.port 8501 --server.address 0.0.0.0 