#!/bin/bash

echo "🚀 Starting CausalLLM Pro - AI-Powered Causal Intelligence"
echo "======================================================"
echo ""
echo "📍 Starting Streamlit application..."
echo "🌐 The app will open in your default browser"
echo "🛑 Press Ctrl+C to stop the server"
echo ""

cd streamlit_app
streamlit run main.py --server.port=8501 --server.headless=false