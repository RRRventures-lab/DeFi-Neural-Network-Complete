#!/bin/bash

# DeFi Neural Network Trading Dashboard Backend Startup Script

set -e

echo "================================"
echo "DeFi Neural Network Backend"
echo "Trading Dashboard API Server"
echo "================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed"
    exit 1
fi

echo "✅ Python found: $(python3 --version)"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo "✅ Virtual environment activated"

# Install/upgrade dependencies
echo ""
echo "Installing dependencies..."
pip install --upgrade -q pip
pip install --break-system-packages -q -r requirements.txt 2>/dev/null || pip install -q -r requirements.txt
echo "✅ Dependencies installed"

# Create .env if it doesn't exist
if [ ! -f ".env" ]; then
    echo ""
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "✅ .env file created"
    echo "   Note: Update .env with your configuration"
fi

# Start the server
echo ""
echo "================================"
echo "Starting Trading Dashboard API"
echo "================================"
echo ""
echo "🚀 Server running at:"
echo "   http://localhost:8000"
echo ""
echo "📚 API Documentation:"
echo "   Swagger UI: http://localhost:8000/docs"
echo "   ReDoc: http://localhost:8000/redoc"
echo ""
echo "💬 WebSocket:"
echo "   ws://localhost:8000/ws/updates"
echo ""
echo "🛑 To stop: Press Ctrl+C"
echo ""
echo "================================"
echo ""

# Start uvicorn server with reload enabled for development
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
