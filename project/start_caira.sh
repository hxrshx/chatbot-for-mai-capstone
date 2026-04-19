#!/bin/bash
# Production startup script for CAIRA

echo "🏛️  Starting CAIRA - CAIRO AI Research Assistant"
echo "=================================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run setup first."
    echo "Run: python -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check if models exist
if [ ! -d "models/qwen2.5-1.5b" ]; then
    echo "📥 Downloading Qwen model (this may take a few minutes)..."
    bash scripts/download_qwen.sh
fi

# Check if embeddings exist
if [ ! -f "preprocessed/embeddings/embeddings.index" ]; then
    echo "❌ Embeddings not found. Please run preprocessing first:"
    echo "python scripts/pre_processing.py"
    echo "python scripts/generate_embeddings.py --input preprocessed/rag_chunks.jsonl --output preprocessed/embeddings"
    exit 1
fi

# Start backend in background
echo "🚀 Starting backend on port 8000..."
nohup python backend/main.py > backend.log 2>&1 &
BACKEND_PID=$!

# Wait for backend to start
sleep 10

# Check if backend is running
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ Backend ready"
else
    echo "❌ Backend failed to start. Check backend.log"
    exit 1
fi

# Start frontend
echo "🎨 Starting frontend on port 7860..."
echo "📱 Access CAIRA at: http://localhost:7860"
echo "🛑 Press Ctrl+C to stop"
python frontend/app.py

# Cleanup on exit
echo "🛑 Stopping backend..."
kill $BACKEND_PID 2>/dev/null