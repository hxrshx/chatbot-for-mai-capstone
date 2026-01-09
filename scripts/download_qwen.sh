#!/bin/bash
"""
Download Qwen2.5-7B-Instruct model locally
Run this once to download the model to your local machine
"""

echo "============================================================"
echo "Downloading Qwen2.5-7B-Instruct Model"
echo "============================================================"
echo "This will download ~15GB of model files"
echo "Model will be saved to: ./models/qwen2.5-7b"
echo ""

# Create models directory if it doesn't exist
mkdir -p models

# Check if huggingface-cli is installed
if ! command -v huggingface-cli &> /dev/null
then
    echo "❌ huggingface-cli not found. Installing huggingface_hub..."
    pip install huggingface_hub
fi

# Download model
echo "📥 Starting download..."
huggingface-cli download Qwen/Qwen2.5-7B-Instruct \
    --local-dir ./models/qwen2.5-7b \
    --local-dir-use-symlinks False

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Model downloaded successfully!"
    echo "📁 Location: ./models/qwen2.5-7b"
    echo ""
    echo "You can now run: python scripts/rag_with_llm.py"
else
    echo ""
    echo "❌ Download failed. Please check your internet connection."
    exit 1
fi
