#!/bin/bash

# Script to run offline training with proper virtual environment

echo "=== Offline Training Setup ==="

# Check if we're in the right directory
if [ ! -f "train_models_offline.py" ]; then
    echo "❌ Error: train_models_offline.py not found"
    echo "   Make sure you're in the project directory: /Users/itay/Desktop/drawings"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Error: Virtual environment 'venv' not found"
    echo "   Please create it first: python -m venv venv"
    exit 1
fi

echo "✓ Found project files"
echo "✓ Found virtual environment"

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Verify Python path
echo "✓ Using Python: $(which python)"

# Check if required packages are installed
echo "🔄 Checking dependencies..."
python -c "import torch, sqlalchemy, numpy; print('✓ Core dependencies available')" || {
    echo "❌ Missing dependencies. Installing..."
    pip install -r requirements.txt
}

echo "🚀 Starting offline training..."
echo ""

# Run the offline training
python train_models_offline.py "$@"

echo ""
echo "=== Training Complete ==="