#!/bin/bash

# Setup script for Children's Drawing Anomaly Detection System

set -e  # Exit on any error

echo "🚀 Setting up Children's Drawing Anomaly Detection System"
echo "============================================================"

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
required_version="3.11"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python 3.11 or higher is required. Found: $python_version"
    exit 1
fi

echo "✅ Python $python_version detected"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "🔄 Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment and install dependencies
echo "🔄 Installing Python dependencies..."
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements-dev.txt
echo "✅ Python dependencies installed"

# Create .env file
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "✅ Created .env file from .env.example"
    else
        echo "⚠️  .env.example not found, skipping .env creation"
    fi
else
    echo "✅ .env file already exists"
fi

# Create necessary directories
mkdir -p uploads static alembic/versions
echo "✅ Created necessary directories"

# Install pre-commit hooks (optional)
if command -v pre-commit &> /dev/null; then
    pre-commit install
    echo "✅ Pre-commit hooks installed"
fi

echo ""
echo "============================================================"
echo "🎉 Backend setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo "2. Start the backend server:"
echo "   uvicorn app.main:app --reload"
echo "3. In another terminal, set up the frontend:"
echo "   cd frontend && npm install && npm run dev"
echo ""
echo "API Documentation will be available at: http://localhost:8000/docs"
echo "Frontend will be available at: http://localhost:3000"