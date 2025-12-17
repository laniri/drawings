#!/bin/bash
# Complete database and file system reset script

echo "🧹 Resetting Children's Drawing Anomaly Detection System..."

# Stop any running servers (optional - you can do this manually)
echo "📋 Please stop the server manually (Ctrl+C) before continuing"
read -p "Press Enter when server is stopped..."

# Remove database
echo "🗄️ Removing database..."
if [ -f "drawings.db" ]; then
    rm drawings.db
    echo "   ✓ Database file removed"
else
    echo "   ℹ️ No database file found"
fi

# Clean uploads directory
echo "📁 Cleaning uploads directory..."
if [ -d "uploads" ]; then
    rm -rf uploads/*
    echo "   ✓ Uploads directory cleaned"
else
    echo "   ℹ️ No uploads directory found"
fi

# Clean static files
echo "🎯 Cleaning generated static files..."
if [ -d "static" ]; then
    rm -rf static/models/*
    rm -rf static/saliency_maps/*
    rm -rf static/embeddings/*
    echo "   ✓ Static files cleaned"
else
    echo "   ℹ️ No static directory found"
fi

# Clean sample drawings (optional)
echo "🎨 Cleaning sample drawings..."
if [ -d "sample_drawings" ]; then
    rm -rf sample_drawings/*
    echo "   ✓ Sample drawings cleaned"
else
    echo "   ℹ️ No sample drawings directory found"
fi

# Clean logs (optional)
echo "📝 Cleaning log files..."
if [ -f "backend.log" ]; then
    rm backend.log
    echo "   ✓ Backend log removed"
fi
if [ -f "app_errors.log" ]; then
    rm app_errors.log
    echo "   ✓ Error log removed"
fi

# Recreate necessary directories
echo "📂 Recreating directories..."
mkdir -p uploads
mkdir -p static/models
mkdir -p static/saliency_maps
mkdir -p static/embeddings
mkdir -p sample_drawings
echo "   ✓ Directories recreated"

echo ""
echo "✅ Reset complete! You can now:"
echo "   1. Start the server: source venv/bin/activate && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"
echo "   2. Generate sample data: python create_sample_drawings.py"
echo "   3. Upload sample data: python upload_sample_drawings.py"
echo "   4. Train models: python train_models.py"