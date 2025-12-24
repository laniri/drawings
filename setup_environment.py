#!/usr/bin/env python3
"""
Setup script for Children's Drawing Anomaly Detection System
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return result
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return None


def main():
    """Main setup function."""
    print("🚀 Setting up Children's Drawing Anomaly Detection System")
    print("=" * 60)
    
    # Check Python version
    if sys.version_info < (3, 11):
        print("❌ Python 3.11 or higher is required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Create virtual environment
    venv_path = Path("venv")
    if not venv_path.exists():
        if not run_command("python -m venv venv", "Creating virtual environment"):
            sys.exit(1)
    else:
        print("✅ Virtual environment already exists")
    
    # Determine activation script path
    if os.name == 'nt':  # Windows
        activate_script = "venv\\Scripts\\activate"
        pip_path = "venv\\Scripts\\pip"
    else:  # macOS/Linux
        activate_script = "venv/bin/activate"
        pip_path = "venv/bin/pip"
    
    # Install dependencies
    if not run_command(f"{pip_path} install --upgrade pip", "Upgrading pip"):
        sys.exit(1)
    
    if not run_command(f"{pip_path} install -r requirements-dev.txt", "Installing Python dependencies"):
        sys.exit(1)
    
    # Create .env file if it doesn't exist
    env_file = Path(".env")
    if not env_file.exists():
        if Path(".env.example").exists():
            run_command("cp .env.example .env", "Creating .env file")
        else:
            print("⚠️  .env.example not found, skipping .env creation")
    else:
        print("✅ .env file already exists")
    
    # Create necessary directories
    for directory in ["uploads", "static", "alembic/versions"]:
        Path(directory).mkdir(parents=True, exist_ok=True)
    print("✅ Created necessary directories")
    
    # Install pre-commit hooks (optional)
    if run_command(f"{pip_path} install pre-commit", "Installing pre-commit"):
        run_command("pre-commit install", "Setting up pre-commit hooks")
    
    print("\n" + "=" * 60)
    print("🎉 Setup completed successfully!")
    print("\nNext steps:")
    print(f"1. Activate the virtual environment:")
    if os.name == 'nt':
        print("   venv\\Scripts\\activate")
    else:
        print("   source venv/bin/activate")
    print("2. Start the backend server:")
    print("   uvicorn app.main:app --reload")
    print("3. In another terminal, set up the frontend:")
    print("   cd frontend && npm install && npm run dev")
    print("\nAPI Documentation will be available at: http://localhost:8000/docs")
    print("Frontend will be available at: http://localhost:3000")


if __name__ == "__main__":
    main()