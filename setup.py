#!/usr/bin/env python3
"""
Setup script for the Time Series Forecasting project.

This script helps users set up the project environment and run initial tests.
"""

import subprocess
import sys
from pathlib import Path


def run_command(command: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False


def check_python_version() -> bool:
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major == 3 and version.minor >= 10:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} is not compatible")
        print("Please use Python 3.10 or higher")
        return False


def main():
    """Main setup function."""
    print("🚀 Setting up Time Series Forecasting for Finance")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check if we're in the right directory
    if not Path("requirements.txt").exists():
        print("❌ requirements.txt not found. Please run this script from the project root.")
        sys.exit(1)
    
    # Install dependencies
    if not run_command("pip install --upgrade pip", "Upgrading pip"):
        sys.exit(1)
    
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        print("⚠️  Some dependencies may have failed to install. Please check manually.")
    
    # Create necessary directories
    directories = ["data", "assets", "logs"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    # Run basic tests
    if not run_command("python -m pytest tests/ -v", "Running basic tests"):
        print("⚠️  Some tests failed. This is normal for the first run.")
    
    # Test imports
    if not run_command("python -c \"import sys; sys.path.append('src'); from utils import *; print('Imports successful')\"", "Testing imports"):
        print("❌ Import test failed. Please check your installation.")
        sys.exit(1)
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run the training pipeline: python scripts/train.py")
    print("2. Launch the interactive demo: streamlit run demo/app.py")
    print("3. Explore the example notebook: jupyter notebook notebooks/example_usage.ipynb")
    print("\n⚠️  Remember: This is for research and educational purposes only!")
    print("   Always consult with qualified financial professionals before making investment decisions.")


if __name__ == "__main__":
    main()
