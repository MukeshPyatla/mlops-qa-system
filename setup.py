#!/usr/bin/env python3
"""
Setup script for the LLM-Powered Multi-Source Q&A System.

This script helps users set up the system with proper configuration
and initial data collection.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import json

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def create_directories():
    """Create necessary directories."""
    directories = [
        "data/raw",
        "data/processed", 
        "models",
        "logs",
        "configs",
        "tests",
        "scripts"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {directory}")

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 9):
        print("❌ Python 3.9 or higher is required")
        sys.exit(1)
    print(f"✅ Python version {sys.version_info.major}.{sys.version_info.minor} is compatible")

def install_dependencies():
    """Install Python dependencies."""
    if not run_command("pip install -r requirements.txt", "Installing Python dependencies"):
        print("❌ Failed to install dependencies")
        sys.exit(1)

def setup_environment():
    """Set up environment variables."""
    env_file = Path(".env")
    if not env_file.exists():
        print("📝 Creating .env file from template...")
        shutil.copy("env.example", ".env")
        print("✅ Environment file created. Please edit .env with your configuration.")

def validate_configuration():
    """Validate the configuration files."""
    config_files = ["configs/data_sources.yaml", "configs/models.yaml"]
    
    for config_file in config_files:
        if not Path(config_file).exists():
            print(f"❌ Configuration file missing: {config_file}")
            return False
    
    print("✅ Configuration files validated")
    return True

def run_tests():
    """Run the test suite."""
    if not run_command("python -m pytest tests/ -v", "Running tests"):
        print("⚠️  Some tests failed, but continuing with setup")

def collect_initial_data():
    """Collect initial data from sources."""
    print("🔄 Collecting initial data from sources...")
    if run_command("python scripts/collect_data.py --all-sources", "Collecting data"):
        print("✅ Initial data collection completed")
    else:
        print("⚠️  Initial data collection failed, but system can still be used")

def build_index():
    """Build the initial vector index."""
    print("🔄 Building initial vector index...")
    if run_command("python scripts/rebuild_index.py", "Building index"):
        print("✅ Initial index built successfully")
    else:
        print("⚠️  Index building failed, but system can still be used")

def print_next_steps():
    """Print next steps for the user."""
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Edit .env file with your configuration")
    print("2. Start the API server: python src/api/main.py")
    print("3. Or use Docker: docker-compose up -d")
    print("4. Access the API at: http://localhost:8000")
    print("5. View documentation at: http://localhost:8000/docs")
    print("\n📚 For more information, see the README.md file")

def main():
    """Main setup function."""
    print("🚀 Setting up LLM-Powered Multi-Source Q&A System")
    print("=" * 60)
    
    # Check Python version
    check_python_version()
    
    # Create directories
    create_directories()
    
    # Install dependencies
    install_dependencies()
    
    # Setup environment
    setup_environment()
    
    # Validate configuration
    if not validate_configuration():
        print("❌ Configuration validation failed")
        sys.exit(1)
    
    # Run tests
    run_tests()
    
    # Collect initial data (optional)
    response = input("\n🤔 Would you like to collect initial data now? (y/n): ")
    if response.lower() in ['y', 'yes']:
        collect_initial_data()
        build_index()
    
    # Print next steps
    print_next_steps()

if __name__ == "__main__":
    main() 