#!/usr/bin/env python3
"""
Quick deployment script for Streamlit Cloud.

This script helps prepare your project for Streamlit Cloud deployment
and provides guidance for the deployment process.
"""

import os
import sys
import subprocess
from pathlib import Path

def check_requirements():
    """Check if all required files exist."""
    required_files = [
        "streamlit_app_simple.py",
        "requirements.txt",
        "README.md"
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return False
    
    print("✅ All required files found")
    return True

def check_git_status():
    """Check git status and provide guidance."""
    try:
        result = subprocess.run(["git", "status", "--porcelain"], 
                              capture_output=True, text=True)
        
        if result.stdout.strip():
            print("⚠️  You have uncommitted changes:")
            print(result.stdout)
            print("\nPlease commit your changes before deploying:")
            print("git add .")
            print("git commit -m 'Prepare for Streamlit deployment'")
            print("git push origin main")
        else:
            print("✅ All changes are committed")
            
    except FileNotFoundError:
        print("⚠️  Git not found. Make sure you're in a git repository.")

def create_deployment_checklist():
    """Create a deployment checklist."""
    print("\n📋 Streamlit Cloud Deployment Checklist:")
    print("=" * 50)
    
    checklist = [
        "✅ All files committed to GitHub",
        "✅ streamlit_app_simple.py exists",
        "✅ requirements.txt includes streamlit",
        "✅ README.md is comprehensive",
        "✅ Repository is public (for free tier)",
        "✅ Python version is 3.9+"
    ]
    
    for item in checklist:
        print(f"  {item}")
    
    print("\n🚀 Deployment Steps:")
    print("1. Go to https://share.streamlit.io")
    print("2. Sign in with GitHub")
    print("3. Click 'New app'")
    print("4. Select your repository")
    print("5. Set main file path to: streamlit_app_simple.py")
    print("6. Click 'Deploy'")

def validate_streamlit_app():
    """Validate the Streamlit app file."""
    app_file = "streamlit_app_simple.py"
    
    if not Path(app_file).exists():
        print(f"❌ {app_file} not found")
        return False
    
    try:
        with open(app_file, 'r') as f:
            content = f.read()
        
        # Check for basic Streamlit components
        required_components = [
            "import streamlit as st",
            "st.set_page_config",
            "st.markdown",
            "st.sidebar"
        ]
        
        missing_components = []
        for component in required_components:
            if component not in content:
                missing_components.append(component)
        
        if missing_components:
            print(f"⚠️  Missing components in {app_file}: {missing_components}")
        else:
            print(f"✅ {app_file} looks good")
            
        return True
        
    except Exception as e:
        print(f"❌ Error reading {app_file}: {e}")
        return False

def check_requirements_txt():
    """Check requirements.txt for Streamlit deployment."""
    req_file = "requirements.txt"
    
    if not Path(req_file).exists():
        print(f"❌ {req_file} not found")
        return False
    
    try:
        with open(req_file, 'r') as f:
            content = f.read()
        
        required_packages = [
            "streamlit",
            "pandas",
            "numpy"
        ]
        
        missing_packages = []
        for package in required_packages:
            if package not in content:
                missing_packages.append(package)
        
        if missing_packages:
            print(f"⚠️  Missing packages in {req_file}: {missing_packages}")
        else:
            print(f"✅ {req_file} includes required packages")
            
        return True
        
    except Exception as e:
        print(f"❌ Error reading {req_file}: {e}")
        return False

def main():
    """Main deployment preparation function."""
    print("🚀 Streamlit Cloud Deployment Preparation")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Please fix the missing files before deploying")
        return
    
    # Validate Streamlit app
    validate_streamlit_app()
    
    # Check requirements.txt
    check_requirements_txt()
    
    # Check git status
    check_git_status()
    
    # Create deployment checklist
    create_deployment_checklist()
    
    print("\n🎯 Ready for deployment!")
    print("\n💡 Tips:")
    print("- Use streamlit_app_simple.py for demo deployment")
    print("- The app includes mock data to showcase MLOps concepts")
    print("- All MLOps features are demonstrated with realistic data")
    print("- Perfect for showcasing to recruiters and in interviews")

if __name__ == "__main__":
    main() 