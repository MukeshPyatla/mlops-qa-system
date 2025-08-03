#!/usr/bin/env python3
"""
Helper script to prepare the repository for Streamlit Cloud deployment.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def check_git_status():
    """Check if we're in a git repository and if there are uncommitted changes."""
    try:
        result = subprocess.run(['git', 'status', '--porcelain'], 
                              capture_output=True, text=True, check=True)
        if result.stdout.strip():
            print("⚠️  Warning: There are uncommitted changes in your repository.")
            print("   Consider committing them before deployment.")
            return False
        return True
    except subprocess.CalledProcessError:
        print("❌ Error: Not in a git repository or git not available.")
        return False
    except FileNotFoundError:
        print("❌ Error: Git not found. Please install git.")
        return False

def check_required_files():
    """Check if all required files for deployment exist."""
    required_files = [
        'streamlit_app_deploy.py',
        'requirements.txt',
        '.streamlit/config.toml',
        'README.md'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required files for deployment:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    
    print("✅ All required files are present.")
    return True

def check_requirements():
    """Check if requirements.txt is optimized for Streamlit Cloud."""
    try:
        with open('requirements.txt', 'r') as f:
            content = f.read()
        
        # Check for heavy ML dependencies that might cause issues
        heavy_deps = ['torch', 'transformers', 'sentence-transformers', 'faiss-cpu']
        found_heavy = []
        
        for dep in heavy_deps:
            if dep in content and not dep.startswith('#'):
                found_heavy.append(dep)
        
        if found_heavy:
            print("⚠️  Warning: Found heavy ML dependencies in requirements.txt:")
            for dep in found_heavy:
                print(f"   - {dep}")
            print("   These might cause deployment issues on Streamlit Cloud.")
            print("   Consider using the demo version without these dependencies.")
            return False
        
        print("✅ Requirements.txt is optimized for Streamlit Cloud.")
        return True
    except FileNotFoundError:
        print("❌ Error: requirements.txt not found.")
        return False

def create_deployment_summary():
    """Create a summary of deployment-ready files."""
    print("\n📋 Deployment Summary:")
    print("=" * 50)
    
    # Check main app file
    if os.path.exists('streamlit_app_deploy.py'):
        print("✅ Main app: streamlit_app_deploy.py")
    else:
        print("❌ Main app: streamlit_app_deploy.py (missing)")
    
    # Check requirements
    if os.path.exists('requirements.txt'):
        print("✅ Dependencies: requirements.txt")
    else:
        print("❌ Dependencies: requirements.txt (missing)")
    
    # Check config
    if os.path.exists('.streamlit/config.toml'):
        print("✅ Config: .streamlit/config.toml")
    else:
        print("❌ Config: .streamlit/config.toml (missing)")
    
    # Check README
    if os.path.exists('README.md'):
        print("✅ Documentation: README.md")
    else:
        print("❌ Documentation: README.md (missing)")

def print_deployment_instructions():
    """Print deployment instructions."""
    print("\n🚀 Deployment Instructions:")
    print("=" * 50)
    print("1. Push your code to GitHub:")
    print("   git add .")
    print("   git commit -m 'Prepare for Streamlit deployment'")
    print("   git push origin main")
    print()
    print("2. Deploy on Streamlit Cloud:")
    print("   - Go to https://share.streamlit.io")
    print("   - Sign in with GitHub")
    print("   - Click 'New app'")
    print("   - Select your repository")
    print("   - Set main file to: streamlit_app_deploy.py")
    print("   - Click 'Deploy'")
    print()
    print("3. Monitor deployment:")
    print("   - Check the deployment logs")
    print("   - Ensure all dependencies install correctly")
    print("   - Test the app functionality")

def main():
    """Main function to prepare for deployment."""
    print("🤖 Streamlit Cloud Deployment Preparation")
    print("=" * 50)
    
    # Check git status
    git_ok = check_git_status()
    
    # Check required files
    files_ok = check_required_files()
    
    # Check requirements
    req_ok = check_requirements()
    
    print("\n" + "=" * 50)
    
    if files_ok and req_ok:
        print("✅ Repository is ready for deployment!")
        create_deployment_summary()
        print_deployment_instructions()
    else:
        print("❌ Repository needs attention before deployment.")
        print("Please fix the issues above before deploying.")
        sys.exit(1)

if __name__ == "__main__":
    main() 