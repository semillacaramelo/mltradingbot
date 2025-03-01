"""
Quick setup script to install test dependencies
"""
import subprocess
import sys

def install_test_requirements():
    """Install required packages for testing"""
    print("Installing test dependencies...")
    try:
        subprocess.check_call([
            sys.executable, 
            "-m", 
            "pip", 
            "install", 
            "-r", 
            "test_requirements.txt"
        ])
        print("Successfully installed test dependencies")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        return False

if __name__ == "__main__":
    success = install_test_requirements()
    if not success:
        sys.exit(1)
