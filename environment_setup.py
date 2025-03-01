"""
Environment setup script for the Deriv ML Trading Bot

This script performs checks and sets up necessary environment components
"""
import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path

def print_header(text):
    """Print a formatted header"""
    print("\n" + "=" * 60)
    print(f" {text} ".center(60, "="))
    print("=" * 60 + "\n")

def print_status(message, success=True):
    """Print a status message with colored output"""
    if success:
        prefix = "[✓]"
    else:
        prefix = "[✗]"
    print(f"{prefix} {message}")

def check_python_version():
    """Verify Python version meets requirements"""
    print_header("Python Version Check")

    major, minor, _ = platform.python_version_tuple()
    print(f"Detected Python version: {platform.python_version()}")

    if int(major) < 3 or (int(major) == 3 and int(minor) < 10):
        print_status("Python 3.10 or higher is recommended for optimal compatibility", False)
        return False
    else:
        print_status("Python version is compatible")
        return True

def check_create_directories():
    """Check and create necessary directories"""
    print_header("Directory Structure Check")

    directories = [
        "models",
        "models/archive",
        "logs",
        "data",
        "data/historical"
    ]

    for directory in directories:
        path = Path(directory)
        if not path.exists():
            try:
                path.mkdir(parents=True, exist_ok=True)
                print_status(f"Created directory: {directory}")
            except Exception as e:
                print_status(f"Failed to create directory {directory}: {str(e)}", False)
                return False
        else:
            print_status(f"Directory exists: {directory}")

    return True

def check_dependencies():
    """Check Python dependencies"""
    print_header("Dependency Check")

    required_packages = [
        "python-deriv-api",
        "python-dotenv",
        "pandas",
        "numpy",
        "tensorflow",
        "websockets",
        "matplotlib",
        "scikit-learn"
    ]

    missing_packages = []

    try:
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
                print_status(f"Package found: {package}")
            except ImportError:
                print_status(f"Package missing: {package}", False)
                missing_packages.append(package)
    except Exception as e:
        print_status(f"Error checking dependencies: {str(e)}", False)
        return False, missing_packages

    return len(missing_packages) == 0, missing_packages

def install_missing_packages(packages):
    """Install missing Python packages"""
    print_header("Installing Missing Packages")

    if not packages:
        print("No packages need to be installed.")
        return True

    try:
        print(f"Installing packages: {', '.join(packages)}")
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + packages)
        print_status("All packages installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print_status(f"Failed to install packages: {str(e)}", False)
        return False
    except Exception as e:
        print_status(f"Error during package installation: {str(e)}", False)
        return False

def check_env_file():
    """Check for .env file"""
    print_header("Environment Configuration Check")

    env_example_path = Path(".env.example")
    env_path = Path(".env")

    if env_path.exists():
        print_status(".env file exists")
        return True
    else:
        print_status(".env file does not exist", False)

        if env_example_path.exists():
            try:
                shutil.copy(env_example_path, env_path)
                print_status("Created .env file from .env.example")
                print("⚠️  Please edit the .env file with your actual API keys and configuration ⚠️")
                return True
            except Exception as e:
                print_status(f"Failed to create .env file: {str(e)}", False)
                return False
        else:
            print_status(".env.example file not found", False)
            return False

def main():
    """Main setup function"""
    print_header("Deriv ML Trading Bot - Environment Setup")

    print("Performing environment checks and setup...")

    # Perform checks
    python_ok = check_python_version()
    dirs_ok = check_create_directories()
    deps_ok, missing_packages = check_dependencies()

    # Install missing packages if needed
    if not deps_ok:
        if input("Do you want to install missing packages? (y/n): ").lower() == 'y':
            deps_ok = install_missing_packages(missing_packages)

    # Check environment file
    env_ok = check_env_file()

    # Summary
    print_header("Setup Summary")
    print_status("Python version", python_ok)
    print_status("Directory structure", dirs_ok)
    print_status("Dependencies", deps_ok)
    print_status("Environment configuration", env_ok)

    if python_ok and dirs_ok and deps_ok and env_ok:
        print("\n✅ Setup completed successfully! The bot is ready to run.")
        return True
    else:
        print("\n⚠️  Setup completed with warnings. Please address the issues above before running the bot.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)