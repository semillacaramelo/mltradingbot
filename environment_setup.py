"""
Environment setup script for Deriv ML Trading Bot
Helps configure the trading environment for local development and execution
"""
import os
import sys
import argparse
import shutil
import platform
import subprocess
from pathlib import Path
from typing import Dict, List
import webbrowser
import time

MIN_PYTHON_VERSION = (3, 11)  # Minimum required Python version
MAX_PYTHON_VERSION = (3, 12)  # Maximum tested Python version

# Required packages with version constraints
REQUIRED_PACKAGES = {
    # Base scientific packages (numpy handled separately)
    "pandas": "==2.0.3",
    "matplotlib": ">=3.10.0",
    "scikit-learn": ">=1.6.1",
    
    # Deep learning
    "tensorflow": ">=2.14.0",
    
    # API and networking
    "python-deriv-api": ">=0.1.6",
    "websockets": ">=10.3",
    "python-dotenv": ">=1.0.1",
}

def ensure_self_dependencies():
    """Ensure the script's own dependencies are installed"""
    core_packages = {
        "setuptools": ">=41.0.0",
        "wheel": ">=0.33.0",
        "pip": ">=19.0.0",
        "python-dotenv": ">=1.0.1"
    }
    
    print("\nChecking core dependencies...")
    for package, version in core_packages.items():
        try:
            print(f"Checking {package}...")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "--upgrade", f"{package}{version}"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE
            )
        except subprocess.CalledProcessError as e:
            print(f"Warning: Could not upgrade {package}: {e}")
            print(f"Attempting simple installation of {package}...")
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", package],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE
                )
            except subprocess.CalledProcessError as e2:
                print(f"Error: Failed to install {package}: {e2}")
                return False
    return True

def get_installed_packages():
    """Get list of installed packages using pip"""
    try:
        import json
        result = subprocess.run(
            [sys.executable, "-m", "pip", "list", "--format=json"],
            capture_output=True,
            text=True
        )
        packages = json.loads(result.stdout)
        return {pkg['name'].lower(): pkg['version'] for pkg in packages}
    except Exception as e:
        print(f"Error getting installed packages: {e}")
        return {}

def check_visual_cpp():
    """Check if Visual C++ Build Tools are installed and guide installation if needed"""
    if platform.system() != 'Windows':
        return True

    def check_vs_install():
        """Check for Visual Studio installation markers"""
        # Common paths for VS Build Tools
        possible_paths = [
            r"C:\Program Files (x86)\Microsoft Visual Studio",
            r"C:\Program Files\Microsoft Visual Studio",
            r"C:\Program Files (x86)\Windows Kits",
            r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools",
            r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
        ]
        
        # Check for cl.exe in PATH
        cl_paths = [
            path for path in os.environ.get("PATH", "").split(os.pathsep)
            if os.path.exists(os.path.join(path, "cl.exe"))
        ]
        
        if cl_paths:
            print("Found Visual C++ compiler in PATH")
            return True

        # Check for installation directories
        for path in possible_paths:
            if os.path.exists(path):
                print(f"Found Visual Studio installation at: {path}")
                return True
                
        return False

    if check_vs_install():
        return True

    print("\nMissing or incomplete Visual C++ Build Tools installation.")
    print("Opening Visual Studio Build Tools download page...")
    webbrowser.open('https://visualstudio.microsoft.com/visual-cpp-build-tools/')
    print("\nPlease:")
    print("1. Download and run the Visual Studio Build Tools installer")
    print("2. In the installer, select 'Desktop development with C++'")
    print("3. Install and restart your computer")
    print("\nNote: If you've already installed it, you may need to:")
    print("1. Run 'Visual Studio Installer'")
    print("2. Modify your installation")
    print("3. Ensure 'Desktop development with C++' is selected")
    print("4. Click 'Modify' to update the installation")
    
    if not args.no_input:
        action = input("\nChoose an action:\n1. Continue anyway\n2. Open installer\n3. Exit\nChoice (1-3): ")
        if action == "2":
            os.startfile("C:\\Program Files (x86)\\Microsoft Visual Studio\\Installer\\vs_installer.exe")
        elif action == "3":
            sys.exit(1)
        # action == "1" or invalid input will continue
    return False

def install_package_safe(package, use_binary=True):
    """Install a package with fallback options"""
    try:
        print(f"\nInstalling {package}")
        cmd = [sys.executable, "-m", "pip", "install"]
        
        if use_binary:
            if platform.system() == 'Windows':
                # Try to find a compatible wheel first
                cmd.extend(["--only-binary", ":all:", "--prefer-binary"])
            else:
                cmd.extend(["--prefer-binary"])
        
        cmd.append(package)
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            if use_binary:
                print("Binary installation failed, trying source installation...")
                return install_package_safe(package, use_binary=False)
            print(f"Error installing {package}:")
            print(result.stderr)
            return False
            
        print(f"Successfully installed {package}")
        return True
    except Exception as e:
        print(f"Error installing {package}: {e}")
        return False

def install_numpy_safe():
    """Special handling for numpy installation"""
    print("\nInstalling numpy with special handling...")
    
    # Try different numpy versions based on Python version
    if sys.version_info >= (3, 12):
        numpy_versions = [
            "numpy==1.26.3",  # Latest version compatible with Python 3.12
            "numpy==1.26.2",
            "numpy",  # Latest stable as fallback
        ]
    else:
        numpy_versions = [
            "numpy==1.24.3",  # Original specified version
            "numpy==1.24.2",
            "numpy",  # Latest stable as fallback
        ]

    for version in numpy_versions:
        try:
            print(f"\nTrying to install {version}...")
            # First try binary
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", 
                 "--only-binary", ":all:", 
                 "--prefer-binary", version],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print(f"Successfully installed {version}")
                return True
                
            # If binary fails, try source
            print("Binary installation failed, trying source installation...")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", version],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print(f"Successfully installed {version}")
                return True
                
        except Exception as e:
            print(f"Error installing {version}: {e}")
            continue
            
    print("Failed to install numpy after trying multiple versions")
    return False

def install_scientific_packages():
    """Install scientific packages in the correct order"""
    # Install numpy first with special handling
    if not install_numpy_safe():
        return False
        
    # Then install other scientific packages
    other_packages = [
        ("pandas", "==2.0.3"),
        ("matplotlib", ">=3.10.0"),
        ("scikit-learn", ">=1.6.1"),
    ]
    
    for package, version in other_packages:
        if not install_package_safe(f"{package}{version}"):
            return False
    return True

def install_dependencies(packages):
    """Install multiple packages using pip"""
    try:
        print("\nInstalling packages...")
        
        # First ensure Visual C++ is available for scientific packages
        if not check_visual_cpp():
            print("Please install Visual C++ Build Tools and try again")
            return False

        # Handle scientific packages first
        if not install_scientific_packages():
            return False
            
        # Install remaining packages
        for package in packages:
            if any(pkg in package for pkg in ["numpy", "pandas", "matplotlib", "scikit-learn"]):
                continue  # Skip already installed scientific packages
            if not install_package_safe(package, use_binary=True):
                return False
                
        return True
    except Exception as e:
        print(f"Error installing packages: {e}")
        if hasattr(e, 'stderr'):
            print(f"Error details:\n{e.stderr}")
        return False

def check_python_version():
    """Check if current Python version is compatible"""
    current = sys.version_info[:2]
    if current < MIN_PYTHON_VERSION or current > MAX_PYTHON_VERSION:
        print(f"Warning: This project requires Python {MIN_PYTHON_VERSION[0]}.{MIN_PYTHON_VERSION[1]} to {MAX_PYTHON_VERSION[0]}.{MAX_PYTHON_VERSION[1]}")
        print(f"Current Python version is {current[0]}.{current[1]}")
        if not args.no_input:
            cont = input("Continue anyway? (y/n): ")
            if cont.lower() != 'y':
                sys.exit(1)
        return False
    return True

def setup_environment():
    """Main environment setup function"""
    parse_args()
    print("\n=== Deriv ML Trading Bot - Environment Setup ===\n")
    
    # First ensure we have our core dependencies
    if not ensure_self_dependencies():
        print("\nFailed to install core dependencies. Please try:")
        print("python -m pip install --upgrade pip setuptools wheel python-dotenv")
        sys.exit(1)

    # Now we can safely import what we need
    try:
        from dotenv import load_dotenv
        import pkg_resources
    except ImportError as e:
        print(f"Error importing core dependencies even after installation: {e}")
        sys.exit(1)
    
    # Check Python version
    check_python_version()
    
    # Create virtual environment if needed
    setup_virtual_environment()
    
    # Create .env file if it doesn't exist
    env_file = Path(".env")
    env_example = Path(".env.example")

    if not env_file.exists():
        if (env_example.exists()):
            shutil.copy(env_example, env_file)
            print("Created .env file from .env.example")
        else:
            create_default_env_file()
            print("Created default .env file")

    try:
        # Load environment variables from .env file
        load_dotenv(env_file)
    except Exception as e:
        print(f"Warning: Could not load .env file: {e}")

    # Collect environment configuration
    collect_environment_variables()

    # Create necessary directories
    create_directories()

    # Check Python dependencies
    check_dependencies()

    # Create VS Code configuration if requested
    if args.vscode:
        create_vscode_config()

    print("\n=== Environment setup completed ===")
    print_usage_instructions()

def parse_args():
    global args
    parser = argparse.ArgumentParser(description="Setup environment for Deriv ML Trading Bot")
    parser.add_argument('--vscode', action='store_true', help='Create VS Code configuration')
    parser.add_argument('--force', action='store_true', help='Override existing files')
    parser.add_argument('--no-input', action='store_true', help='Don\'t prompt for user input, use defaults')
    args = parser.parse_args()
    return args

def create_default_env_file():
    """Create a default .env file with required variables"""
    default_env = """# Deriv ML Trading Bot Environment Configuration
# Updated by environment_setup.py

# Trading Environment (demo/real)
DERIV_BOT_ENV=demo

# Deriv API Tokens (get from https://app.deriv.com/account/api-token)
# You need tokens with "Read", "Trade", "Payments" and "Admin" permissions
DERIV_API_TOKEN_DEMO=YOUR_DEMO_TOKEN_HERE
DERIV_API_TOKEN_REAL=YOUR_REAL_TOKEN_HERE

# IMPORTANT: Set to "yes" to enable real trading mode (safety measure)
DERIV_REAL_MODE_CONFIRMED=no

# Application ID
APP_ID=1089

# Training Parameters
SEQUENCE_LENGTH=30
TRAINING_EPOCHS=50
MODEL_SAVE_PATH=models
"""

    with open(".env", "w") as f:
        f.write(default_env)

def collect_environment_variables():
    """Collect and save environment variables from user input"""
    if args.no_input:
        print("Skipping user input due to --no-input flag")
        return

    print("\nConfiguring environment variables:")
    print("(Press Enter to keep current value)\n")

    # Helper function to update env vars
    def update_env_var(var_name, prompt, current_value=None, is_secret=False):
        if current_value is None:
            current_value = os.getenv(var_name, "")

        if is_secret and current_value:
            display_value = "*" * 8 + current_value[-4:] if len(current_value) > 4 else "*" * len(current_value)
            new_value = input(f"{prompt} [{display_value}]: ")
        else:
            new_value = input(f"{prompt} [{current_value}]: ")

        if new_value:
            return new_value
        return current_value

    # Environment selection
    env = update_env_var("DERIV_BOT_ENV", "Trading environment (demo/real)")
    if env.lower() not in ["demo", "real"]:
        print("Invalid environment. Using 'demo' as default.")
        env = "demo"

    # API tokens
    demo_token = update_env_var("DERIV_API_TOKEN_DEMO", 
                               "Demo API token (from app.deriv.com/account/api-token)", 
                               is_secret=True)
    real_token = update_env_var("DERIV_API_TOKEN_REAL", 
                               "Real account API token (from app.deriv.com/account/api-token)",
                               is_secret=True)

    # Safety confirmation for real mode
    real_confirmed = "no"
    if env.lower() == "real":
        real_confirmed = update_env_var("DERIV_REAL_MODE_CONFIRMED", 
                                        "Confirm real trading mode (yes/no)", 
                                        current_value="no")
        if real_confirmed.lower() != "yes":
            print("Real mode requires explicit confirmation. Setting DERIV_REAL_MODE_CONFIRMED=no")
            real_confirmed = "no"

    # App ID
    app_id = update_env_var("APP_ID", "App ID", "1089")

    # Training parameters
    sequence_length = update_env_var("SEQUENCE_LENGTH", "Sequence length for training", "30")
    training_epochs = update_env_var("TRAINING_EPOCHS", "Training epochs", "50")
    model_path = update_env_var("MODEL_SAVE_PATH", "Model save path", "models")

    # Update .env file
    update_env_file({
        "DERIV_BOT_ENV": env,
        "DERIV_API_TOKEN_DEMO": demo_token,
        "DERIV_API_TOKEN_REAL": real_token,
        "DERIV_REAL_MODE_CONFIRMED": real_confirmed,
        "APP_ID": app_id,
        "SEQUENCE_LENGTH": sequence_length,
        "TRAINING_EPOCHS": training_epochs,
        "MODEL_SAVE_PATH": model_path
    })

    print("\nEnvironment variables updated successfully!")

def update_env_file(new_vars):
    """Update .env file with new variables"""
    # Read current content
    if os.path.exists(".env"):
        with open(".env", "r") as f:
            content = f.read()
    else:
        content = ""

    # Parse current variables
    lines = content.split("\n")
    env_vars = {}

    for line in lines:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            env_vars[key.strip()] = value.strip()

    # Update with new values
    env_vars.update(new_vars)

    # Write back to file
    new_content = []

    # Add header
    new_content.append("# Deriv ML Trading Bot Environment Configuration")
    new_content.append("# Updated by environment_setup.py on " + get_timestamp())
    new_content.append("")

    # Trading environment
    new_content.append("# Trading Environment (demo/real)")
    new_content.append(f"DERIV_BOT_ENV={env_vars['DERIV_BOT_ENV']}")
    new_content.append("")

    # API tokens
    new_content.append("# Deriv API Tokens (get from https://app.deriv.com/account/api-token)")
    new_content.append("# You need tokens with \"Read\", \"Trade\", \"Payments\" and \"Admin\" permissions")
    new_content.append(f"DERIV_API_TOKEN_DEMO={env_vars['DERIV_API_TOKEN_DEMO']}")
    new_content.append(f"DERIV_API_TOKEN_REAL={env_vars['DERIV_API_TOKEN_REAL']}")
    new_content.append("")

    # Safety confirmation
    new_content.append("# IMPORTANT: Set to \"yes\" to enable real trading mode (safety measure)")
    new_content.append(f"DERIV_REAL_MODE_CONFIRMED={env_vars['DERIV_REAL_MODE_CONFIRMED']}")
    new_content.append("")

    # Application ID
    new_content.append("# Application ID")
    new_content.append(f"APP_ID={env_vars['APP_ID']}")
    new_content.append("")

    # Training parameters
    new_content.append("# Training Parameters")
    new_content.append(f"SEQUENCE_LENGTH={env_vars['SEQUENCE_LENGTH']}")
    new_content.append(f"TRAINING_EPOCHS={env_vars['TRAINING_EPOCHS']}")
    new_content.append(f"MODEL_SAVE_PATH={env_vars['MODEL_SAVE_PATH']}")

    # Write to file
    with open(".env", "w") as f:
        f.write("\n".join(new_content))

def get_timestamp():
    """Get current timestamp in string format"""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def create_directories():
    """Create necessary directories for the project"""
    directories = ["models", "model_archive", "logs", "data"]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Verified directory: {directory}")

def check_dependencies():
    """Check and report on required Python dependencies"""
    ensure_self_dependencies()
    
    try:
        # Get installed packages
        installed = get_installed_packages()
        
        missing_packages = []
        outdated_packages = []
        
        for package, version_constraint in REQUIRED_PACKAGES.items():
            pkg_name = package.replace("-", "_").lower()
            if pkg_name not in installed:
                missing_packages.append(f"{package}{version_constraint}")
                continue
                
            # Simple version check (only for exact versions)
            if "==" in version_constraint:
                required_version = version_constraint.replace("==", "").strip()
                if installed[pkg_name] != required_version:
                    outdated_packages.append(f"{package}{version_constraint}")

        if missing_packages or outdated_packages:
            print("\nPackage issues found:")
            if missing_packages:
                print("\nMissing packages:")
                for pkg in missing_packages:
                    print(f"- {pkg}")
            
            if outdated_packages:
                print("\nOutdated packages:")
                for pkg in outdated_packages:
                    print(f"- {pkg}")

            print("\nAttempting automatic installation...")
            all_packages = missing_packages + outdated_packages
            
            # Install one by one in the defined order
            for package in all_packages:
                success = install_dependencies([package])
                if not success:
                    print(f"\nFailed to install {package}")
                    print("You may need to install the following packages manually:")
                    for remaining in all_packages:
                        print(f"pip install {remaining}")
                    break
        else:
            print("\nAll required Python dependencies are installed!")
            
    except Exception as e:
        print(f"\nError checking dependencies: {str(e)}")
        print("Please install the required packages manually:")
        for package, version in REQUIRED_PACKAGES.items():
            print(f"pip install {package}{version}")

def setup_virtual_environment():
    """Create and activate virtual environment if needed"""
    venv_path = Path('tradingenv')
    if not venv_path.exists():
        print("Creating virtual environment...")
        try:
            subprocess.run([sys.executable, '-m', 'venv', str(venv_path)], check=True)
            print("Virtual environment created successfully")
        except subprocess.CalledProcessError as e:
            print(f"Error creating virtual environment: {e}")
            sys.exit(1)

def create_vscode_config():
    """Create VS Code configuration files"""
    vscode_dir = Path(".vscode")
    os.makedirs(vscode_dir, exist_ok=True)

    # Create launch.json for debugging
    launch_file = vscode_dir / "launch.json"
    if not launch_file.exists() or args.force:
        launch_config = """{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Check API Connection",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/test_api_connectivity.py",
            "console": "integratedTerminal",
            "justMyCode": true
        },
        {
            "name": "Run Demo Mode",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/main.py",
            "args": ["--env", "demo"],
            "console": "integratedTerminal",
            "justMyCode": true
        },
        {
            "name": "Run Real Mode",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/main.py",
            "args": ["--env", "real"],
            "console": "integratedTerminal",
            "justMyCode": true
        },
        {
            "name": "Train Only",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/main.py",
            "args": ["--train-only"],
            "console": "integratedTerminal",
            "justMyCode": true
        },
        {
            "name": "Clean Models",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/clean_models.py",
            "args": ["--action", "both"],
            "console": "integratedTerminal",
            "justMyCode": true
        }
    ]
}"""
        with open(launch_file, "w") as f:
            f.write(launch_config)
        print(f"Created VS Code launch configuration: {launch_file}")
    else:
        print(f"VS Code launch configuration already exists: {launch_file}")

    # Create settings.json for Python configuration
    settings_file = vscode_dir / "settings.json"
    if not settings_file.exists() or args.force:
        settings_config = """{
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.formatting.provider": "autopep8",
    "python.formatting.autopep8Args": ["--max-line-length", "100"],
    "editor.formatOnSave": true,
    "python.analysis.extraPaths": [
        "${workspaceFolder}"
    ]
}"""
        with open(settings_file, "w") as f:
            f.write(settings_config)
        print(f"Created VS Code settings configuration: {settings_file}")
    else:
        print(f"VS Code settings configuration already exists: {settings_file}")

def print_usage_instructions():
    """Print usage instructions for the trading bot"""
    print("\nYou can now run the trading bot using the following commands:")

    print("\n=== Checking API Connectivity ===")
    print("python test_api_connectivity.py")

    print("\n=== Training Models ===")
    print("python main.py --train-only")

    print("\n=== Running in Demo Mode ===")
    print("python main.py --env demo")

    print("\n=== Running in Real Mode ===")
    print("1. First set DERIV_REAL_MODE_CONFIRMED=yes in .env file")
    print("2. Then run: python main.py --env real")

    print("\n=== Managing Model Files ===")
    print("python clean_models.py --action both  # Archive old and clean expired")
    print("python clean_models.py --action stats  # Show storage statistics")

    if args.vscode:
        print("\n=== VS Code Integration ===")
        print("Open the project in VS Code:")
        print("code .")
        print("\nUse the Run and Debug tab to select and launch configurations")

if __name__ == "__main__":
    setup_environment()