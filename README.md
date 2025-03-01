# ML Trading Bot for Deriv.com

A machine learning-powered trading bot for automated trading on Deriv.com.

[En Español](docs/es/README.md) | [English](docs/en/README.md)

## Quick Start

```bash
# Clone the repository
git clone <repository-url>
cd deriv-ml-trading-bot

# Set up environment (automatic configuration)
python environment_setup.py --vscode

# Verify API connection
python test_api_connectivity.py

# Train models
python main.py --train-only

# Run in demo mode
python main.py --env demo
```

## Requirements
- Python 3.11+
- Deriv.com account with API tokens
- Dependencies listed in requirements.txt

## Basic Usage

### Demo Mode
```bash
python main.py --env demo --symbol frxEURUSD --stake-amount 20
```

### Real Mode
**WARNING**: Uses real funds. Ensure you understand the risks.
```bash
python main.py --env real --symbol frxEURUSD --stake-amount 10
```

## Documentation Structure

- [Installation Guide](docs/INSTALLATION.md) - Detailed setup instructions
- [Documentation](docs/DOCUMENTATION.md) - Complete system documentation
- [VS Code Guide](docs/VSCODE_SETUP.md) - VS Code specific setup
- [API Reference](docs/API.md) - API integration details
- [Configuration](docs/CONFIGURATION.md) - Configuration options
- [Troubleshooting](docs/TROUBLESHOOTING.md) - Common issues and solutions

## Support & Contributing
Please check our [Contributing Guide](CONTRIBUTING.md) for details on how to contribute to this project.

## License
This project is licensed under [LICENSE INFORMATION]