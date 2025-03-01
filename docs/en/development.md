
# Development Guide

## Project Structure
```
trading_bot_replit_v4/
├── deriv_bot/              # Main package
│   ├── data/              # Data handling
│   ├── strategy/          # Trading strategies
│   ├── risk/             # Risk management
│   ├── monitor/          # Monitoring & logging
│   └── utils/            # Utilities
├── docs/                  # Documentation
├── tests/                # Unit tests
└── models/               # Trained models
```

## Development Setup

1. **Environment Setup**
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

2. **VS Code Configuration**
```bash
python environment_setup.py --vscode
```

## Development Workflow

1. **Code Changes**
- Follow PEP 8 style guide
- Add type hints
- Update docstrings
- Write unit tests

2. **Testing**
```bash
# Run API connectivity test
python test_api_connectivity.py

# Run in demo mode
python main.py --env demo --debug
```

3. **Model Development**
- Use `ModelTrainer` class for experiments
- Test predictions with `ModelPredictor`
- Validate against historical data
```