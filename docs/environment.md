# Environment Configuration Guide

## Required Environment Variables

### API Configuration
```ini
# API Tokens (get from https://app.deriv.com/account/api-token)
DERIV_API_TOKEN_DEMO=your_demo_token
DERIV_API_TOKEN_REAL=your_real_token

# Trading Environment 
DERIV_BOT_ENV=demo  # or "real"

# Safety Confirmation
DERIV_REAL_MODE_CONFIRMED=no  # must be "yes" for real trading

# Application ID
APP_ID=1089
```

### Model Configuration
```ini
# Training Parameters
SEQUENCE_LENGTH=30
TRAINING_EPOCHS=50
MODEL_SAVE_PATH=models

# Inference Settings
PREDICTION_THRESHOLD=0.7
MIN_PREDICTION_CONFIDENCE=0.6
```

### Risk Management
```ini
# Position Sizing
MAX_STAKE_AMOUNT=100.0
MAX_DAILY_LOSS=50.0
MAX_OPEN_TRADES=3

# Stop Loss Settings
STOP_LOSS_PCT=0.01
TRAILING_STOP_PCT=0.005
```

## Configuration Files

### .env File
Primary configuration file for environment variables.

### config.json
Optional configuration for advanced settings:

```json
{
    "websocket": {
        "ping_interval": 30,
        "max_retries": 5,
        "reconnect_wait": 10
    },
    "data": {
        "cache_size": 1000,
        "history_days": 30,
        "update_interval": 60
    },
    "monitoring": {
        "log_level": "INFO",
        "save_predictions": true,
        "metrics_interval": 300
    }
}
```

## Environment Setup Script

Run the setup script to configure your environment:

```bash
python environment_setup.py [--vscode] [--demo] [--prod]
```

Options:
- `--vscode`: Configure VS Code settings
- `--demo`: Set up demo environment
- `--prod`: Set up production environment
