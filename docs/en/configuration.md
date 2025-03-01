# Configuration Guide

## Environment Variables

### Required Variables
- `DERIV_API_TOKEN_DEMO`: Your Deriv demo account API token
- `DERIV_API_TOKEN_REAL`: Your Deriv real account API token
- `DERIV_BOT_ENV`: Trading environment (`demo` or `real`)
- `APP_ID`: Deriv application ID (default: 1089)

### Safety Variables
- `DERIV_REAL_MODE_CONFIRMED`: Must be "yes" for real trading

### Training Parameters
- `SEQUENCE_LENGTH`: Data sequence length for training (default: 30)
- `TRAINING_EPOCHS`: Number of training epochs (default: 50)
- `MODEL_SAVE_PATH`: Directory to save models (default: models)

## Configuration File (.env)

Example configuration:
```ini
# Trading Environment
DERIV_BOT_ENV=demo

# API Tokens
DERIV_API_TOKEN_DEMO=your_demo_token_here
DERIV_API_TOKEN_REAL=your_real_token_here

# Safety Configuration
DERIV_REAL_MODE_CONFIRMED=no

# Application Settings
APP_ID=1089

# Training Parameters
SEQUENCE_LENGTH=30
TRAINING_EPOCHS=50
MODEL_SAVE_PATH=models
```

## Security Notes
- Never commit your `.env` file
- Keep API tokens secure
- Always test in demo mode first