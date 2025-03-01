# Quick Start Guide

## Prerequisites
- Python 3.11 or higher
- Deriv.com account
- API tokens (demo/real)

## Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/trading_bot_replit_v4.git
cd trading_bot_replit_v4
```

2. Run setup script
```bash
python environment_setup.py
```

3. Test API connectivity
```bash
python test_api_connectivity.py
```

## Configuration

1. Get API tokens from Deriv.com:
   - Go to [Deriv API Token Page](https://app.deriv.com/account/api-token)
   - Create tokens with required permissions:
     - Read
     - Trade
     - Payments
     - Admin

2. Configure environment:
   - Edit `.env` file
   - Add your API tokens
   - Set `DERIV_BOT_ENV=demo` for testing

## Usage

### Demo Mode
```bash
python main.py --env demo
```

### Training Mode
```bash
python main.py --train-only
```

### Real Trading
```bash
# First set in .env:
DERIV_BOT_ENV=real
DERIV_REAL_MODE_CONFIRMED=yes

# Then run:
python main.py --env real
```

## Next Steps
- Review [Configuration Guide](configuration.md)
- Study [Risk Management](risk_management.md)
- Check [Troubleshooting](troubleshooting.md)