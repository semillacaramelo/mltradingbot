# Trading Guide

## Operation Modes

### Training Mode
Demo Trading
```powershell
# Basic demo run
python main.py --env demo

# Demo with advanced options
python main.py --env demo \
    --symbol frxEURUSD \
    --stake-amount 10 \
    --stop-loss 0.05
```

Real Trading

Before enabling real trading:

1. Verify API tokens
2. Set DERIV_REAL_MODE_CONFIRMED=yes in .env
3. Test thoroughly in demo mode
```powershell
python main.py --env real
```
### Command Line Options
```Table
Option	Description	Default
--env	Trading environment (demo/real)	demo
--symbol	Trading symbol	frxEURUSD
--stake-amount	Trade amount	10.0
--stop-loss	Stop loss percentage	0.01
--debug	Enable debug logging	False
```

### Trade Management
Entry Conditions
* Prediction confidence > 70%
* Market volatility within limits
* Sufficient account balance

Exit Conditions

* Target profit reached
* Stop loss triggered
* Maximum trade duration

