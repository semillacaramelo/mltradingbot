# Risk Management Guide

## Risk Parameters

### Position Sizing
- Maximum position size: 2% of account balance
- Position scaling based on prediction confidence
- Dynamic adjustment based on market volatility

### Stop Loss Settings
- Fixed stop loss: 1% from entry (real), 5% (demo)
- Trailing stop loss: Enabled above 2% profit
- Emergency stop: -10% daily drawdown

### Account Protection
- Maximum daily loss: 5% of balance
- Maximum trades per day: 20
- Minimum time between trades: 5 minutes

## Risk Modes

### Demo Mode
```ini
MAX_POSITION_SIZE=100.0
MAX_DAILY_LOSS=50.0
STOP_LOSS_PCT=0.05
```

### Real Mode
```ini
MAX_POSITION_SIZE=25.0
MAX_DAILY_LOSS=10.0
STOP_LOSS_PCT=0.01
```

## Risk Validation
Each trade is validated against:
- Current account balance
- Open position exposure
- Daily loss limits
- Market volatility