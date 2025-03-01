# API Reference Documentation

## Core API Classes

### DerivBot
Main bot class handling trading operations.

```python
class DerivBot:
    def __init__(self, api_token: str, env: str = "demo"):
        """Initialize trading bot.
        
        Args:
            api_token: Deriv API token
            env: Trading environment ("demo"/"real")
        """
        pass
        
    async def connect(self):
        """Establish WebSocket connection."""
        pass
        
    async def subscribe_to_ticks(self, symbol: str):
        """Subscribe to price updates."""
        pass
```

### ModelPredictor 
ML model prediction interface.

```python
class ModelPredictor:
    def __init__(self, model_path: str):
        """Load trained model.
        
        Args:
            model_path: Path to saved model file
        """
        pass
        
    def predict(self, data: np.ndarray) -> float:
        """Generate price movement prediction.
        
        Args:
            data: Input features
            
        Returns:
            Prediction value between -1 and 1
        """
        pass
```

## API Endpoints

### WebSocket API

Base URL: `wss://ws.binaryws.com/websockets/v3`

#### Authorization
```json
{
    "authorize": "${api_token}",
    "req_id": "auth_1" 
}
```

#### Market Data
```json
{
    "ticks": "${symbol}",
    "subscribe": 1
}
```

#### Trading
```json 
{
    "buy": 1,
    "parameters": {
        "amount": "${stake}",
        "basis": "stake",
        "contract_type": "${type}",
        "symbol": "${symbol}",
        "duration": "${duration}"
    }
}
```

For complete API documentation, see [Deriv API docs](https://api.deriv.com/).
