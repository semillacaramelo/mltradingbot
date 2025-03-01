# API Reference

## Core Modules

### deriv_bot.data

#### DerivConnector
Main class for API communication with Deriv.com

```python
class DerivConnector:
    async def connect() -> bool
    async def disconnect() -> None
    async def subscribe_to_ticks(symbol: str, callback) -> None
    async def get_candles(symbol: str, count: int = 1000) -> List
    async def get_server_time() -> Dict
```

#### DataFetcher
Handles market data retrieval and caching

```python
class DataFetcher:
    async def fetch_historical_data(symbol: str, interval: int) -> DataFrame
    async def check_trading_enabled(symbol: str) -> bool
    def is_symbol_available(symbol: str) -> bool
```

### deriv_bot.strategy

#### ModelTrainer
Manages model training and validation

```python
class ModelTrainer:
    def train(X: ndarray, y: ndarray) -> History
    def evaluate(X_test: ndarray, y_test: ndarray) -> float
```

#### ModelPredictor
Handles price predictions using trained models

```python
class ModelPredictor:
    def predict(sequence: ndarray) -> Dict
    def load_models(path: str) -> bool
```

## WebSocket API Endpoints

### Authentication
- `authorize`: API token authorization
- `ping`: Connection keepalive
- `time`: Server time synchronization

### Market Data
- `ticks_history`: Historical price data
- `ticks`: Real-time price updates
- `active_symbols`: Available trading symbols

### Trading
- `buy`: Place a new trade
- `sell`: Close an open position
- `proposal`: Get price quotation