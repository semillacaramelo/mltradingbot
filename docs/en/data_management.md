```markdown
```
# Data Management Guide

## Data Flow
### Historical Data

* Storage: data/historical/
* Format: Parquet files
* Retention: 90 days
* Resolution: 1-minute candles

### Cache Management
```python
# Cache configuration
MAX_CACHE_SIZE = 100  # MB
CACHE_EXPIRY = 3600   # seconds
MIN_CACHE_ITEMS = 1000
```
### Data Validation Rules

1. Price Validation
* Non-zero positive values
* Within typical ranges
* Recent timestamp
2. Sequence Validation
* Minimum length: 30
* No missing values
* Chronological order

Data Processing Pipeline
1. Raw Data Collection

```python
await data_fetcher.fetch_historical_data(
    symbol="frxEURUSD",
    interval=60,
    count=1000
)
```
2. Feature Engineering
* Technical indicators
* Market regime features
* Volatility metrics

3. Sequence Preparation
* Normalization
* Sequence creation
* Feature scaling