# WebSocket Connection Guide

## Connection Management

### Basic Connection
Heartbeat
* Ping interval: 30 seconds
* Connection timeout: 60 seconds
* Auto-reconnect: Enabled

Error Recovery

1. Connection Lost
* Exponential backoff retry
* Maximum 5 retry attempts
* Connection state tracking
2. Rate Limiting
* Maximum 60 requests/minute
* Request queuing
* Cooldown periods
####Message Types

Authentication
```python
{
    "authorize": "YOUR_API_TOKEN",
    "req_id": "auth_1"
}
```
```python
{
    "ticks": "frxEURUSD",
    "subscribe": 1
}
```
```python
{
    "buy": 1,
    "parameters": {
        "amount": 10,
        "basis": "stake",
        "contract_type": "CALL",
        "symbol": "frxEURUSD",
        "duration": 60
    }
}
```