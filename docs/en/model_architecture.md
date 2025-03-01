# Model Architecture Guide

## Neural Network Structure

### Input Layer
- Sequence length: 30
- Features per timepoint: 46
- Batch size: 32 (configurable)

### LSTM Layers
Dense Layers
```Dense(32, activation='relu')
Dense(16, activation='relu')
Dense(1, activation='tanh')
```
### Model Types
Short-term Model
* 1-5 minute predictions
* Higher update frequency
* Focused on price action

Medium-term Model

* 5-30 minute predictions
* Technical indicators
* Market regimes

Long-term Model

* 30+ minute predictions
* Multiple timeframes
* Trend analysis
### Model Files

```plaintext
models/
├── short_term_YYYYMMDD_HHMMSS.keras
├── medium_term_YYYYMMDD_HHMMSS.keras
└── long_term_YYYYMMDD_HHMMSS.keras
```
### Training Configuration
```python
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2
```