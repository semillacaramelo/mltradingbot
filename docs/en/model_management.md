# Model Management Guide

## Directory Structure
```
models/
├── active/              # Currently used models
├── archive/            # Historical model versions
└── metadata/           # Model configuration and metrics
```

## Model Types
1. **Short-term Predictor**
   - 1-5 minute predictions
   - Higher update frequency
   - Uses most recent data points

2. **Medium-term Predictor**
   - 5-30 minute predictions
   - Balanced between speed and accuracy
   - Uses technical indicators

3. **Long-term Predictor**
   - 30+ minute predictions
   - Focus on trend analysis
   - Uses multiple timeframe data

## Model Maintenance

### Training New Models
```bash
# Train all model types
python main.py --train-only

# Train specific model type
python main.py --train-only --model-type short_term
```

### Managing Model Files
```bash
# View model statistics
python clean_models.py --action stats

# Archive old models
python clean_models.py --action archive --keep 5

# Clean expired archives
python clean_models.py --action clean --days 30
```

### Model Validation
Before deployment, models are validated against:
- Minimum accuracy threshold
- Maximum prediction deviation
- Historical performance metrics