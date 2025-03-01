# Model Maintenance Guide

## Automatic Retraining
The bot can retrain models automatically during execution:
```bash
python main.py --train-interval 4  # Retrain every 4 hours
```

## Manual Retraining
```bash
python main.py --train-only
```

## Model File Management
```bash
# View storage statistics
python clean_models.py --action stats

# Archive old models (keep 5 most recent)
python clean_models.py --action archive --keep 5

# Delete archived files older than 30 days
python clean_models.py --action clean --days 30

# Perform both operations
python clean_models.py --action both
```

## TensorFlow Compatibility

- Uses native `.keras` format for models
- Custom checkpoint mechanism for compatibility
- TensorFlow 2.10+ recommended

### Common TensorFlow Error Solutions
If you encounter `The following argument(s) are not supported with the native Keras format: ['options']`:
- Custom BestModelCheckpoint implementation handles this
- No code modification needed
- Solution integrated into system