"""
Model Evaluation Module

Purpose:
Handles model validation, evaluation metrics, and quality assessment.
"""

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import logging

logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self, validation_threshold=0.1):
        self.validation_threshold = validation_threshold
        self.metrics = {}

    def validate_training_data(self, data, min_samples=1000):
        """Validate training data quality"""
        if data is None:
            return False, "Data is None"
            
        if len(data) < min_samples:
            return False, f"Insufficient data points: {len(data)} < {min_samples}"
            
        # Check for missing values
        missing_count = data.isnull().sum().sum()
        if missing_count > 0:
            return False, f"Data contains {missing_count} missing values"
            
        # Check for data variance
        for column in data.select_dtypes(include=[np.number]).columns:
            if data[column].std() == 0:
                return False, f"No variance in column {column}"
                
        return True, "Data validation passed"

    def evaluate_model(self, model, X_val, y_val, history=None):
        """Evaluate model performance"""
        try:
            # Predict validation set
            y_pred = model.predict(X_val)
            
            # Calculate metrics
            mse = mean_squared_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)
            
            # Get final validation loss from training history
            val_loss = history.history['val_loss'][-1] if history else None
            
            self.metrics = {
                'mse': mse,
                'r2': r2,
                'val_loss': val_loss
            }
            
            # Check if model meets quality threshold
            meets_threshold = (
                val_loss is None or 
                val_loss < self.validation_threshold
            )
            
            return meets_threshold, self.metrics
            
        except Exception as e:
            logger.error(f"Model evaluation error: {str(e)}")
            return False, {}

    def get_metrics_summary(self):
        """Get formatted metrics summary"""
        return "\n".join([
            "Model Evaluation Metrics:",
            f"MSE: {self.metrics.get('mse', 'N/A'):.6f}",
            f"R² Score: {self.metrics.get('r2', 'N/A'):.6f}",
            f"Validation Loss: {self.metrics.get('val_loss', 'N/A'):.6f}"
        ])

    def validate_prediction(self, prediction, confidence_threshold=0.7):
        """Validate model prediction"""
        if prediction is None:
            return False, "No prediction available"
            
        if 'confidence' not in prediction:
            return False, "No confidence score available"
            
        if prediction['confidence'] < confidence_threshold:
            return False, f"Confidence {prediction['confidence']} below threshold {confidence_threshold}"
            
        return True, "Prediction validated"
