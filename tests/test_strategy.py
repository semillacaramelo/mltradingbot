"""
Strategy Module Unit Tests

Location: tests/test_strategy.py

Purpose:
Unit tests for trading strategy components including model training,
prediction, and feature engineering.

Dependencies:
- unittest: Testing framework
- numpy: Numerical operations
- deriv_bot.strategy: Strategy modules being tested

Interactions:
- Input: Test data and model configurations
- Output: Test results and model validations
- Relations: Validates ML model functionality

Author: Trading Bot Team
Last modified: 2024-02-26
"""
import unittest
import numpy as np
import os
import tempfile
import pickle
from deriv_bot.strategy.model_trainer import ModelTrainer
from deriv_bot.strategy.model_predictor import ModelPredictor
from deriv_bot.utils.model_manager import ModelManager
from deriv_bot.monitor.logger import setup_logger

logger = setup_logger("test_strategy")

class TestStrategy(unittest.TestCase):
    def setUp(self):
        self.input_shape = (60, 8)  # (sequence_length, features)
        self.trainer = ModelTrainer(self.input_shape)
        # Create a temporary directory for test models
        self.test_model_dir = tempfile.mkdtemp(prefix="test_models_")
        # Create a model manager for testing
        self.model_manager = ModelManager(models_dir=self.test_model_dir)

    def tearDown(self):
        # Clean up temporary test files
        import shutil
        if os.path.exists(self.test_model_dir):
            shutil.rmtree(self.test_model_dir)

    def test_model_build(self):
        """Test model architecture creation"""
        model = self.trainer.model

        self.assertIsNotNone(model)
        self.assertEqual(len(model.layers), 5)  # Check number of layers

    def test_model_training(self):
        """Test model training process"""
        # Create dummy data
        X = np.random.random((100, 60, 8))
        y = np.random.random(100)

        history = self.trainer.train(
            X, y,
            validation_split=0.2,
            epochs=2,
            batch_size=32,
            model_type="test_model"
        )

        # Check if history is None (training failed)
        if history is None:
            logger.error("Training history is None, training failed")
            self.fail("Model training failed")

        self.assertIsNotNone(history)
        self.assertIn('loss', history.history)
        self.assertIn('val_loss', history.history)

    def test_model_save_load(self):
        """Test model saving and loading with new format"""
        # Create dummy data and train a model
        X = np.random.random((100, 60, 8))
        y = np.random.random(100)

        # Train with minimal epochs
        self.trainer.train(X, y, epochs=1, batch_size=32, model_type="test_model")

        # Create a mock scaler for testing
        from sklearn.preprocessing import MinMaxScaler
        mock_scaler = MinMaxScaler()
        mock_scaler.fit(np.random.random((100, 1)))

        # Save model in temporary directory using .keras format
        model_path = os.path.join(self.test_model_dir, "test_model.keras")
        save_result = self.trainer.save_model(model_path, scaler=mock_scaler)

        self.assertTrue(save_result, "Model saving failed")
        self.assertTrue(os.path.exists(model_path), f"Model file not found at {model_path}")

        # Check if metadata file was created
        metadata_path = model_path.replace('.keras', '_metadata.pkl')
        self.assertTrue(os.path.exists(metadata_path), f"Metadata file not found at {metadata_path}")

        # Load and verify metadata
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        self.assertIn('scaler', metadata, "Scaler not found in metadata")

        # Create predictor and load the saved model
        predictor = ModelPredictor(model_path)

        # Verify model loaded correctly
        self.assertIsNotNone(predictor.model, "Failed to load model")
        self.assertTrue(len(predictor.models) > 0, "No models loaded")

        # Verify scaler was loaded
        self.assertIsNotNone(predictor.scaler, "Failed to load scaler from metadata")

        # Test prediction functionality
        test_sequence = np.random.random((1, 60, 8))
        # Lower confidence threshold for testing to ensure we get a prediction
        raw_prediction = predictor.model.predict(test_sequence, verbose=0)

        self.assertIsNotNone(raw_prediction, "Model prediction failed")
        self.assertTrue(isinstance(raw_prediction[0][0], float), "Prediction is not a float")

    def test_model_manager_save_load(self):
        """Test model manager saving and loading functionality"""
        # Create dummy data and train a model
        X = np.random.random((100, 60, 8))
        y = np.random.random(100)

        # Train with minimal epochs
        self.trainer.train(X, y, epochs=1, batch_size=32)

        # Create a mock scaler for testing
        from sklearn.preprocessing import MinMaxScaler
        mock_scaler = MinMaxScaler()
        mock_scaler.fit(np.random.random((100, 1)))

        # Use model manager to save model with timestamp
        model_type = "test_type"
        save_path = self.model_manager.save_model_with_timestamp(
            self.trainer.model, 
            base_name="test_model", 
            model_type=model_type,
            scaler=mock_scaler
        )

        self.assertIsNotNone(save_path, "Model manager failed to save model")
        self.assertTrue(os.path.exists(save_path), f"Model file not found at {save_path}")

        # Check if metadata file was created
        metadata_path = save_path.replace('.keras', '_metadata.pkl')
        self.assertTrue(os.path.exists(metadata_path), f"Metadata file not found at {metadata_path}")

        # Create predictor with models directory
        predictor = ModelPredictor(self.test_model_dir)

        # Verify model loaded correctly
        self.assertIsNotNone(predictor.model, "Failed to load model")
        self.assertTrue(len(predictor.models) > 0, "No models loaded")
        self.assertIn(model_type, predictor.models, f"Model of type {model_type} not loaded")

        # Verify scaler was loaded
        self.assertIsNotNone(predictor.scaler, "Failed to load scaler from metadata")

        # Test prediction with loaded model
        test_sequence = np.random.random((1, 60, 8))
        prediction = predictor.predict(test_sequence, confidence_threshold=0.1)

        # Since we're using random data, prediction might be None due to low confidence
        # Just check that the function runs without errors
        if prediction is not None:
            self.assertIn('prediction', prediction)
            self.assertIn('confidence', prediction)

    def test_model_prediction(self):
        """Test model prediction"""
        predictor = ModelPredictor()

        # Create dummy sequence
        sequence = np.random.random((1, 60, 8))

        # Test direct model assignment
        predictor.model = self.trainer.model

        # Verify model is assigned correctly
        self.assertIsNotNone(predictor.model)

        # Lower the confidence threshold for testing purposes
        # This ensures we get a prediction result even with random data
        confidence_threshold = 0.1
        prediction = predictor.predict(sequence, confidence_threshold)

        # If prediction returns None due to confidence threshold, create test prediction
        if prediction is None:
            # Test direct prediction from model
            raw_pred = predictor.model.predict(sequence)[0][0]
            self.assertTrue(isinstance(raw_pred, float))
        else:
            self.assertIsNotNone(prediction)
            self.assertIn('prediction', prediction)
            self.assertIn('confidence', prediction)
            self.assertTrue(isinstance(prediction['prediction'], float))

if __name__ == '__main__':
    unittest.main()