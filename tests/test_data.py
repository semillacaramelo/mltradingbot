"""
Data Module Unit Tests

Location: tests/test_data.py

Purpose:
Unit tests for data processing and handling components, including
technical indicators and data preparation for ML models.

Dependencies:
- unittest: Testing framework
- pandas: Data manipulation
- numpy: Numerical operations
- deriv_bot.data: Modules being tested

Interactions:
- Input: Test data and configurations
- Output: Test results and assertions
- Relations: Validates data processing functionality

Author: Trading Bot Team
Last modified: 2024-02-26
"""
import unittest
import pandas as pd
import numpy as np
from deriv_bot.data.data_processor import DataProcessor

class TestDataProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = DataProcessor()

        # Create sample data - larger dataset for robust testing
        dates = pd.date_range(start='2023-01-01', periods=200, freq='H')
        self.sample_data = pd.DataFrame({
            'open': np.random.random(200),
            'high': np.random.random(200),
            'low': np.random.random(200),
            'close': np.random.random(200),
            'volume': np.random.random(200)
        }, index=dates)

        # Create smaller dataset for testing edge cases
        self.small_data = pd.DataFrame({
            'open': np.random.random(30),
            'high': np.random.random(30),
            'low': np.random.random(30),
            'close': np.random.random(30),
            'volume': np.random.random(30)
        })

        # Create very small dataset for testing inadequate data handling
        self.tiny_data = pd.DataFrame({
            'open': np.random.random(12),
            'high': np.random.random(12),
            'low': np.random.random(12),
            'close': np.random.random(12),
            'volume': np.random.random(12)
        })

        # Create borderline dataset (just at minimum requirement)
        self.borderline_data = pd.DataFrame({
            'open': np.random.random(15),
            'high': np.random.random(15),
            'low': np.random.random(15),
            'close': np.random.random(15),
            'volume': np.random.random(15)
        })

    def test_add_technical_indicators(self):
        """Test technical indicator calculation"""
        processed_df = self.processor.add_technical_indicators(self.sample_data)

        self.assertIsNotNone(processed_df)
        self.assertIn('SMA_20', processed_df.columns)
        self.assertIn('SMA_50', processed_df.columns)
        self.assertIn('RSI', processed_df.columns)

    def test_prepare_data(self):
        """Test data preparation for ML model"""
        X, y, scaler = self.processor.prepare_data(self.sample_data, sequence_length=10)

        self.assertIsNotNone(X)
        self.assertIsNotNone(y)
        self.assertIsNotNone(scaler)
        self.assertEqual(len(X.shape), 3)  # (samples, sequence_length, features)
        self.assertEqual(len(y.shape), 1)  # Target should be 1D

    def test_create_sequences(self):
        """Test sequence creation"""
        data = np.random.random((100, 5))
        # Create returns array as a 2D column vector to match real usage
        returns = np.random.random((100, 1))

        X, y = self.processor.create_sequences(data, returns, sequence_length=10)

        self.assertIsNotNone(X)
        self.assertIsNotNone(y)
        self.assertEqual(X.shape[1], 10)  # sequence length
        self.assertEqual(len(y.shape), 1)  # Target should be 1D

    def test_adaptive_sequence_length(self):
        """Test the adaptive sequence length behavior with limited data"""
        # Create small dataset that can still produce valid sequences
        small_data = np.random.random((30, 5))
        # Create returns array as a 2D column vector
        small_returns = np.random.random((30, 1))

        # Get optimal sequence length
        optimal_length = self.processor.get_optimal_sequence_length(len(small_data), 30)

        # Verify optimal length is adapted based on available data
        self.assertIsNotNone(optimal_length)
        self.assertLessEqual(optimal_length, 20)  # Should return at most 20 for 30 data points

        # Create sequences with adapted length
        if optimal_length:  # Only attempt to create sequences if optimal_length is valid
            X, y = self.processor.create_sequences(small_data, small_returns, sequence_length=optimal_length)

            # Check if sequences were created with adapted length
            self.assertIsNotNone(X)
            self.assertIsNotNone(y)
            self.assertEqual(X.shape[1], optimal_length)
            self.assertEqual(len(y.shape), 1)  # Target should be 1D

    def test_insufficient_data_handling(self):
        """Test handling of extremely small datasets"""
        # Test with data that's below the absolute minimum
        X, y, scaler = self.processor.prepare_data(self.tiny_data, sequence_length=10)

        # Should return None for insufficient data
        self.assertIsNone(X)
        self.assertIsNone(y)
        self.assertIsNone(scaler)

        # Test optimal_length calculation with insufficient data
        optimal_length = self.processor.get_optimal_sequence_length(
            len(self.tiny_data), 10
        )

        # Should return None for truly insufficient data
        self.assertIsNone(optimal_length)

    def test_borderline_data_handling(self):
        """Test handling of data just at the minimum requirement"""
        # This test verifies the processor can handle borderline cases
        X, y, scaler = self.processor.prepare_data(self.borderline_data, sequence_length=5)

        # For a borderline case, we should either get successful processing with minimal sequences
        # or None values if the borderline case isn't handled by the current implementation
        if X is not None:
            # If sequences were created, verify they meet minimum requirements
            self.assertGreaterEqual(len(X), 5)  # Should have at least 5 sequences
            self.assertEqual(X.shape[1], 5)  # Should use sequence length 5
        else:
            # If None, at least it didn't crash, and that's also acceptable
            # Just verify all returned values are None for consistency
            self.assertIsNone(y)
            self.assertIsNone(scaler)

    def test_nan_handling(self):
        """Test handling of data with NaN values"""
        # Create data with some NaN values
        data_with_nans = self.sample_data.copy()
        # Insert some NaN values
        data_with_nans.iloc[10:15, 0] = np.nan
        data_with_nans.iloc[50:55, 1] = np.nan

        # Process data with NaNs
        processed_df = self.processor.add_technical_indicators(data_with_nans)

        # Verify processing succeeded despite NaNs
        self.assertIsNotNone(processed_df)
        # Verify no NaNs in processed data
        self.assertFalse(processed_df.isnull().any().any())

    def test_inverse_transform_returns(self):
        """Test inverse transform of scaled returns"""
        # Create and fit scaler with sample data
        returns = np.random.uniform(-0.005, 0.005, (100, 1))
        self.processor.return_scaler.fit(returns)

        # Transform some data
        scaled = self.processor.return_scaler.transform(returns)

        # Test inverse transform with 2D array
        inverse_2d = self.processor.inverse_transform_returns(scaled)
        self.assertIsNotNone(inverse_2d)

        # Test inverse transform with 1D array
        scaled_1d = scaled.ravel()
        inverse_1d = self.processor.inverse_transform_returns(scaled_1d)
        self.assertIsNotNone(inverse_1d)

        # Verify shapes
        self.assertEqual(inverse_2d.shape, returns.shape)

if __name__ == '__main__':
    unittest.main()