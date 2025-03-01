"""
Module for processing and preparing market data for ML model
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from deriv_bot.monitor.logger import setup_logger

logger = setup_logger(__name__)

class DataProcessor:
    def __init__(self):
        self.price_scaler = MinMaxScaler()
        self.feature_scaler = MinMaxScaler()
        self.return_scaler = MinMaxScaler(feature_range=(-1, 1))
        self.max_expected_return = 0.005  # 0.5% max expected return for Forex
        self.min_sequence_length = 10  # Minimum acceptable sequence length
        # Define fallback sequence lengths for different data sizes
        self.fallback_sequence_lengths = [
            (100, 20),  # If data points >= 100, use sequence length 20
            (50, 15),   # If data points >= 50, use sequence length 15 
            (30, 10),   # If data points >= 30, use sequence length 10
            (20, 5)     # If data points >= 20, use sequence length 5
        ]
        # Absolute minimum required data points
        self.absolute_min_data_points = 15  # Minimum to create at least a few valid sequences
        # Set a default for feature dimensions
        self.default_feature_dim = 46  # Expected by existing models
        self.base_features = 9
        self.target_features = 46
        self.padding_value = 0.0

    def prepare_data(self, df, sequence_length=30):
        """
        Prepare data for LSTM model using percentage returns

        Args:
            df: DataFrame with OHLCV data
            sequence_length: Number of time steps for LSTM input

        Returns:
            X: Input sequences, shape (samples, sequence_length, features)
            y: Target values, shape (samples,)
            scaler: Fitted scaler for the returns
        """
        try:
            if df is None or df.empty:
                logger.error("Input DataFrame is None or empty")
                return None, None, None

            # Validate sequence length
            if sequence_length <= 0:
                logger.error("Invalid sequence length")
                return None, None, None

            # Validate data length early to provide clear error message
            original_data_length = len(df)
            logger.info(f"Preparing data with shape: {df.shape}")

            # Enhanced early validation - ensure we have the absolute minimum data needed
            if original_data_length < self.absolute_min_data_points:
                logger.error(f"Insufficient data: {original_data_length} points available, absolute minimum required: {self.absolute_min_data_points}")
                return None, None, None

            # Calculate percentage returns for prediction target
            df['returns'] = df['close'].pct_change()

            # Clip returns to realistic range for Forex
            df['returns'] = df['returns'].clip(-self.max_expected_return, self.max_expected_return)
            df.dropna(inplace=True)

            # Check if we still have enough data after removing NaNs
            if len(df) < self.absolute_min_data_points:
                logger.error(f"Insufficient data after cleaning: {len(df)} points available, minimum required: {self.absolute_min_data_points}")
                return None, None, None

            # Log initial statistics
            logger.info(f"Price range - Min: {df['close'].min():.5f}, Max: {df['close'].max():.5f}")
            logger.info(f"Returns range - Min: {df['returns'].min():.5f}, Max: {df['returns'].max():.5f}")

            # Calculate technical indicators
            df = self.add_technical_indicators(df)
            if df is None or df.empty or len(df) < self.absolute_min_data_points:
                logger.error(f"Insufficient data after indicators: {0 if df is None or df.empty else len(df)} points")
                return None, None, None

            logger.info(f"Data shape after indicators: {df.shape}")

            # Scale returns first (target variable)
            returns_data = df['returns'].values.reshape(-1, 1)
            scaled_returns = self.return_scaler.fit_transform(returns_data)
            logger.info(f"Returns scaling params - Min: {self.return_scaler.data_min_}, Max: {self.return_scaler.data_max_}")

            # Adjust sequence length if needed based on available data
            # Ensure sequence_length is not None before passing it
            if sequence_length is None:
                logger.warning("Received None for sequence_length, using default minimum value")
                sequence_length = self.min_sequence_length

            adjusted_sequence_length = self.get_optimal_sequence_length(len(df), sequence_length)

            # Final validation before sequence creation
            if adjusted_sequence_length is None:
                logger.error(f"Could not determine a valid sequence length for {len(df)} data points")
                return None, None, None

            # Create sequences for LSTM with potentially adjusted sequence length
            X, y = self.create_sequences(df.drop('returns', axis=1), scaled_returns, adjusted_sequence_length)
            if X is None or y is None:
                logger.error("Failed to create sequences with specified parameters")
                return None, None, None

            # Validate created sequences have sufficient samples
            if len(X) < 5:  # Minimum number of training samples
                logger.error(f"Insufficient training samples created: {len(X)}. Need at least 5.")
                return None, None, None

            # Check if we need to pad features to match model expectations
            current_feature_dim = X.shape[2]
            if current_feature_dim != self.default_feature_dim:
                logger.warning(f"Feature dimension mismatch: got {current_feature_dim}, expected {self.default_feature_dim}")
                X = self._pad_or_trim_features(X, self.default_feature_dim)
                if X is None:
                    logger.error("Feature adjustment failed")
                    return None, None, None
                logger.info(f"Adjusted feature dimensions to match model expectations: {X.shape}")

            logger.info(f"Created sequences - X shape: {X.shape}, y shape: {y.shape}")
            return X, y, self.return_scaler

        except Exception as e:
            logger.error(f"Error in prepare_data: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None, None, None

    def _pad_or_trim_features(self, X, target_dim):
        """
        Adjust feature dimension to match expected model input

        Args:
            X: Input array of shape (samples, sequence_length, features)
            target_dim: Target feature dimension

        Returns:
            Adjusted array with matching feature dimension
        """
        try:
            if X is None:
                logger.error("Cannot pad/trim None array")
                return None

            current_dim = X.shape[2]

            # If dimensions match, return as is
            if current_dim == target_dim:
                return X

            samples, seq_len, _ = X.shape
            result = np.zeros((samples, seq_len, target_dim))

            if current_dim < target_dim:
                # Pad with zeros
                result[:, :, :current_dim] = X
                logger.info(f"Padded features from {current_dim} to {target_dim}")
            else:
                # Trim excess features
                result = X[:, :, :target_dim]
                logger.info(f"Trimmed features from {current_dim} to {target_dim}")

            return result
        except Exception as e:
            logger.error(f"Error adjusting feature dimensions: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            # Return None to indicate failure, but let calling function handle the error
            return None

    def get_optimal_sequence_length(self, data_length, requested_length):
        """
        Determine optimal sequence length based on available data

        Args:
            data_length: Number of available data points
            requested_length: Requested sequence length

        Returns:
            Optimal sequence length to use or None if insufficient data
        """
        # Ensure we have positive values and valid inputs
        if data_length <= 0:
            logger.error(f"Invalid data_length parameter: {data_length}")
            return None

        if requested_length is None or not isinstance(requested_length, int):
            logger.warning(f"Invalid requested_length: {requested_length}, using minimum sequence length")
            requested_length = self.min_sequence_length

        if requested_length <= 0:
            logger.error(f"Invalid requested_length after validation: {requested_length}")
            return None

        # Calculate minimum required points for creating at least 5 sequences
        min_sequences = 5
        min_required_for_training = requested_length + min_sequences

        # If we have enough data, use the requested length
        if data_length >= min_required_for_training:
            logger.info(f"Using requested sequence length: {requested_length}")
            return requested_length

        # Otherwise, try to find the best fallback sequence length
        # that allows at least 5 sequences to be created
        for threshold, fallback_length in self.fallback_sequence_lengths:
            min_required = fallback_length + min_sequences
            if data_length >= min_required:
                logger.warning(f"Insufficient data for requested length ({data_length} points), "
                              f"falling back to sequence length {fallback_length} to create at least {min_sequences} sequences")
                return fallback_length

        # If we reach here, check if we can use the minimum sequence length
        min_required_absolute = self.min_sequence_length + min_sequences
        if data_length >= min_required_absolute:
            logger.warning(f"Severely limited data ({data_length} points), "
                          f"using minimum sequence length: {self.min_sequence_length}")
            return self.min_sequence_length

        # If we're here, we don't have enough data even for minimum sequence length
        logger.error(f"Insufficient data: {data_length} points available. "
                    f"Need at least {min_required_absolute} for minimum viable training.")
        return None

    def add_technical_indicators(self, df):
        """Add technical indicators to the dataset"""
        try:
            # Save original length for validation
            original_length = len(df)
            if original_length < self.absolute_min_data_points:
                logger.error(f"Insufficient data for calculating indicators: {original_length} points")
                return None

            # Moving averages - Use shorter windows for limited data
            sma_window = min(20, max(5, original_length // 10))
            df['SMA_20'] = df['close'].rolling(window=sma_window).mean()

            # Only add longer MA if we have enough data
            if original_length >= 50:
                df['SMA_50'] = df['close'].rolling(window=50).mean()
            else:
                # Use a shorter window as fallback
                short_window = max(5, original_length // 8)
                df['SMA_50'] = df['close'].rolling(window=short_window).mean()
                logger.warning(f"Using shortened MA window ({short_window}) due to limited data")

            # RSI - Adapt window size based on available data
            rsi_window = min(14, max(5, original_length // 12))
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=rsi_window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_window).mean()
            # Avoid division by zero
            loss = loss.replace(0, 0.00001)
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))

            # Momentum - Adapt period based on data length
            momentum_period = min(10, max(3, original_length // 15))
            df['momentum'] = df['close'].pct_change(periods=momentum_period)

            # Volatility - Adapt window based on data length
            vol_window = min(20, max(5, original_length // 10))
            df['volatility'] = df['close'].pct_change().rolling(window=vol_window).std()

            # Fill NaN values with forward fill, then backward fill for remaining NaNs
            # This is safer than dropping rows when data is limited
            df = df.fillna(method='ffill').fillna(method='bfill')

            # Only drop NaN values if we have enough data to spare
            remaining_after_fill = df.dropna().shape[0]
            if remaining_after_fill >= self.absolute_min_data_points:
                df.dropna(inplace=True)
                logger.info(f"Dropped NaN values, {df.shape[0]} rows remaining")
            else:
                # If dropping would leave too few rows, keep with filled values
                logger.warning(f"Keeping rows with filled NaN values to maintain minimum data requirements")
                # Ensure there are no NaNs left
                df = df.fillna(0)

            if df.empty:
                logger.warning("All data was dropped after calculating indicators")
                return None

            # Validate we didn't lose too much data
            if len(df) < self.absolute_min_data_points:
                logger.warning(f"Too few data points ({len(df)}) remain after indicator calculation")
                return None

            logger.info(f"Indicator calculation complete. Shape: {df.shape}")
            return df

        except Exception as e:
            logger.error(f"Error calculating technical indicators: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def create_sequences(self, data, returns, sequence_length):
        """
        Create input sequences and target returns for LSTM model

        Args:
            data: Feature data (DataFrame or numpy array)
            returns: Target returns (scaled)
            sequence_length: Sequence length for LSTM input (must be integer)

        Returns:
            X: Input sequences, shape (samples, sequence_length, features)
            y: Target values, shape (samples,)
        """
        try:
            # Ensure sequence_length is a valid integer
            if not isinstance(sequence_length, int) or sequence_length <= 0:
                logger.error(f"Invalid sequence length: {sequence_length}. Must be a positive integer.")
                return None, None

            # Validate data
            if isinstance(data, pd.DataFrame):
                data_array = data.values
            else:
                data_array = data

            if data_array is None or len(data_array) == 0:
                logger.error("Input data array is empty")
                return None, None

            original_data_length = len(data_array)

            # Calculate how many sequences we can create
            num_sequences = original_data_length - sequence_length

            # Validate we can create at least a few sequences
            min_sequences = 5
            if num_sequences < min_sequences:
                logger.error(f"Cannot create enough sequences: sequence length {sequence_length} with data length {original_data_length} "
                            f"would only create {num_sequences} sequences, need at least {min_sequences}")
                return None, None

            logger.info(f"Creating {num_sequences} sequences with length {sequence_length}")

            # Create sequences
            X = []
            y = []

            # Validate returns shape and length match data length
            if len(returns) != original_data_length:
                logger.error(f"Returns length ({len(returns)}) doesn't match data length ({original_data_length})")
                return None, None

            for i in range(num_sequences):
                # If we're working with DataFrame, extract rows correctly
                if isinstance(data, pd.DataFrame):
                    sequence = data.iloc[i:(i + sequence_length)].values
                else:
                    sequence = data_array[i:(i + sequence_length)]

                X.append(sequence)

                # Extract the target value and ensure it's a scalar
                try:
                    if len(returns.shape) > 1:
                        target_value = returns[i + sequence_length, 0]  # Extract scalar from 2D array
                    else:
                        target_value = returns[i + sequence_length]  # Already a scalar

                    y.append(target_value)
                except IndexError:
                    logger.error(f"Index error accessing returns at position {i + sequence_length}")
                    return None, None

            # Check if we actually created any sequences
            if len(X) == 0:
                logger.error("No sequences could be created with current parameters")
                return None, None

            X = np.array(X)
            y = np.array(y)

            # Ensure y is 1D
            if len(y.shape) > 1:
                logger.warning(f"Target array has unexpected shape: {y.shape}, reshaping to 1D")
                y = y.ravel()

            # Log sequence statistics
            logger.info(f"Sequence stats - X shape: {X.shape}, y shape: {y.shape}")
            logger.info(f"Target returns range - Min: {y.min():.5f}, Max: {y.max():.5f}")

            return X, y

        except Exception as e:
            logger.error(f"Error creating sequences: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None, None

    def inverse_transform_returns(self, scaled_returns):
        """Convert scaled returns back to percentage returns"""
        try:
            # Handle both 1D and 2D arrays consistently
            if scaled_returns is None:
                logger.error("Cannot inverse transform None value")
                return None

            if len(scaled_returns.shape) == 1:
                scaled_returns = scaled_returns.reshape(-1, 1)

            return self.return_scaler.inverse_transform(scaled_returns)
        except Exception as e:
            logger.error(f"Error in inverse transform: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def pad_features(self, data):
        """Pad feature dimensions to match model expectations"""
        if data.shape[-1] == self.target_features:
            return data
            
        padding_size = self.target_features - data.shape[-1]
        if padding_size <= 0:
            return data
            
        # Create padding with same batch and sequence dimensions
        pad_shape = list(data.shape)
        pad_shape[-1] = padding_size
        padding = np.full(pad_shape, self.padding_value)
        
        return np.concatenate([data, padding], axis=-1)

    def prepare_sequences(self, data, sequence_length=None):
        # ...existing code...
        # After creating sequences but before returning
        if X.shape[-1] != self.target_features:
            X = self.pad_features(X)
        return X, y