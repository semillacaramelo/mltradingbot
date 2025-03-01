"""
Module for making predictions using the trained model
"""
"""
Model Prediction Module

Location: deriv_bot/strategy/model_predictor.py

Purpose:
Handles price predictions using trained models.
Implements prediction logic, confidence scoring, and ensemble methods.

Dependencies:
- numpy: Numerical computations
- tensorflow: Model inference
- deriv_bot.monitor.logger: Logging functionality

Interactions:
- Input: Current market data
- Output: Price predictions and confidence scores
- Relations: Used by strategy executor for trade decisions

Author: Trading Bot Team
Last modified: 2024-02-26
"""
import os
import numpy as np
import glob
import pickle
from tensorflow.keras.models import load_model
from deriv_bot.monitor.logger import setup_logger

logger = setup_logger(__name__)

class ModelPredictor:
    def __init__(self, model_path=None, scaler=None):
        self.models = {}
        self._single_model = None  # Private attribute for single model access
        self.max_expected_return = 0.005  # 0.5% max return for Forex
        self.scaler = scaler  # Store the scaler for denormalizing predictions
        if model_path:
            self.load_models(model_path)

    @property
    def model(self):
        """Getter for model - returns the default model for backward compatibility"""
        if self._single_model is not None:
            return self._single_model
        elif 'default' in self.models:
            return self.models['default']
        elif len(self.models) > 0:
            # Return the first model as default if there's no 'default' key
            return list(self.models.values())[0]
        return None

    @model.setter
    def model(self, model_instance):
        """Setter for model - allows setting model directly for backward compatibility"""
        self._single_model = model_instance
        self.models['default'] = model_instance
        logger.info("Model assigned directly to predictor")

    def load_models(self, base_path):
        """Load all models in the ensemble"""
        try:
            # Initialize model loading results
            models_found = False

            # Check if base_path is a directory or a file
            if os.path.isdir(base_path):
                logger.info(f"Looking for models in directory: {base_path}")
                # If it's a directory, try to load model types
                model_types = ['short_term', 'medium_term', 'long_term']

                # Try to load models with different model types
                for model_type in model_types:
                    # Look for model files with the model_type in the name (both .keras and .h5)
                    model_loaded = self._try_load_model_by_type(base_path, model_type)
                    if model_loaded:
                        models_found = True

                # If no type-specific models found, try to load a generic model
                if not models_found:
                    models_found = self._try_load_generic_model(base_path)
            else:
                # If it's a file, try direct loading with both formats
                models_found = self._try_load_direct_model(base_path)

            if not self.models:
                logger.error(f"No models found at path: {base_path}")
                return False

            # Try to load scaler if a metadata file exists
            self._try_load_scaler(base_path)

            logger.info(f"Successfully loaded {len(self.models)} model(s)")
            return True

        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            return False

    def _try_load_scaler(self, base_path):
        """Try to load scaler from metadata file if it exists"""
        try:
            # Check if base_path is a model file or directory
            if os.path.isfile(base_path):
                # If it's a file, construct the metadata filename
                if base_path.endswith('.h5'):
                    metadata_path = base_path.replace('.h5', '_metadata.pkl')
                elif base_path.endswith('.keras'):
                    metadata_path = base_path.replace('.keras', '_metadata.pkl')
                else:
                    metadata_path = f"{base_path}_metadata.pkl"
            else:
                # If it's a directory, look for metadata files
                metadata_files = glob.glob(os.path.join(base_path, '*_metadata.pkl'))
                if not metadata_files:
                    return False
                # Use the newest metadata file
                metadata_path = max(metadata_files, key=os.path.getmtime)

            # Load metadata if it exists
            if os.path.exists(metadata_path):
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                    if 'scaler' in metadata:
                        self.scaler = metadata['scaler']
                        logger.info(f"Loaded scaler from {metadata_path}")
                        return True
            return False
        except Exception as e:
            logger.warning(f"Error loading scaler: {str(e)}")
            return False

    def _try_load_model_by_type(self, base_path, model_type):
        """Helper to try loading a model by type with multiple extensions"""
        # Prioritize .keras over .h5 format
        for ext in ['.keras', '.h5']:
            # Look for model files with explicit naming patterns
            possible_paths = [
                os.path.join(base_path, f"best_model_{model_type}{ext}"),
                os.path.join(base_path, f"{model_type}_model{ext}"),
                os.path.join(base_path, f"{model_type}{ext}")
            ]

            for path in possible_paths:
                if os.path.exists(path):
                    try:
                        self.models[model_type] = load_model(path)
                        logger.info(f"Loaded {model_type} model from {path}")

                        # Check for metadata file
                        metadata_path = path.replace(ext, '_metadata.pkl')
                        if os.path.exists(metadata_path):
                            with open(metadata_path, 'rb') as f:
                                metadata = pickle.load(f)
                                if 'scaler' in metadata and self.scaler is None:
                                    self.scaler = metadata['scaler']
                                    logger.info(f"Loaded scaler from {metadata_path}")
                        return True
                    except Exception as load_err:
                        logger.warning(f"Failed to load {path}: {str(load_err)}")

            # Also look for timestamped model files with the model type
            timestamped_pattern = os.path.join(base_path, f"*{model_type}*{ext}")
            timestamped_files = glob.glob(timestamped_pattern)

            if timestamped_files:
                # Sort by modification time (newest first)
                newest_file = max(timestamped_files, key=os.path.getmtime)
                try:
                    self.models[model_type] = load_model(newest_file)
                    logger.info(f"Loaded {model_type} model from {newest_file}")

                    # Check for metadata file
                    metadata_path = newest_file.replace(ext, '_metadata.pkl')
                    if os.path.exists(metadata_path):
                        with open(metadata_path, 'rb') as f:
                            metadata = pickle.load(f)
                            if 'scaler' in metadata and self.scaler is None:
                                self.scaler = metadata['scaler']
                                logger.info(f"Loaded scaler from {metadata_path}")
                    return True
                except Exception as load_err:
                    logger.warning(f"Failed to load {newest_file}: {str(load_err)}")

        return False

    def _try_load_generic_model(self, base_path):
        """Helper to try loading a generic model from directory"""
        # Prioritize .keras over .h5 format
        for ext in ['.keras', '.h5']:
            # Try best_model files first
            best_model_path = os.path.join(base_path, f"best_model{ext}")
            if os.path.exists(best_model_path):
                try:
                    self.models['default'] = load_model(best_model_path)
                    logger.info(f"Loaded best model from {best_model_path}")

                    # Check for metadata file
                    metadata_path = best_model_path.replace(ext, '_metadata.pkl')
                    if os.path.exists(metadata_path):
                        with open(metadata_path, 'rb') as f:
                            metadata = pickle.load(f)
                            if 'scaler' in metadata and self.scaler is None:
                                self.scaler = metadata['scaler']
                                logger.info(f"Loaded scaler from {metadata_path}")
                    return True
                except Exception as load_err:
                    logger.warning(f"Failed to load {best_model_path}: {str(load_err)}")

            # Then try generic model files
            generic_model_path = os.path.join(base_path, f"model{ext}")
            if os.path.exists(generic_model_path):
                try:
                    self.models['default'] = load_model(generic_model_path)
                    logger.info(f"Loaded generic model from {generic_model_path}")

                    # Check for metadata file
                    metadata_path = generic_model_path.replace(ext, '_metadata.pkl')
                    if os.path.exists(metadata_path):
                        with open(metadata_path, 'rb') as f:
                            metadata = pickle.load(f)
                            if 'scaler' in metadata and self.scaler is None:
                                self.scaler = metadata['scaler']
                                logger.info(f"Loaded scaler from {metadata_path}")
                    return True
                except Exception as load_err:
                    logger.warning(f"Failed to load {generic_model_path}: {str(load_err)}")

        # If no specific model found, try any model file
        all_model_files = []
        for ext in ['.keras', '.h5']:
            all_model_files.extend(glob.glob(os.path.join(base_path, f"*{ext}")))

        if all_model_files:
            # Sort by modification time (newest first)
            newest_model = max(all_model_files, key=os.path.getmtime)
            try:
                self.models['default'] = load_model(newest_model)
                logger.info(f"Loaded newest model from {newest_model}")

                # Check for metadata file
                ext = '.keras' if newest_model.endswith('.keras') else '.h5'
                metadata_path = newest_model.replace(ext, '_metadata.pkl')
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'rb') as f:
                        metadata = pickle.load(f)
                        if 'scaler' in metadata and self.scaler is None:
                            self.scaler = metadata['scaler']
                            logger.info(f"Loaded scaler from {metadata_path}")
                return True
            except Exception as load_err:
                logger.warning(f"Failed to load {newest_model}: {str(load_err)}")

        return False

    def _try_load_direct_model(self, base_path):
        """Helper to try loading a model directly from a file path"""
        # First check if the specified path exists
        if os.path.exists(base_path):
            try:
                self.models['default'] = load_model(base_path)
                logger.info(f"Loaded model from {base_path}")

                # Check for metadata file
                if base_path.endswith('.h5'):
                    metadata_path = base_path.replace('.h5', '_metadata.pkl')
                elif base_path.endswith('.keras'):
                    metadata_path = base_path.replace('.keras', '_metadata.pkl')
                else:
                    metadata_path = f"{base_path}_metadata.pkl"

                if os.path.exists(metadata_path):
                    with open(metadata_path, 'rb') as f:
                        metadata = pickle.load(f)
                        if 'scaler' in metadata and self.scaler is None:
                            self.scaler = metadata['scaler']
                            logger.info(f"Loaded scaler from {metadata_path}")
                return True
            except Exception as load_err:
                logger.warning(f"Failed to load {base_path}: {str(load_err)}")

        # If not found, try adding extensions
        for ext in ['.keras', '.h5']:
            # Try adding extension if not present
            if not (base_path.endswith('.keras') or base_path.endswith('.h5')):
                model_path = f"{base_path}{ext}"
                if os.path.exists(model_path):
                    try:
                        self.models['default'] = load_model(model_path)
                        logger.info(f"Loaded model from {model_path}")

                        # Check for metadata file
                        metadata_path = model_path.replace(ext, '_metadata.pkl')
                        if os.path.exists(metadata_path):
                            with open(metadata_path, 'rb') as f:
                                metadata = pickle.load(f)
                                if 'scaler' in metadata and self.scaler is None:
                                    self.scaler = metadata['scaler']
                                    logger.info(f"Loaded scaler from {metadata_path}")
                        return True
                    except Exception as load_err:
                        logger.warning(f"Failed to load {model_path}: {str(load_err)}")

        # Try changing extension (.h5 to .keras or vice versa)
        if base_path.endswith('.h5'):
            keras_path = base_path[:-3] + '.keras'
            if os.path.exists(keras_path):
                try:
                    self.models['default'] = load_model(keras_path)
                    logger.info(f"Loaded model from {keras_path}")

                    # Check for metadata file
                    metadata_path = keras_path.replace('.keras', '_metadata.pkl')
                    if os.path.exists(metadata_path):
                        with open(metadata_path, 'rb') as f:
                            metadata = pickle.load(f)
                            if 'scaler' in metadata and self.scaler is None:
                                self.scaler = metadata['scaler']
                                logger.info(f"Loaded scaler from {metadata_path}")
                    return True
                except Exception as load_err:
                    logger.warning(f"Failed to load {keras_path}: {str(load_err)}")
        elif base_path.endswith('.keras'):
            h5_path = base_path[:-6] + '.h5'
            if os.path.exists(h5_path):
                try:
                    self.models['default'] = load_model(h5_path)
                    logger.info(f"Loaded model from {h5_path}")

                    # Check for metadata file
                    metadata_path = h5_path.replace('.h5', '_metadata.pkl')
                    if os.path.exists(metadata_path):
                        with open(metadata_path, 'rb') as f:
                            metadata = pickle.load(f)
                            if 'scaler' in metadata and self.scaler is None:
                                self.scaler = metadata['scaler']
                                logger.info(f"Loaded scaler from {metadata_path}")
                    return True
                except Exception as load_err:
                    logger.warning(f"Failed to load {h5_path}: {str(load_err)}")

        return False

    def predict(self, sequence, confidence_threshold=0.6):
        """
        Make ensemble prediction with confidence score

        Args:
            sequence: Input sequence of shape (1, sequence_length, features)
            confidence_threshold: Minimum confidence required for valid prediction
        """
        try:
            # Add shape validation 
            expected_shape = self.model.get_layer(index=0).input_shape[1:]
            actual_shape = sequence.shape[1:]
            
            if expected_shape != actual_shape:
                logger.error(f"Input shape mismatch: expected {expected_shape}, got {actual_shape}")
                return None

            if not self.models and not self._single_model:
                raise ValueError("Models not loaded")

            # Use the directly assigned model if available
            if self._single_model is not None and not self.models:
                self.models['default'] = self._single_model

            # Get predictions from all models (returns as percentage change)
            predictions = {}
            for name, model in self.models.items():
                pred = model.predict(sequence, verbose=0)
                pred_pct = pred[0][0]  # Already in percentage form (-1 to 1 scale)

                # Validate prediction range and handle excessive values
                if abs(pred_pct) > self.max_expected_return:
                    logger.warning(f"Model {name} prediction {pred_pct:.2%} exceeds normal range")
                    # Clip the prediction to the expected range rather than returning None
                    pred_pct = max(min(pred_pct, self.max_expected_return), -self.max_expected_return)
                    logger.info(f"Clipped prediction to {pred_pct:.2%}")

                predictions[name] = pred_pct

            # Calculate ensemble prediction
            pred_array = np.array(list(predictions.values()))
            ensemble_pred = np.mean(pred_array)

            # Calculate prediction confidence based on model agreement
            pred_std = np.std(pred_array)
            max_expected_std = self.max_expected_return * 0.1  # 10% of max return
            agreement_score = 1.0 - min(pred_std / max_expected_std, 1.0)

            # Calculate confidence based on prediction magnitude
            magnitude_score = min(1.0, 1.0 - (abs(ensemble_pred) / self.max_expected_return))

            # Combined confidence score
            confidence = 0.7 * agreement_score + 0.3 * magnitude_score

            # Return prediction only if confidence meets threshold
            if confidence >= confidence_threshold:
                result = {
                    'prediction': ensemble_pred,
                    'confidence': confidence,
                    'model_predictions': predictions
                }
                logger.info(f"Prediction made - Value: {ensemble_pred:.2%}, Confidence: {confidence:.2f}")
                return result
            else:
                logger.warning(f"Low confidence prediction ({confidence:.2f}) rejected")
                return None

        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            return None

    def get_prediction_metrics(self, sequence):
        """
        Get detailed prediction metrics from each model

        Args:
            sequence: Input sequence
        """
        try:
            metrics = {
                'individual_predictions': {},
                'prediction_spread': None,
                'confidence_score': None,
                'prediction_magnitude': None
            }

            # Handle case where direct model assignment was used
            if self._single_model is not None and not self.models:
                self.models['default'] = self._single_model

            # Get individual model predictions
            predictions = []
            for name, model in self.models.items():
                pred = model.predict(sequence, verbose=0)[0][0]

                # Clip predictions to expected range
                if abs(pred) > self.max_expected_return:
                    pred = max(min(pred, self.max_expected_return), -self.max_expected_return)

                metrics['individual_predictions'][name] = pred
                predictions.append(pred)

            # Calculate prediction spread and confidence
            pred_array = np.array(predictions)
            metrics['prediction_spread'] = np.max(pred_array) - np.min(pred_array)
            metrics['prediction_magnitude'] = abs(np.mean(pred_array))

            # Calculate confidence components
            pred_std = np.std(pred_array)
            max_expected_std = self.max_expected_return * 0.1
            agreement_score = 1.0 - min(pred_std / max_expected_std, 1.0)
            magnitude_score = min(1.0, 1.0 - (metrics['prediction_magnitude'] / self.max_expected_return))

            metrics['confidence_score'] = 0.7 * agreement_score + 0.3 * magnitude_score

            return metrics

        except Exception as e:
            logger.error(f"Error calculating prediction metrics: {str(e)}")
            return None