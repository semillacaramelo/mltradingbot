"""
Utility module for managing model files
Handles archiving, cleanup and versioning of training files
"""
"""
Model Management Module

Location: deriv_bot/utils/model_manager.py

Purpose:
Manages model files including saving, loading, archiving,
and cleanup of old model versions.

Dependencies:
- os: File operations
- datetime: Timestamp handling
- deriv_bot.monitor.logger: Logging functionality

Interactions:
- Input: Model files and management commands
- Output: Model file operations and maintenance
- Relations: Used by model trainer and predictor

Author: Trading Bot Team
Last modified: 2024-02-26
"""
import os
import glob
import shutil
import datetime
import logging
import pickle
from pathlib import Path
from deriv_bot.monitor.logger import setup_logger

logger = setup_logger(__name__)

class ModelManager:
    def __init__(self, models_dir="models", archive_dir="model_archive", max_models=5):
        """
        Initialize model manager

        Args:
            models_dir: Directory where active models are stored
            archive_dir: Directory where old models will be archived
            max_models: Maximum number of model versions to keep in models_dir
        """
        self.models_dir = models_dir
        self.archive_dir = archive_dir
        self.max_models = max_models

        # Create directories if they don't exist
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.archive_dir, exist_ok=True)

        logger.info(f"ModelManager initialized with max_models={max_models}")

    def archive_old_models(self, model_type=None):
        """
        Archive old model files to save disk space
        Keeps the most recent max_models and moves older ones to archive

        Args:
            model_type: Optional model type filter (e.g., 'short_term', 'long_term')
                        If None, archives all model types
        """
        try:
            logger.info(f"Archiving old models of type: {model_type or 'all'}")

            # Get all model files sorted by modification time (newest first)
            model_files = []
            # Support both legacy .h5 and new .keras formats
            for ext in ['*.h5', '*.keras', '*.pb', '*.savedmodel']:
                if model_type:
                    model_files.extend(glob.glob(os.path.join(self.models_dir, f"*{model_type}*{ext}")))
                else:
                    model_files.extend(glob.glob(os.path.join(self.models_dir, ext)))

            # Sort by modification time (newest first)
            model_files.sort(key=os.path.getmtime, reverse=True)

            if not model_files:
                logger.info(f"No model files found to archive for type: {model_type or 'all'}")
                return 0

            # Keep the newest max_models
            models_to_keep = model_files[:self.max_models]
            models_to_archive = model_files[self.max_models:]

            logger.info(f"Keeping {len(models_to_keep)} recent models, archiving {len(models_to_archive)} old models")

            # Archive older models with timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            archived_count = 0

            for model_path in models_to_archive:
                model_filename = os.path.basename(model_path)
                archive_path = os.path.join(self.archive_dir, f"{timestamp}_{model_filename}")

                try:
                    shutil.move(model_path, archive_path)
                    archived_count += 1
                    logger.debug(f"Archived model: {model_filename} → {archive_path}")

                    # Also move metadata file if it exists
                    if model_path.endswith('.h5'):
                        metadata_path = model_path.replace('.h5', '_metadata.pkl')
                    elif model_path.endswith('.keras'):
                        metadata_path = model_path.replace('.keras', '_metadata.pkl')
                    else:
                        metadata_path = f"{model_path}_metadata.pkl"

                    if os.path.exists(metadata_path):
                        metadata_filename = os.path.basename(metadata_path)
                        archive_metadata_path = os.path.join(self.archive_dir, f"{timestamp}_{metadata_filename}")
                        shutil.move(metadata_path, archive_metadata_path)
                        logger.debug(f"Archived metadata: {metadata_filename} → {archive_metadata_path}")

                except Exception as e:
                    logger.error(f"Error archiving model {model_filename}: {str(e)}")

            logger.info(f"Successfully archived {archived_count} model files")
            return archived_count

        except Exception as e:
            logger.error(f"Error in archive_old_models: {str(e)}")
            return 0

    def cleanup_archive(self, keep_days=30, dry_run=False):
        """
        Remove archived models older than specified days

        Args:
            keep_days: Number of days to keep archived models
            dry_run: If True, only list files that would be deleted without actually deleting

        Returns:
            Number of files deleted or that would be deleted in dry_run mode
        """
        try:
            cutoff_time = datetime.datetime.now() - datetime.timedelta(days=keep_days)
            cutoff_timestamp = cutoff_time.timestamp()

            archived_files = []
            # Support both legacy .h5 and new .keras formats
            for ext in ['*.h5', '*.keras', '*.pb', '*.savedmodel', '*_metadata.pkl']:
                archived_files.extend(glob.glob(os.path.join(self.archive_dir, ext)))
                archived_files.extend(glob.glob(os.path.join(self.archive_dir, f"*_{ext}")))

            to_delete = [f for f in archived_files if os.path.getmtime(f) < cutoff_timestamp]

            if dry_run:
                logger.info(f"Dry run: Would delete {len(to_delete)} archived models older than {keep_days} days")
                for file_path in to_delete[:10]:  # Show first 10 as examples
                    file_age = (datetime.datetime.now() - 
                               datetime.datetime.fromtimestamp(os.path.getmtime(file_path))).days
                    file_size = os.path.getsize(file_path) / (1024 * 1024)  # size in MB
                    logger.info(f"Would delete: {os.path.basename(file_path)} (Age: {file_age} days, Size: {file_size:.2f}MB)")
                return len(to_delete)

            deleted_count = 0
            total_size_freed = 0

            for file_path in to_delete:
                try:
                    file_size = os.path.getsize(file_path)
                    os.remove(file_path)
                    deleted_count += 1
                    total_size_freed += file_size
                    logger.debug(f"Deleted old archived model: {os.path.basename(file_path)}")
                except Exception as e:
                    logger.error(f"Error deleting {file_path}: {str(e)}")

            total_size_mb = total_size_freed / (1024 * 1024)  # Convert to MB
            logger.info(f"Cleaned up {deleted_count} archived models older than {keep_days} days")
            logger.info(f"Freed approximately {total_size_mb:.2f} MB of storage")
            return deleted_count

        except Exception as e:
            logger.error(f"Error in cleanup_archive: {str(e)}")
            return 0

    def get_best_model_path(self, model_prefix="best_model", model_type=None):
        """
        Get path to the best model file

        Args:
            model_prefix: Prefix of the best model file
            model_type: Optional model type filter (e.g., 'short_term', 'long_term')

        Returns:
            Path to the best model file or None if not found
        """
        try:
            logger.debug(f"Looking for best model with prefix '{model_prefix}' and type '{model_type or 'any'}'")

            # Check for models in both .keras (preferred) and .h5 (legacy) formats
            best_model = None
            newest_time = 0

            # Try searching for .keras files first
            for ext in ['.keras', '.h5']:
                search_pattern = f"{model_prefix}*{('_' + model_type) if model_type else ''}*{ext}"
                pattern = os.path.join(self.models_dir, search_pattern)
                matches = glob.glob(pattern)

                if matches:
                    # Find the newest file among the matches
                    for match in matches:
                        mod_time = os.path.getmtime(match)
                        if mod_time > newest_time:
                            newest_time = mod_time
                            best_model = match

            if not best_model:
                logger.warning(f"No model file found matching prefix {model_prefix} and type {model_type or 'any'}")
                return None

            logger.debug(f"Found best model: {best_model}")
            return best_model

        except Exception as e:
            logger.error(f"Error in get_best_model_path: {str(e)}")
            return None

    def save_model_with_timestamp(self, model, base_name="trained_model", model_type=None, scaler=None):
        """
        Save model with timestamp to prevent overwriting

        Args:
            model: TensorFlow model to save
            base_name: Base name for the model file
            model_type: Optional type identifier (e.g., 'short_term', 'long_term')
            scaler: Optional scaler to save with model for later denormalization

        Returns:
            Path to the saved model file or None if failed
        """
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            type_suffix = f"_{model_type}" if model_type else ""
            # Use .keras extension instead of .h5 for new models
            filename = f"{base_name}{type_suffix}_{timestamp}.keras"
            save_path = os.path.join(self.models_dir, filename)

            # Save the model in native Keras format without any additional parameters
            model.save(save_path)
            logger.info(f"Model saved to {save_path}")

            # Initialize metadata dictionary
            metadata = {}

            # Save scaler as metadata if provided
            if scaler is not None:
                metadata_path = save_path.replace('.keras', '_metadata.pkl')
                metadata = {'scaler': scaler}
                with open(metadata_path, 'wb') as f:
                    pickle.dump(metadata, f)
                logger.info(f"Model metadata with scaler saved to {metadata_path}")

            # Save a best model copy if specified as base_name
            if base_name == "best_model" and model_type:
                best_model_path = os.path.join(self.models_dir, f"best_model_{model_type}.keras")
                try:
                    # Save the best model without any additional parameters
                    model.save(best_model_path)
                    logger.info(f"Best model saved to {best_model_path}")

                    # Save scaler as metadata for best model as well
                    if scaler is not None:
                        best_metadata_path = best_model_path.replace('.keras', '_metadata.pkl')
                        with open(best_metadata_path, 'wb') as f:
                            pickle.dump(metadata, f)
                        logger.info(f"Best model metadata saved to {best_metadata_path}")
                except Exception as e:
                    logger.error(f"Error saving best model: {str(e)}")

            # Archive old models if we now have too many
            if model_type:
                # Count both .keras and .h5 files for backward compatibility
                count = len(glob.glob(os.path.join(self.models_dir, f"*{model_type}*.keras")))
                count += len(glob.glob(os.path.join(self.models_dir, f"*{model_type}*.h5")))
            else:
                count = len(glob.glob(os.path.join(self.models_dir, "*.keras")))
                count += len(glob.glob(os.path.join(self.models_dir, "*.h5")))

            if count > self.max_models:
                self.archive_old_models(model_type=model_type)

            return save_path

        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return None

    def get_model_size_stats(self):
        """
        Get size statistics for model directories

        Returns:
            Dictionary with size statistics
        """
        try:
            models_size = self._get_directory_size(self.models_dir)
            archive_size = self._get_directory_size(self.archive_dir)

            # Count both .keras and .h5 files
            active_keras_count = len(glob.glob(os.path.join(self.models_dir, "*.keras")))
            active_h5_count = len(glob.glob(os.path.join(self.models_dir, "*.h5")))
            archived_keras_count = len(glob.glob(os.path.join(self.archive_dir, "*.keras")))
            archived_h5_count = len(glob.glob(os.path.join(self.archive_dir, "*.h5")))

            return {
                'active_models_count': active_keras_count + active_h5_count,
                'active_models_size_mb': models_size / (1024 * 1024),
                'archived_models_count': archived_keras_count + archived_h5_count,
                'archived_models_size_mb': archive_size / (1024 * 1024),
                'total_size_mb': (models_size + archive_size) / (1024 * 1024)
            }
        except Exception as e:
            logger.error(f"Error getting model size stats: {str(e)}")
            return {}

    def _get_directory_size(self, directory):
        """Get total size of a directory in bytes"""
        total_size = 0
        if not os.path.exists(directory):
            return 0

        for dirpath, _, filenames in os.walk(directory):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                total_size += os.path.getsize(file_path)

        return total_size