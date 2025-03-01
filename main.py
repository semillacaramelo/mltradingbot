"""
Main Entry Point for Deriv ML Trading Bot

Location: deriv_bot/main.py

Purpose:
Main orchestrator for the trading bot that coordinates all components including
data fetching, model training, trading execution and monitoring.

Dependencies:
- deriv_bot.data: API connection and data handling
- deriv_bot.strategy: ML models and prediction
- deriv_bot.risk: Risk management
- deriv_bot.execution: Trade execution
- deriv_bot.monitor: Logging and performance tracking
- deriv_bot.utils: Configuration and utilities

Interactions:
- Input: Command line arguments, API data streams
- Output: Trading actions, logs, performance metrics
- Relations: Coordinates all other modules

Author: Trading Bot Team
Last modified: 2024-02-26
"""
import asyncio
import os
import argparse
import sys
import signal
from datetime import datetime, timedelta
import pandas as pd  # Add explicit pandas import
import numpy as np   # Add numpy import for completeness
from deriv_bot.data.deriv_connector import DerivConnector
from deriv_bot.data.data_fetcher import DataFetcher
from deriv_bot.data.data_processor import DataProcessor
from deriv_bot.strategy.model_trainer import ModelTrainer
from deriv_bot.strategy.model_predictor import ModelPredictor
from deriv_bot.risk.risk_manager import RiskManager
from deriv_bot.execution.order_executor import OrderExecutor
from deriv_bot.monitor.logger import setup_logger
from deriv_bot.monitor.performance import PerformanceTracker
from deriv_bot.utils.config import Config
from deriv_bot.utils.model_manager import ModelManager
from deriv_bot.utils.asset_selector import AssetSelector
import time
from deriv_bot.strategy.model_evaluator import ModelEvaluator
import tensorflow as tf  # Ensure tensorflow is installed (e.g. pip install tensorflow)

logger = setup_logger(__name__)

# Global variable to handle graceful shutdown
shutdown_requested = False

def signal_handler(sig, frame):
    """Handle Ctrl+C and other termination signals"""
    global shutdown_requested
    logger.info("Shutdown signal received. Cleaning up...")
    shutdown_requested = True

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Deriv ML Trading Bot')
    parser.add_argument('--env', choices=['demo', 'real'], default=None,
                        help='Trading environment (demo or real). If not specified, uses DERIV_BOT_ENV from .env file')
    parser.add_argument('--train-only', action='store_true',
                        help='Only train the model without trading')
    parser.add_argument('--symbol', default='frxEURUSD',
                        help='Trading symbol')
    parser.add_argument('--clean-models', action='store_true',
                        help='Clean up old model files before starting')
    parser.add_argument('--stake-amount', type=float,
                        help='Stake amount for trades')
    parser.add_argument('--train-interval', type=int, default=4,
                        help='Hours between model retraining (default: 4)')
    parser.add_argument('--check-connection', action='store_true',
                        help='Only check API connection and exit')
    parser.add_argument('--data-source', choices=['api', 'file', 'both'], default='api',
                        help='Source for training data: api (from Deriv API), file (from saved data files), '
                             'or both (combine API and file data)')
    parser.add_argument('--save-data', action='store_true',
                        help='Save fetched historical data for future use')
    parser.add_argument('--model-types', nargs='+',
                        default=['short_term', 'medium_term', 'long_term'],
                        help='Types of models to train (space-separated list)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with more verbose output')

    # Training custom parameters    python -u test_api_connectivity.py 2>&1 | tee api_test.log
    parser.add_argument('--sequence-length', type=int, 
                       default=10,  # Match model's expected shape
                       help='Sequence length for input data')
    parser.add_argument('--epochs', type=int,
                        help='Number of training epochs')
    return parser.parse_args()

async def initialize_components(args, config):
    """Initialize and connect all components"""
    try:
        # Set trading environment from args, then from env var if not specified
        if args.env:
            env_mode = args.env
        else:
            env_mode = os.getenv('DERIV_BOT_ENV', 'demo').lower()

        if not config.set_environment(env_mode):
            logger.error(f"Failed to set environment to {env_mode}")
            return None

        # Update trading configuration if provided in args
        if args.symbol:
            config.trading_config['symbol'] = args.symbol
        if args.stake_amount:
            config.trading_config['stake_amount'] = args.stake_amount

        # Initialize components in correct order
        connector = DerivConnector(config)
        
        # Connect to Deriv API with retry
        max_retries = 5
        retry_delay = 10
        connected = False

        for attempt in range(max_retries):
            connected = await connector.connect()
            if connected:
                break
            logger.warning(f"Connection attempt {attempt + 1} failed, retrying in {retry_delay}s...")
            await asyncio.sleep(retry_delay)

        if not connected:
            raise Exception("Failed to connect to Deriv API after multiple attempts")

        # Initialize components with proper dependencies
        data_fetcher = DataFetcher(connector)
        data_processor = DataProcessor()
        asset_selector = AssetSelector(data_fetcher)

        # Always use demo risk profile when in demo mode
        is_demo = config.is_demo()
        risk_manager = RiskManager(is_demo=is_demo)
        risk_manager.connector = connector  # Set connector reference

        # Create order executor with risk manager
        order_executor = OrderExecutor(
            api_connector=connector,
            risk_manager=risk_manager  # Explicitly pass risk_manager
        )

        performance_tracker = PerformanceTracker()
        model_manager = ModelManager(max_models=int(os.getenv('MAX_MODELS_KEPT', '5')))
        
        # Initialize model trainer
        from deriv_bot.strategy.model_trainer import ModelTrainer
        model_trainer = ModelTrainer()

        # Clean up old model files if requested
        if args.clean_models:
            archived = model_manager.archive_old_models()
            logger.info(f"Archived {archived} old model files")
            deleted = model_manager.cleanup_archive(keep_days=30)
            logger.info(f"Deleted {deleted} expired archive files")

        # Warning for real mode
        if not is_demo:
            logger.warning("⚠️ RUNNING IN REAL TRADING MODE - ACTUAL FUNDS WILL BE USED! ⚠️")

        logger.info("All components initialized successfully")
        return {
            'config': config,
            'connector': connector,
            'data_fetcher': data_fetcher,
            'data_processor': data_processor,
            'risk_manager': risk_manager,
            'order_executor': order_executor,
            'performance_tracker': performance_tracker,
            'model_manager': model_manager,
            'model_trainer': model_trainer,
            'asset_selector': asset_selector
        }

    except Exception as e:
        logger.error(f"Error initializing components: {str(e)}")
        return None

async def maintain_connection(connector):
    """Maintain API connection"""
    reconnect_attempts = 0
    max_reconnect_attempts = 10
    base_delay = 30  # Initial delay in seconds
    max_delay = 300  # Maximum delay of 5 minutes

    while not shutdown_requested:
        try:
            # Check if we already have an active connection
            connection_active = await connector.check_connection()
            
            if not connection_active:
                logger.warning("Connection lost, attempting to reconnect...")
                
                if reconnect_attempts >= max_reconnect_attempts:
                    logger.error(f"Failed to reconnect after {max_reconnect_attempts} attempts. Exiting...")
                    return False

                # Calculate delay with exponential backoff
                delay = min(base_delay * (2 ** reconnect_attempts), max_delay)
                logger.info(f"Reconnection attempt {reconnect_attempts + 1}/{max_reconnect_attempts} in {delay}s")
                
                await asyncio.sleep(delay)
                
                # Ensure clean state before reconnecting
                await connector.close()
                await asyncio.sleep(1)  # Brief pause after closing
                
                connected = await connector.connect()
                if connected:
                    logger.info("Successfully reconnected")
                    # Only reset counter on successful connection AND valid ping
                    if await connector.check_connection():
                        reconnect_attempts = 0
                        await asyncio.sleep(5)  # Allow connection to stabilize
                        continue
                else:
                    reconnect_attempts += 1
            else:
                # Reset counter when connection is stable
                reconnect_attempts = 0
                await asyncio.sleep(15)  # Regular check interval

        except Exception as e:
            logger.error(f"Error in connection maintenance: {str(e)}")
            reconnect_attempts += 1
            await asyncio.sleep(5)

    logger.info("Connection maintenance loop terminated")
    return True

async def load_historical_data(data_fetcher, args, symbol, count=1000):
    """Load historical data from API and/or local files"""
    try:
        # First verify that we have a valid connection
        if not await data_fetcher.connector.check_connection():
            logger.warning("No active connection, attempting to reconnect...")
            if not await data_fetcher.connector.reconnect():
                logger.error("Failed to establish connection")
                return None

        # Verify symbol availability before fetching
        logger.info(f"Verifying availability for {symbol}...")
        if not await data_fetcher.check_trading_enabled(symbol):
            logger.error(f"Symbol {symbol} is not available for trading")
            return None

        data_source = args.data_source.lower()
        
        # Try file data first if requested
        file_data = None
        if data_source in ['file', 'both']:
            # ...existing file loading code...
            pass  # Added to avoid empty block error

        # Fetch from API with retry mechanism
        api_data = None
        if data_source in ['api', 'both']:
            for attempt in range(3):  # Try up to 3 times
                try:
                    logger.info(f"Fetching {count} historical data points for {symbol} (attempt {attempt + 1})")
                    api_data = await data_fetcher.fetch_historical_data(
                        symbol, 
                        interval=60, 
                        count=count,
                        retry_attempts=2
                    )
                    
                    if api_data is not None and not api_data.empty:
                        break
                        
                    await asyncio.sleep(2 * (attempt + 1))
                except Exception as e:
                    logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
                    if attempt < 2:  # Don't sleep on last attempt
                        await asyncio.sleep(2 * (attempt + 1))

            if api_data is None:
                logger.error("Failed to fetch data from API after all attempts")
                if data_source == 'both' and file_data is not None:
                    logger.info("Using only file data for training")
                    return file_data
                return None

        # ... rest of the existing function ...

    except Exception as e:
        logger.error(f"Error loading historical data: {str(e)}")
        return None

async def train_model(components, historical_data, model_type='standard', save_timestamp=True, args=None):
    """Train model with latest data"""
    try:
        # Initialize model trainer if not in components
        from deriv_bot.strategy.model_trainer import ModelTrainer
        from deriv_bot.strategy.model_evaluator import ModelEvaluator
        
        # Check if model_trainer exists in components, if not create it
        if 'model_trainer' not in components:
            # Create a new model trainer
            components['model_trainer'] = ModelTrainer()
            logger.info("Created new model trainer instance")
        
        model_trainer = components['model_trainer']
        # Initialize model evaluator
        model_evaluator = ModelEvaluator()
        
        # Validate data quality
        is_valid, message = model_evaluator.validate_training_data(historical_data)
        if not is_valid:
            logger.error(f"Data validation failed: {message}")
            return None

        # Get custom training parameters with better defaults
        sequence_length = args.sequence_length if args and hasattr(args, 'sequence_length') else 30
        epochs = args.epochs if args and hasattr(args, 'epochs') else 100  # Increased default epochs
        
        logger.info(f"Training {model_type} model with {len(historical_data)} data points")
        
        # Process data with compatible parameters (remove unsupported parameters)
        processed_data = components['data_processor'].prepare_data(
            df=historical_data,
            sequence_length=sequence_length
        )
        
        if processed_data is None:
            logger.error(f"Failed to process data for {model_type} model")
            return None

        # Unpack correctly based on the structure from data_processor.prepare_data
        if len(processed_data) == 3:
            X_train, y_train, scaler = processed_data
            X_val, y_val = None, None
        else:
            # Handle unpacking for newer versions that return validation split
            X_train, y_train, X_val, y_val, scaler = processed_data
        
        # Add early stopping and reduce learning rate on plateau
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=10,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=5
            )
        ]
        
        # Train with callbacks and appropriate parameters
        if X_val is not None:
            # If validation data is available
            history = model_trainer.train(
                X_train, y_train,
                validation_data=(X_val, y_val),
                model_type=model_type,
                epochs=epochs,
                callbacks=callbacks
            )
        else:
            # Otherwise train without validation data
            history = model_trainer.train(
                X_train, y_train,
                model_type=model_type,
                epochs=epochs,
                callbacks=callbacks
            )
        
        # Evaluate model
        if history:
            # Use validation data if available, otherwise use training data
            eval_X = X_val if X_val is not None else X_train
            eval_y = y_val if y_val is not None else y_train
            
            meets_threshold, metrics = model_evaluator.evaluate_model(
                model_trainer.model,
                eval_X, eval_y,
                history
            )
            
            logger.info(model_evaluator.get_metrics_summary())
            
            if not meets_threshold:
                logger.warning("Model did not meet quality threshold")
                return None
                
            # Save model if quality threshold met
            try:
                model_path = components['model_manager'].save_model_with_timestamp(
                    model_trainer.model,
                    base_name="trained_model",
                    model_type=model_type,
                    scaler=scaler,
                    metrics=metrics  # Save metrics with model
                )
                
                if model_path:
                    logger.info(f"Saved {model_type} model to {model_path}")
                    return ModelPredictor(model_path)
                    
            except Exception as e:
                logger.error(f"Error saving model: {str(e)}")
                # Try emergency save
                emergency_path = save_emergency_model(model_trainer.model, model_type)
                if emergency_path:
                    return ModelPredictor(emergency_path)
                    
        return None

    except Exception as e:
        logger.error(f"Error in model training: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

def save_emergency_model(model, model_type):
    """Emergency save of model"""
    try:
        emergency_path = f'models/emergency_{model_type}_{int(time.time())}.keras'
        model.save(emergency_path)
        logger.warning(f"Emergency save successful: {emergency_path}")
        return emergency_path
    except Exception as e:
        logger.error(f"Emergency save failed: {str(e)}")
        return None

async def execute_trade(components, predictor, symbol, sequence):
    """Execute a trade based on model prediction"""
    try:
        # Check if asset is tradeable first
        if not components['asset_selector'].is_asset_tradeable(symbol):
            logger.warning(f"Trading for {symbol} is not available at this time. Waiting...")
            return None

        prediction_result = predictor.predict(sequence)

        if prediction_result is not None:
            prediction = prediction_result['prediction']
            confidence = prediction_result['confidence']

            logger.info(f"Prediction: {prediction:.2%} (confidence: {confidence:.2f})")

            # Execute trade if prediction is significant
            amount = components['config'].trading_config['stake_amount']
            if abs(prediction) >= 0.001:  # 0.1% minimum move
                if components['risk_manager'].validate_trade(symbol, amount, prediction, connector=components['connector']):
                    contract_type = 'CALL' if prediction > 0 else 'PUT'
                    duration = components['config'].trading_config['duration']

                    result = await components['order_executor'].place_order(
                        symbol,
                        contract_type,
                        amount,
                        duration
                    )

                    if result:
                        logger.info(f"Trade executed: {contract_type} {amount}")
                        return True
            else:
                logger.info(f"No trade: predicted move ({prediction:.2%}) below threshold")

        return False

    except Exception as e:
        logger.error(f"Error executing trade: {str(e)}")
        return False

async def check_api_connectivity():
    """Simple function to check API connectivity and configuration"""
    config = Config()

    try:
        # Get environment
        env_mode = os.getenv('DERIV_BOT_ENV', 'demo').lower()
        if not config.set_environment(env_mode):
            logger.error(f"Failed to set environment to {env_mode}")
            return False

        # Check if tokens exist
        demo_token = os.getenv('DERIV_API_TOKEN_DEMO')
        real_token = os.getenv('DERIV_API_TOKEN_REAL')
        real_confirmed = os.getenv('DERIV_REAL_MODE_CONFIRMED', 'no').lower() == 'yes'

        print(f"\nChecking environment configuration:")
        print(f"- Current mode: {env_mode.upper()}")
        print(f"- Demo token: {'Configured' if demo_token else 'Not configured'}")
        print(f"- Real token: {'Configured' if real_token else 'Not configured'}")
        print(f"- Real mode confirmed: {'Yes' if real_confirmed else 'No'}")

        # Connect and check API
        connector = DerivConnector(config)
        print(f"\nAttempting to connect to Deriv API ({env_mode.upper()} mode)...")
        connected = await connector.connect()

        if connected:
            print("✅ Successfully connected to Deriv API")

            # Get active symbols
            data_fetcher = DataFetcher(connector)
            symbols = await data_fetcher.get_available_symbols()

            if symbols:
                print(f"✅ Successfully fetched {len(symbols)} trading symbols")
                print(f"Sample symbols: {symbols[:5]}")
            else:
                print("❌ Failed to fetch trading symbols")

            # Close connection
            await connector.close()
            print("✅ Connection closed properly")
            return True
        else:
            print("❌ Failed to connect to Deriv API")
            return False

    except Exception as e:
        print(f"❌ Error during API connectivity check: {str(e)}")
        return False

async def main():
    # Parse command line arguments
    args = parse_arguments()

    # Set debug mode if requested
    if args.debug:
        import logging
        logging.getLogger('deriv_bot').setLevel(logging.DEBUG)
        logger.info("Debug mode enabled")

    # Just check API connectivity if requested
    if args.check_connection:
        success = await check_api_connectivity()
        if not success:
            print("\n❌ API connectivity check failed. Please check your configuration.")
        sys.exit(0 if success else 1)

    # Initialize configuration
    config = Config()
    components = None
    last_training = None
    training_interval = timedelta(hours=args.train_interval)  # Use specified training interval
    execution_start = datetime.now()
    reconnection_task = None
    predictors = {}

    try:
        # Initialize components
        components = await initialize_components(args, config)
        if not components:
            logger.error("Failed to initialize components")
            return

        symbol = components['config'].trading_config['symbol']

        # Training-only mode
        if args.train_only:
            logger.info("=== Starting Model Training Only Mode ===")
            logger.info(f"Symbol: {symbol}")
            logger.info(f"Model types to train: {args.model_types}")

            # Fetch historical data for training
            logger.info("Loading historical data for training...")
            historical_data = await load_historical_data(
                components['data_fetcher'], args, symbol, count=1000
            )

            if historical_data is not None:
                logger.info(f"Successfully loaded {len(historical_data)} data points")

                # Train each model type
                for model_type in args.model_types:
                    logger.info(f"Training {model_type} model...")
                    # Train the model with save_timestamp=True instead of providing model_path
                    predictor = await train_model(
                        components,
                        historical_data,
                        model_type=model_type,
                        save_timestamp=True,  # Fix: use save_timestamp instead of model_path
                        args=args
                    )

                    if predictor:
                        logger.info(f"{model_type} model training successful!")
                        predictors[model_type] = predictor
                    else:
                        logger.error(f"{model_type} model training failed")

                # Report results
                success_count = len(predictors)
                if success_count > 0:
                    logger.info(f"Successfully trained {success_count}/{len(args.model_types)} models")
                else:
                    logger.error("All model training failed")
            else:
                logger.error("Failed to load training data")

            logger.info("Training only mode completed - exiting")
            return

        # Regular trading mode
        env_mode = "REAL" if not config.is_demo() else "DEMO"
        logger.info("=== Starting Deriv ML Trading Bot ===")
        logger.info(f"Start time: {execution_start}")
        logger.info(f"Mode: {env_mode}")
        logger.info(f"Symbol: {symbol}")
        logger.info("Trading Parameters:")
        logger.info(f"- Stake Amount: {components['config'].trading_config['stake_amount']}")
        logger.info(f"- Duration: {components['config'].trading_config['duration']}s")
        logger.info(f"- Training Interval: {training_interval.total_seconds()/3600:.1f}h")
        logger.info(f"- Model Types: {args.model_types}")
        logger.info("====================================")

        # Start connection maintenance task
        reconnection_task = asyncio.create_task(maintain_connection(components['connector']))

        consecutive_errors = 0
        max_consecutive_errors = 5

        # Initial training if no models exist or they're too old
        for model_type in args.model_types:
            model_path = components['model_manager'].get_best_model_path(model_type=model_type)
            if model_path and os.path.exists(model_path):
                # Use existing model
                model_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(model_path))
                if model_age > timedelta(days=1):
                    logger.info(f"Existing {model_type} model is {model_age.days} days old. Retraining...")
                    predictors[model_type] = None
                else:
                    logger.info(f"Loading existing {model_type} model from {model_path}")
                    predictors[model_type] = ModelPredictor(model_path)
            else:
                logger.info(f"No existing {model_type} model found. Training new model...")
                predictors[model_type] = None

        while not shutdown_requested:
            try:
                # Check if retraining is needed
                current_time = datetime.now()
                execution_time = current_time - execution_start
                logger.info(f"Bot running for: {execution_time}")

                needs_training = (
                    last_training is None or
                    (current_time - last_training) > training_interval or
                    any(predictor is None for predictor in predictors.values())
                )

                if needs_training:
                    logger.info("Starting model retraining cycle...")
                    historical_data = await load_historical_data(
                        components['data_fetcher'],
                        args,
                        symbol,
                        count=1000
                    )

                    if historical_data is not None:
                        # Train all required model types
                        training_success = True
                        for model_type in args.model_types:
                            predictor = await train_model(
                                components,
                                historical_data,
                                model_type=model_type,
                                save_timestamp=True,
                                args=args
                            )

                            if predictor:
                                predictors[model_type] = predictor
                                logger.info(f"{model_type} model successfully trained")
                            else:
                                training_success = False
                                logger.error(f"{model_type} model training failed")

                        if training_success:
                            last_training = current_time
                            logger.info("All models successfully retrained")
                            logger.info(f"Next training scheduled for: {current_time + training_interval}")
                            consecutive_errors = 0  # Reset error counter on successful training
                        else:
                            logger.warning("Some models failed to train")
                            consecutive_errors += 1
                    else:
                        logger.error("Failed to fetch training data")
                        consecutive_errors += 1
                        await asyncio.sleep(60)
                        continue

                # Check if we've had too many consecutive errors
                if consecutive_errors >= max_consecutive_errors:
                    logger.error(f"Too many consecutive errors ({consecutive_errors}). Restarting bot...")
                    # Force reconnection
                    await components['connector'].close()
                    await asyncio.sleep(30)
                    connected = await components['connector'].connect()
                    if connected:
                        logger.info("Successfully reconnected after multiple errors")
                        consecutive_errors = 0
                    else:
                        logger.error("Failed to reconnect after multiple errors. Exiting...")
                        break

                # Check if trading is available for the symbol
                trading_enabled = await components['data_fetcher'].check_trading_enabled(symbol)
                if not trading_enabled:
                    logger.warning(f"Trading for {symbol} is not available at this time. Waiting 5 minutes...")
                    await asyncio.sleep(300)  # Wait 5 minutes before retrying
                    consecutive_errors = 0  # Reset errors since this is an expected condition
                    continue

                # Get latest market data
                latest_data = await components['data_fetcher'].fetch_historical_data(
                    symbol,
                    interval=60,
                    count=60
                )

                if latest_data is None:
                    logger.warning("Failed to fetch latest data, retrying...")
                    consecutive_errors += 1
                    await asyncio.sleep(60)
                    continue

                # Prepare data for prediction
                processed_sequence = components['data_processor'].prepare_data(latest_data)
                if processed_sequence is None:
                    logger.warning("Failed to process latest data, retrying...")
                    consecutive_errors += 1
                    await asyncio.sleep(60)
                    continue

                X_latest, _, _ = processed_sequence
                if X_latest is None or len(X_latest) == 0:
                    logger.warning("Invalid sequence data, retrying...")
                    consecutive_errors += 1
                    await asyncio.sleep(60)
                    continue

                # Get the last sequence for prediction
                sequence = X_latest[-1:]

                # Execute trade based on ensemble prediction from all model types
                # Use the first available model for now (in future, could implement ensemble voting)
                for model_type, predictor in predictors.items():
                    if predictor:
                        logger.info(f"Using {model_type} model for prediction")
                        trade_executed = await execute_trade(components, predictor, symbol, sequence)
                        if trade_executed:
                            consecutive_errors = 0  # Reset error counter on successful trade
                        break
                else:
                    logger.warning("No valid predictors available for trading")

                # Log performance metrics periodically
                if execution_time.total_seconds() % 3600 < 60:  # Every hour
                    metrics = components['performance_tracker'].get_statistics()
                    logger.info("\n=== Hourly Performance Update ===")
                    logger.info(f"Total Runtime: {execution_time}")
                    logger.info(f"Performance Metrics: {metrics}")
                    logger.info("=================================")

                # Wait before next iteration
                await asyncio.sleep(60)

            except Exception as e:
                logger.error(f"Error in trading loop: {str(e)}")
                consecutive_errors += 1
                await asyncio.sleep(60)

    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
    finally:
        if reconnection_task:
            reconnection_task.cancel()

        if components and components['connector']:
            await components['connector'].close()
            logger.info("Connection closed.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBot shutdown requested. Exiting...")
    except Exception as e:
        print(f"Unhandled exception: {str(e)}")
        sys.exit(1)