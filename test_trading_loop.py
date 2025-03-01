"""
Test script for simulating the trading loop without executing real trades
"""
import asyncio
import pandas as pd
import os
from datetime import datetime
from deriv_bot.data.deriv_connector import DerivConnector
from deriv_bot.data.data_fetcher import DataFetcher
from deriv_bot.data.data_processor import DataProcessor
from deriv_bot.strategy.model_trainer import ModelTrainer
from deriv_bot.strategy.model_predictor import ModelPredictor
from deriv_bot.strategy.feature_engineering import FeatureEngineer
from deriv_bot.risk.risk_manager import RiskManager
from deriv_bot.monitor.performance import PerformanceTracker
from deriv_bot.monitor.logger import setup_logger
from deriv_bot.utils.model_manager import ModelManager

logger = setup_logger('trading_simulation')

class MockOrderExecutor:
    """Mock order executor for simulation"""
    async def place_order(self, symbol, contract_type, amount, duration, stop_loss_pct=None):
        """Simulate order placement with stop loss"""
        logger.info(f"SIMULATION: Would place {contract_type} order for {symbol}, "
                   f"amount: {amount}, stop_loss: {stop_loss_pct}%")
        return {
            'contract_id': 'mock_id_' + datetime.now().strftime('%H%M%S'),
            'transaction_id': 'mock_tx_' + datetime.now().strftime('%H%M%S'),
            'entry_tick': 0,
            'entry_tick_time': datetime.now().timestamp()
        }

async def run_trading_simulation():
    """Run trading simulation with real data but mock order execution"""
    connector = None
    try:
        # Initialize components with DEMO profile
        connector = DerivConnector()
        data_fetcher = DataFetcher(connector)
        data_processor = DataProcessor()
        feature_engineer = FeatureEngineer()
        risk_manager = RiskManager(is_demo=True, max_position_size=200, max_daily_loss=150)
        mock_executor = MockOrderExecutor()
        performance_tracker = PerformanceTracker()
        model_manager = ModelManager()  # Initialize model manager for proper model handling

        # Log risk profile
        risk_profile = risk_manager.get_risk_profile()
        logger.info("=== DEMO Trading Simulation Configuration ===")
        logger.info(f"Risk Profile: {risk_profile}")
        logger.info("Trading Parameters:")
        logger.info("- Confidence Threshold: 0.6 (DEMO)")
        logger.info("- Position Size: 20.0 (Higher for DEMO)")
        logger.info("- Trade Duration: 30s (Faster for DEMO)")
        logger.info("- Stop Loss: 5.0% (Wider for DEMO)")
        logger.info("==========================================")

        # Connect to API
        connected = await connector.connect()
        if not connected:
            logger.error("Failed to connect to Deriv API")
            return

        logger.info("Connected to Deriv API")

        # Test with EUR/USD
        symbol = "frxEURUSD"

        # Fetch historical data
        logger.info(f"Fetching historical data for {symbol}")
        historical_data = await data_fetcher.fetch_historical_data(
            symbol,
            interval=60,
            count=500
        )

        if historical_data is None:
            logger.error("Failed to fetch historical data")
            return

        logger.info(f"Successfully fetched {len(historical_data)} candles")

        # Add enhanced features
        historical_data = feature_engineer.calculate_features(historical_data)
        if historical_data is None:
            logger.error("Failed to calculate features")
            return

        logger.info(f"Calculated features. Shape: {historical_data.shape}")
        logger.info(f"Features: {historical_data.columns.tolist()}")

        # Process data with shorter sequence length for demo
        sequence_length = 30
        logger.info(f"Processing data with sequence length: {sequence_length}")

        processed_data = data_processor.prepare_data(
            df=historical_data,
            sequence_length=sequence_length
        )

        if processed_data is None:
            logger.error("Failed to process historical data")
            return

        X, y, scaler = processed_data
        if X is None or y is None:
            logger.error("Invalid processed data")
            return

        logger.info(f"Processed data shapes - X: {X.shape}, y: {y.shape}")

        # Train ensemble model
        logger.info("Training ensemble models")
        model_trainer = ModelTrainer(input_shape=(X.shape[1], X.shape[2]))

        # Set model type for this test
        model_type = "short_term"
        logger.info(f"Training {model_type} model for simulation")

        history = model_trainer.train(X, y, epochs=10, model_type=model_type)  # Quick training for testing

        if not history:
            logger.error("Model training failed")
            return

        logger.info("Model training completed")

        # Create models directory if it doesn't exist
        models_dir = 'models'
        os.makedirs(models_dir, exist_ok=True)

        # Save model with native Keras format and model type
        model_path = os.path.join(models_dir, f'{model_type}_model.keras')
        saved = model_trainer.save_model(model_path, scaler=scaler)

        if not saved:
            logger.error(f"Failed to save model to {model_path}")
            return

        logger.info(f"Model successfully saved to {model_path}")

        # Save a timestamped model version using model manager
        timestamp_path = model_manager.save_model_with_timestamp(
            model_trainer.model, 
            base_name="trained_model", 
            model_type=model_type,
            scaler=scaler
        )

        if timestamp_path:
            logger.info(f"Timestamped model saved to {timestamp_path}")
        else:
            logger.warning("Failed to save timestamped model")

        # Initialize predictor with trained model
        predictor = ModelPredictor(model_path)

        if not predictor.models:
            logger.error("Failed to load models for prediction")
            return

        logger.info(f"Successfully loaded {len(predictor.models)} model(s) for prediction")

        # Run simulation loop with DEMO confidence threshold
        logger.info("Starting trading simulation loop")
        iteration = 0
        trades_executed = 0
        successful_trades = 0
        confidence_threshold = 0.6  # Lower threshold for DEMO

        while iteration < 10:  # Run 10 iterations for testing
            try:
                logger.info(f"\n=== Iteration {iteration + 1}/10 ===")
                logger.info(f"Current Stats - Trades: {trades_executed}, Successful: {successful_trades}")

                # Get latest data
                latest_data = await data_fetcher.fetch_historical_data(
                    symbol,
                    interval=60,
                    count=200
                )

                if latest_data is None:
                    logger.warning("Failed to fetch latest data, retrying...")
                    await asyncio.sleep(5)
                    continue

                # Calculate features
                latest_data = feature_engineer.calculate_features(latest_data)
                if latest_data is None:
                    logger.warning("Failed to calculate features, retrying...")
                    await asyncio.sleep(5)
                    continue

                # Process data
                processed_sequence = data_processor.prepare_data(
                    df=latest_data,
                    sequence_length=sequence_length
                )

                if processed_sequence is None:
                    logger.warning("Failed to process latest data, retrying...")
                    await asyncio.sleep(5)
                    continue

                X_latest, _, _ = processed_sequence
                if X_latest is None or len(X_latest) == 0:
                    logger.warning("Invalid sequence data, retrying...")
                    await asyncio.sleep(5)
                    continue

                # Get the last sequence for prediction
                sequence = X_latest[-1:]
                prediction_result = predictor.predict(sequence, confidence_threshold)

                if prediction_result is not None:
                    prediction = prediction_result['prediction']
                    confidence = prediction_result['confidence']
                    logger.info(f"Prediction value: {prediction:.4%} (confidence: {confidence:.2f})")

                    current_price = latest_data['close'].iloc[-1]
                    predicted_return = prediction  # Already in percentage form
                    logger.info(f"Current price: {current_price:.5f}")
                    logger.info(f"Predicted return: {predicted_return:.2%}")

                    # Get prediction metrics
                    metrics = predictor.get_prediction_metrics(sequence)
                    logger.info(f"Prediction metrics: {metrics}")

                    # Simulate trade execution if prediction is significant
                    amount = 20.0  # Higher amount for DEMO
                    if abs(predicted_return) >= 0.0005:  # 0.05% minimum move for Forex
                        if risk_manager.validate_trade(symbol, amount, predicted_return):
                            contract_type = 'CALL' if predicted_return > 0 else 'PUT'
                            logger.info(f"Placing {contract_type} order, predicted return: {predicted_return:.2%}")

                            result = await mock_executor.place_order(
                                symbol,
                                contract_type,
                                amount,
                                30,  # Shorter duration for DEMO
                                stop_loss_pct=5.0  # Wider stop loss for DEMO
                            )

                            if result:
                                # Wait for contract duration
                                await asyncio.sleep(60)

                                # Fetch latest price
                                next_data = await data_fetcher.fetch_historical_data(
                                    symbol,
                                    interval=60,
                                    count=1
                                )

                                if next_data is not None:
                                    trades_executed += 1
                                    next_price = next_data['close'].iloc[-1]
                                    actual_return = (next_price - current_price) / current_price

                                    # Determine if trade was successful
                                    if (contract_type == 'CALL' and actual_return > 0) or \
                                       (contract_type == 'PUT' and actual_return < 0):
                                        successful_trades += 1
                                        logger.info(f"Successful trade! Price moved {actual_return:.2%}")
                                    else:
                                        logger.info(f"Unsuccessful trade. Price moved {actual_return:.2%}")

                                    # Record trade for performance tracking
                                    trade_data = {
                                        'symbol': symbol,
                                        'type': contract_type,
                                        'amount': amount,
                                        'entry_price': current_price,
                                        'exit_price': next_price,
                                        'predicted_change': predicted_return,
                                        'actual_change': actual_return,
                                        'confidence': confidence,
                                        'timestamp': datetime.now().isoformat()
                                    }
                                    performance_tracker.add_trade(trade_data)

                                    # Log detailed trade performance
                                    logger.info("\n=== Trade Performance Update ===")
                                    logger.info(f"Win Rate: {(successful_trades/trades_executed)*100:.1f}%")
                                    logger.info(f"Prediction Accuracy: {abs(predicted_return - actual_return):.2%}")
                                    stats = performance_tracker.get_statistics()
                                    if stats:
                                        logger.info(f"Performance Stats: {stats}")
                                else:
                                    logger.error("Failed to fetch next price after trade")

                    else:
                        logger.info(f"No trade: predicted move ({predicted_return:.2%}) below threshold")
                else:
                    logger.info("No trade: prediction confidence below threshold")

                iteration += 1
                logger.info(f"Completed simulation iteration {iteration}/10")

                # Wait before next iteration
                await asyncio.sleep(5)  # Shorter wait for testing

            except Exception as e:
                logger.error(f"Error in simulation loop: {str(e)}")
                await asyncio.sleep(5)

        # Final performance report
        logger.info("\n=== Final Simulation Report ===")
        logger.info(f"Total Trades Executed: {trades_executed}")
        if trades_executed > 0:
            win_rate = (successful_trades / trades_executed) * 100
            logger.info(f"Final Win Rate: {win_rate:.1f}%")

            # Get detailed statistics
            stats = performance_tracker.get_statistics()
            if stats:
                logger.info("\nDetailed Performance Metrics:")
                for key, value in stats.items():
                    logger.info(f"{key}: {value}")

            # Export results
            performance_tracker.export_history('simulation_results.csv')
            logger.info("\nSimulation results exported to simulation_results.csv")
        else:
            logger.warning("No trades executed during simulation")

        # Archive old models
        archived_count = model_manager.archive_old_models(model_type=model_type)
        logger.info(f"Archived {archived_count} old models")

        logger.info("\nTrading simulation completed")

    except Exception as e:
        logger.error(f"Fatal error in simulation: {str(e)}")
    finally:
        if connector:
            await connector.close()

if __name__ == "__main__":
    print("Starting trading simulation...")
    print("=" * 30)
    asyncio.run(run_trading_simulation())