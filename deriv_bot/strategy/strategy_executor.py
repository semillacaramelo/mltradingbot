"""
Module for executing trading strategies based on ML predictions
"""
from deriv_bot.monitor.logger import setup_logger

logger = setup_logger(__name__)

class StrategyExecutor:
    def __init__(self, predictor, risk_manager, order_executor):
        self.predictor = predictor
        self.risk_manager = risk_manager
        self.order_executor = order_executor

        # Base strategy parameters
        self.base_params = {
            'min_prediction_threshold': 0.015,
            'position_hold_time': 30,
            'max_position_size': 0.02,
            'stop_loss_pct': 0.01
        }

        # Adjust parameters based on account type
        self._adjust_strategy_parameters()

    def _adjust_strategy_parameters(self):
        """Adjust strategy parameters based on account type"""
        if self.risk_manager.is_demo:
            # More aggressive parameters for demo account
            self.min_prediction_threshold = self.base_params['min_prediction_threshold'] * 0.7  # Lower threshold
            self.position_hold_time = self.base_params['position_hold_time'] * 0.5  # Faster trades
            self.max_position_size = self.base_params['max_position_size'] * 2  # Larger positions
            self.stop_loss_pct = self.base_params['stop_loss_pct'] * 5  # Wider stop loss
            logger.info("Strategy configured for DEMO account with aggressive parameters")
        else:
            # Conservative parameters for real account
            self.min_prediction_threshold = self.base_params['min_prediction_threshold']
            self.position_hold_time = self.base_params['position_hold_time']
            self.max_position_size = self.base_params['max_position_size']
            self.stop_loss_pct = self.base_params['stop_loss_pct']
            logger.info("Strategy configured for REAL account with conservative parameters")

    async def execute_strategy(self, market_data, symbol, stake_amount):
        """
        Execute trading strategy based on predictions

        Args:
            market_data: Processed market data sequence
            symbol: Trading symbol
            stake_amount: Base stake amount
        """
        try:
            # Get ensemble prediction with adjusted confidence threshold for demo
            confidence_threshold = 0.6 if self.risk_manager.is_demo else 0.7
            prediction_result = self.predictor.predict(market_data, confidence_threshold=confidence_threshold)

            if prediction_result is None:
                logger.warning("No prediction available or confidence too low")
                return None

            prediction = prediction_result['prediction']
            confidence = prediction_result['confidence']

            # Calculate prediction strength
            current_price = market_data[-1][-1]
            price_diff = prediction - current_price
            prediction_pct = abs(price_diff / current_price)

            # More permissive threshold for demo account
            effective_threshold = self.min_prediction_threshold
            if self.risk_manager.is_demo:
                effective_threshold *= 0.7  # 30% lower threshold for demo

            # Check if prediction meets minimum threshold
            if prediction_pct < effective_threshold:
                logger.info(f"Prediction strength {prediction_pct:.2%} below threshold")
                return None

            # Determine trade direction
            contract_type = 'CALL' if price_diff > 0 else 'PUT'

            # Adjust position size based on confidence and account type
            position_multiplier = 2.0 if self.risk_manager.is_demo else 1.0
            adjusted_stake = stake_amount * min(1.0, confidence) * position_multiplier
            adjusted_stake = min(adjusted_stake, stake_amount * self.max_position_size)

            # Risk validation with account-specific parameters
            if not self.risk_manager.validate_trade(symbol, adjusted_stake, prediction):
                logger.warning("Trade failed risk validation")
                return None

            # Execute order with account-specific stop loss
            order_params = {
                'symbol': symbol,
                'contract_type': contract_type,
                'amount': adjusted_stake,
                'duration': self.position_hold_time,
                'stop_loss_pct': self.stop_loss_pct
            }

            order_result = await self.order_executor.place_order(**order_params)

            if order_result:
                logger.info(f"Strategy executed: {contract_type} order placed with confidence {confidence:.2f}")
                return order_result
            else:
                logger.warning("Order execution failed")
                return None

        except Exception as e:
            logger.error(f"Error executing strategy: {str(e)}")
            return None

    def update_strategy_parameters(self, market_conditions):
        """
        Update strategy parameters based on market conditions

        Args:
            market_conditions: Dictionary containing market metrics
        """
        try:
            # Base parameter adjustments
            if 'volatility' in market_conditions:
                base_threshold = 0.01 if self.risk_manager.is_demo else 0.015
                self.min_prediction_threshold = max(
                    base_threshold,
                    min(0.03, market_conditions['volatility'] * 0.3)
                )

            if 'trend_strength' in market_conditions:
                base_hold_time = 15 if self.risk_manager.is_demo else 30
                self.position_hold_time = max(
                    base_hold_time,
                    min(60, int(30 * market_conditions['trend_strength']))
                )

            if 'market_regime' in market_conditions:
                regime_risk = market_conditions.get('regime_risk', 1.0)
                base_size = 0.04 if self.risk_manager.is_demo else 0.02
                self.max_position_size = max(0.01, min(0.06, base_size * (1 / regime_risk)))

            logger.info(f"Strategy parameters updated for {'DEMO' if self.risk_manager.is_demo else 'REAL'} account")

        except Exception as e:
            logger.error(f"Error updating strategy parameters: {str(e)}")