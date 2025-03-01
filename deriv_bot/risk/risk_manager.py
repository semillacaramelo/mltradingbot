"""
Risk Management Module

Location: deriv_bot/risk/risk_manager.py

Purpose:
Implements risk management strategies and trade validation rules.
Handles position sizing, stop-loss calculations, and exposure limits.

Dependencies:
- deriv_bot.monitor.logger: Logging functionality
- deriv_bot.utils.config: Configuration management

Interactions:
- Input: Trade parameters and account status
- Output: Trade validation decisions
- Relations: Used by strategy executor before trade execution

Author: Trading Bot Team
Last modified: 2024-02-26
"""
from deriv_bot.monitor.logger import setup_logger

logger = setup_logger(__name__)

class RiskManager:
    def __init__(self, is_demo=True, max_position_size=100, max_daily_loss=50):
        self.is_demo = is_demo
        # Demo account uses more aggressive parameters
        if is_demo:
            self.max_position_size = max_position_size * 4  # Quadruple the position size for demo
            self.max_daily_loss = max_daily_loss * 5  # 5x the daily loss limit for demo
            self.stop_loss_pct = 0.08  # 8% stop loss for demo
            self.risk_multiplier = 3.0  # More aggressive risk profile
        else:
            self.max_position_size = max_position_size
            self.max_daily_loss = max_daily_loss
            self.stop_loss_pct = 0.01  # 1% stop loss for real
            self.risk_multiplier = 1.0  # Normal risk profile

        self.daily_loss = 0
        self.initial_balance = None
        logger.info(f"RiskManager initialized with {'DEMO' if is_demo else 'REAL'} profile")
        logger.info(f"Parameters: max_position={self.max_position_size}, "
                   f"max_daily_loss={self.max_daily_loss}, "
                   f"stop_loss={self.stop_loss_pct}%, "
                   f"risk_multiplier={self.risk_multiplier}")

        # Store reference to connector for later use
        self.connector = None

    def validate_trade(self, symbol, amount, prediction, connector=None):
        """
        Validate if trade meets risk parameters

        Args:
            symbol: Trading symbol
            amount: Trade amount
            prediction: Predicted price movement
            connector: API connector instance for account operations
        """
        try:
            # Store connector reference if provided
            if connector is not None:
                self.connector = connector

            # More permissive validation for demo account
            if self.is_demo:
                adjusted_amount = amount * self.risk_multiplier
                if adjusted_amount > self.max_position_size:
                    logger.warning(f"Demo trade amount {adjusted_amount} exceeds maximum position size")
                    return False

                if self.daily_loss + adjusted_amount > self.max_daily_loss:
                    logger.warning("Demo maximum daily loss limit would be exceeded")
                    logger.info(f"Current daily loss: {self.daily_loss}, Limit: {self.max_daily_loss}")

                    # Only attempt reset if connector is available
                    if self.connector:
                        # Use async_to_sync to properly await the async function
                        self._reset_demo_balance_sync()
                    else:
                        logger.warning("Cannot reset demo balance: No connector available")
                    return True  # Allow trade after reset in demo

                logger.info(f"Demo trade validated - Amount: {adjusted_amount}, "
                          f"Current daily loss: {self.daily_loss}")
                return True
            else:
                # Strict validation for real account
                if amount > self.max_position_size:
                    logger.warning(f"Trade amount {amount} exceeds maximum position size")
                    return False

                if self.daily_loss + amount > self.max_daily_loss:
                    logger.warning("Maximum daily loss limit would be exceeded")
                    return False

                logger.info(f"Real trade validated - Amount: {amount}, "
                          f"Current daily loss: {self.daily_loss}")
                return True

        except Exception as e:
            logger.error(f"Error in risk validation: {str(e)}")
            return False

    def update_daily_loss(self, loss_amount):
        """Update daily loss tracker"""
        previous_loss = self.daily_loss
        self.daily_loss += loss_amount
        logger.info(f"Updated daily loss: {self.daily_loss} (change: {loss_amount:+.2f})")

        # Auto-reset for demo account if loss is too high
        if self.is_demo and self.daily_loss >= self.max_daily_loss and self.connector:
            logger.warning(f"Demo daily loss ({self.daily_loss}) exceeded limit ({self.max_daily_loss})")
            self._reset_demo_balance_sync()

    def _reset_demo_balance_sync(self):
        """Synchronous wrapper for reset_demo_balance"""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.reset_demo_balance(self.connector))
        except Exception as e:
            logger.error(f"Error in sync reset demo balance: {str(e)}")
            return False

    async def reset_demo_balance(self, connector):
        """Reset demo account balance and loss tracking"""
        if not self.is_demo:
            logger.warning("Balance reset attempted on real account - operation denied")
            return False

        try:
            # Resetear balance virtual a través del conector
            reset_success = await connector.reset_virtual_balance()
            if reset_success:
                previous_loss = self.daily_loss
                self.daily_loss = 0
                logger.info(f"Demo account reset successful - Previous loss: {previous_loss}")
                return True

            logger.error("Failed to reset demo account balance")
            return False

        except Exception as e:
            logger.error(f"Error in demo balance reset: {str(e)}")
            return False

    def check_balance_limits(self, current_balance, connector):
        """Check if balance reset is needed"""
        if not self.is_demo:
            return False

        if current_balance < 100 or self.daily_loss >= self.max_daily_loss:
            logger.warning(f"Demo account limits reached - Balance: {current_balance}, "
                         f"Daily loss: {self.daily_loss}")
            return True

        return False

    def get_risk_profile(self):
        """Return current risk profile settings"""
        return {
            'account_type': 'DEMO' if self.is_demo else 'REAL',
            'max_position_size': self.max_position_size,
            'max_daily_loss': self.max_daily_loss,
            'stop_loss_pct': self.stop_loss_pct,
            'risk_multiplier': self.risk_multiplier,
            'current_daily_loss': self.daily_loss
        }

class RiskManager:
    def __init__(self, is_demo=True, max_position_size=100, max_daily_loss=50):
        self.is_demo = is_demo
        # Demo account uses more aggressive parameters
        if is_demo:
            self.max_position_size = max_position_size * 4  # Quadruple the position size for demo
            self.max_daily_loss = max_daily_loss * 5  # 5x the daily loss limit for demo
            self.stop_loss_pct = 0.08  # 8% stop loss for demo
            self.risk_multiplier = 3.0  # More aggressive risk profile
        else:
            self.max_position_size = max_position_size
            self.max_daily_loss = max_daily_loss
            self.stop_loss_pct = 0.01  # 1% stop loss for real
            self.risk_multiplier = 1.0  # Normal risk profile

        self.daily_loss = 0
        self.initial_balance = None
        logger.info(f"RiskManager initialized with {'DEMO' if is_demo else 'REAL'} profile")
        logger.info(f"Parameters: max_position={self.max_position_size}, "
                   f"max_daily_loss={self.max_daily_loss}, "
                   f"stop_loss={self.stop_loss_pct}%, "
                   f"risk_multiplier={self.risk_multiplier}")

        # Store reference to connector for later use
        self.connector = None

    def validate_trade(self, symbol, amount, prediction, connector=None):
        """
        Validate if trade meets risk parameters

        Args:
            symbol: Trading symbol
            amount: Trade amount
            prediction: Predicted price movement
            connector: API connector instance for account operations
        """
        try:
            # Store connector reference if provided
            if connector is not None:
                self.connector = connector

            # More permissive validation for demo account
            if self.is_demo:
                adjusted_amount = amount * self.risk_multiplier
                if adjusted_amount > self.max_position_size:
                    logger.warning(f"Demo trade amount {adjusted_amount} exceeds maximum position size")
                    return False

                if self.daily_loss + adjusted_amount > self.max_daily_loss:
                    logger.warning("Demo maximum daily loss limit would be exceeded")
                    logger.info(f"Current daily loss: {self.daily_loss}, Limit: {self.max_daily_loss}")

                    # Only attempt reset if connector is available
                    if self.connector:
                        # Use async_to_sync to properly await the async function
                        self._reset_demo_balance_sync()
                    else:
                        logger.warning("Cannot reset demo balance: No connector available")
                    return True  # Allow trade after reset in demo

                logger.info(f"Demo trade validated - Amount: {adjusted_amount}, "
                          f"Current daily loss: {self.daily_loss}")
                return True
            else:
                # Strict validation for real account
                if amount > self.max_position_size:
                    logger.warning(f"Trade amount {amount} exceeds maximum position size")
                    return False

                if self.daily_loss + amount > self.max_daily_loss:
                    logger.warning("Maximum daily loss limit would be exceeded")
                    return False

                logger.info(f"Real trade validated - Amount: {amount}, "
                          f"Current daily loss: {self.daily_loss}")
                return True

        except Exception as e:
            logger.error(f"Error in risk validation: {str(e)}")
            return False

    def update_daily_loss(self, loss_amount):
        """Update daily loss tracker"""
        previous_loss = self.daily_loss
        self.daily_loss += loss_amount
        logger.info(f"Updated daily loss: {self.daily_loss} (change: {loss_amount:+.2f})")

        # Auto-reset for demo account if loss is too high
        if self.is_demo and self.daily_loss >= self.max_daily_loss and self.connector:
            logger.warning(f"Demo daily loss ({self.daily_loss}) exceeded limit ({self.max_daily_loss})")
            self._reset_demo_balance_sync()

    def _reset_demo_balance_sync(self):
        """Synchronous wrapper for reset_demo_balance"""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.reset_demo_balance(self.connector))
        except Exception as e:
            logger.error(f"Error in sync reset demo balance: {str(e)}")
            return False

    async def reset_demo_balance(self, connector):
        """Reset demo account balance and loss tracking"""
        if not self.is_demo:
            logger.warning("Balance reset attempted on real account - operation denied")
            return False

        try:
            # Resetear balance virtual a través del conector
            reset_success = await connector.reset_virtual_balance()
            if reset_success:
                previous_loss = self.daily_loss
                self.daily_loss = 0
                logger.info(f"Demo account reset successful - Previous loss: {previous_loss}")
                return True

            logger.error("Failed to reset demo account balance")
            return False

        except Exception as e:
            logger.error(f"Error in demo balance reset: {str(e)}")
            return False

    def check_balance_limits(self, current_balance, connector):
        """Check if balance reset is needed"""
        if not self.is_demo:
            return False

        if current_balance < 100 or self.daily_loss >= self.max_daily_loss:
            logger.warning(f"Demo account limits reached - Balance: {current_balance}, "
                         f"Daily loss: {self.daily_loss}")
            return True

        return False

    def get_risk_profile(self):
        """Return current risk profile settings"""
        return {
            'account_type': 'DEMO' if self.is_demo else 'REAL',
            'max_position_size': self.max_position_size,
            'max_daily_loss': self.max_daily_loss,
            'stop_loss_pct': self.stop_loss_pct,
            'risk_multiplier': self.risk_multiplier,
            'current_daily_loss': self.daily_loss
        }