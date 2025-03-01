"""
Module for calculating position sizes based on risk parameters
"""
from deriv_bot.monitor.logger import setup_logger

logger = setup_logger(__name__)

class PositionSizer:
    def __init__(self, max_risk_percent=2.0, max_position_size=100.0):
        """
        Initialize position sizer
        
        Args:
            max_risk_percent: Maximum risk per trade as percentage of account balance
            max_position_size: Maximum position size allowed
        """
        self.max_risk_percent = max_risk_percent
        self.max_position_size = max_position_size
        
    def calculate_position_size(self, account_balance, win_rate, risk_reward_ratio):
        """
        Calculate optimal position size based on Kelly Criterion
        
        Args:
            account_balance: Current account balance
            win_rate: Historical win rate as decimal
            risk_reward_ratio: Risk/Reward ratio of the strategy
        """
        try:
            # Kelly Criterion formula: f = (p(b+1) - 1)/b
            # where: f = fraction of account to risk
            #        p = probability of win
            #        b = odds received on win
            
            kelly_fraction = (win_rate * (risk_reward_ratio + 1) - 1) / risk_reward_ratio
            
            # Conservative adjustment (half Kelly)
            kelly_fraction = kelly_fraction * 0.5
            
            # Calculate position size
            position_size = account_balance * kelly_fraction * (self.max_risk_percent / 100)
            
            # Apply limits
            position_size = min(position_size, self.max_position_size)
            position_size = max(position_size, 1.0)  # Minimum position size
            
            logger.info(f"Calculated position size: {position_size:.2f}")
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return 1.0  # Return minimum position size on error
            
    def adjust_for_volatility(self, base_position_size, volatility_factor):
        """
        Adjust position size based on market volatility
        
        Args:
            base_position_size: Base position size from Kelly calculation
            volatility_factor: Market volatility metric (1.0 = normal volatility)
        """
        try:
            # Reduce position size when volatility is high
            adjusted_size = base_position_size / volatility_factor
            
            # Apply limits
            adjusted_size = min(adjusted_size, self.max_position_size)
            adjusted_size = max(adjusted_size, 1.0)
            
            logger.info(f"Volatility adjusted position size: {adjusted_size:.2f}")
            return adjusted_size
            
        except Exception as e:
            logger.error(f"Error adjusting position size: {str(e)}")
            return base_position_size

