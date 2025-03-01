"""
Module for validating trades against risk parameters
"""
from deriv_bot.monitor.logger import setup_logger

logger = setup_logger(__name__)

class RiskValidator:
    def __init__(self):
        self.max_daily_trades = 20
        self.max_concurrent_trades = 3
        self.max_daily_drawdown = 0.05  # 5% of account balance
        self.min_distance_between_trades = 300  # seconds
        
    def validate_trade(self, trade_params, account_state):
        """
        Validate trade against risk parameters
        
        Args:
            trade_params: Dictionary containing trade parameters
            account_state: Dictionary containing account state
        """
        try:
            validations = [
                self._validate_daily_limits(account_state),
                self._validate_drawdown(trade_params, account_state),
                self._validate_exposure(trade_params, account_state),
                self._validate_trade_frequency(account_state)
            ]
            
            # All validations must pass
            is_valid = all(validations)
            
            if not is_valid:
                logger.warning("Trade validation failed")
            
            return is_valid
            
        except Exception as e:
            logger.error(f"Error in trade validation: {str(e)}")
            return False
            
    def _validate_daily_limits(self, account_state):
        """Check if daily trade limits are exceeded"""
        if account_state['daily_trades'] >= self.max_daily_trades:
            logger.warning("Daily trade limit exceeded")
            return False
            
        if account_state['concurrent_trades'] >= self.max_concurrent_trades:
            logger.warning("Maximum concurrent trades reached")
            return False
            
        return True
        
    def _validate_drawdown(self, trade_params, account_state):
        """Check if trade would exceed maximum drawdown"""
        potential_loss = trade_params['stake_amount']
        daily_drawdown = account_state['daily_drawdown']
        
        if (daily_drawdown + potential_loss) / account_state['initial_balance'] > self.max_daily_drawdown:
            logger.warning("Trade would exceed maximum daily drawdown")
            return False
            
        return True
        
    def _validate_exposure(self, trade_params, account_state):
        """Check if trade would exceed maximum market exposure"""
        total_exposure = account_state['current_exposure'] + trade_params['stake_amount']
        
        if total_exposure > account_state['balance'] * 0.2:  # Max 20% account exposure
            logger.warning("Trade would exceed maximum market exposure")
            return False
            
        return True
        
    def _validate_trade_frequency(self, account_state):
        """Check if minimum time between trades has elapsed"""
        if 'last_trade_time' in account_state:
            time_since_last_trade = account_state['current_time'] - account_state['last_trade_time']
            
            if time_since_last_trade < self.min_distance_between_trades:
                logger.warning("Minimum time between trades not elapsed")
                return False
                
        return True

