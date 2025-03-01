from datetime import datetime

class PriceValidator:
    def __init__(self):
        self.min_price_precision = 0.00001  # 5 decimal places for forex
        self.max_price_age = 5  # seconds

    def validate_price(self, price, timestamp=None):
        """
        Validate price for API submission
        
        Args:
            price (float): Price to validate
            timestamp (float, optional): Price timestamp
            
        Returns:
            bool: True if price is valid
        """
        if not isinstance(price, (int, float)):
            return False
            
        # Check precision
        if round(price % self.min_price_precision, 8) != 0:
            return False
            
        # Check timestamp if provided
        if timestamp:
            now = datetime.now().timestamp()
            if now - timestamp > self.max_price_age:
                return False
                
        return True
