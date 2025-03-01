"""
Module for executing trading orders
"""
from deriv_bot.monitor.logger import setup_logger
from deriv_bot.validation.price_validator import PriceValidator
import time
from typing import Optional

logger = setup_logger(__name__)

class OrderExecutor:
    def __init__(self, api_connector, risk_manager=None):  # Make risk_manager optional
        self.api = api_connector
        self.risk_manager = risk_manager
        self.price_validator = PriceValidator()
        self.price_cache = {}
        self.price_cache_validity = 2  # seconds

    async def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get latest price with caching to prevent rapid requests"""
        now = time.time()
        
        if symbol in self.price_cache:
            price, timestamp = self.price_cache[symbol]
            if now - timestamp < self.price_cache_validity:
                return price

        try:
            response = await self.api.send_request({
                "ticks": symbol
            })

            if "error" in response:
                logger.error(f"Error getting price: {response['error']['message']}")
                return None

            price = response.get("tick", {}).get("quote")
            if price:
                self.price_cache[symbol] = (price, now)
                return price

            return None

        except Exception as e:
            logger.error(f"Error fetching price: {str(e)}")
            return None

    async def place_order(self, symbol, contract_type, amount, duration, stop_loss_pct=None):
        """Place a trade with proper parameter validation"""
        try:
            quote = await self.api.get_price_quote(symbol)
            
            if not self.price_validator.validate_price(quote['price'], quote['timestamp']):
                raise ValueError("Invalid price or quote expired")
                
            # Continue with order placement
            order_params = {
                "symbol": symbol,
                "contract_type": contract_type,
                "amount": amount,
                "duration": duration,
                "price": quote['price']
            }
            
            if stop_loss_pct:
                order_params["stop_loss"] = self._calculate_stop_loss(
                    quote['price'], 
                    stop_loss_pct,
                    contract_type
                )
                
            return await self.api.buy_contract(**order_params)
            
        except Exception as e:
            logger.error(f"Order placement failed: {str(e)}")
            return None

    async def close_position(self, contract_id):
        """
        Close an open position

        Args:
            contract_id: ID of the contract to close
        """
        try:
            logger.info(f"Closing position {contract_id}")

            request = {
                "sell": contract_id
            }

            response = await self.api.send_request(request)

            if "error" in response:
                logger.error(f"Position closure failed: {response['error']['message']}")
                return False

            logger.info(f"Position closed successfully: {contract_id}")
            logger.debug(f"Close position response: {response}")
            return True

        except Exception as e:
            logger.error(f"Error closing position: {str(e)}")
            return False

    async def check_position_status(self, contract_id):
        """
        Check the status of an open position

        Args:
            contract_id: ID of the contract to check
        """
        try:
            request = {
                "proposal_open_contract": 1,
                "contract_id": contract_id
            }

            response = await self.api.send_request(request)

            if "error" in response:
                logger.error(f"Status check failed: {response['error']['message']}")
                return None

            logger.debug(f"Position status: {response}")
            return response.get('proposal_open_contract')

        except Exception as e:
            logger.error(f"Error checking position status: {str(e)}")
            return None