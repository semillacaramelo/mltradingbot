"""
Deriv API Connector Module

Location: deriv_bot/data/deriv_connector.py

Purpose:
Handles WebSocket connections to Deriv.com API, manages authentication,
and provides low-level API communication functionality.
"""
import os
import json
import asyncio
import logging
from typing import Optional, Dict, Any
from deriv_api import DerivAPI  # Using the correct import

logger = logging.getLogger(__name__)

class DerivConnector:
    def __init__(self, config=None):
        self.config = config
        self.app_id = os.getenv('APP_ID', '1089')  # Default APP_ID if not set
        self.api_token = self.config.get_api_token() if config else None
        self.demo = self.config and self.config.is_demo()
        self.api: Optional[DerivAPI] = None
        self._is_connected = False
        self._heartbeat_task = None
        self._req_id = 0
        # Use explicit WebSocket endpoint
        self.endpoint = "ws.binaryws.com/websockets/v3"  # Direct WebSocket endpoint

    async def connect(self):
        """Establish connection to Deriv API with improved error handling"""
        try:
            if self._is_connected:
                return True

            logger.info(f"Connecting to {self.endpoint} with APP_ID: {self.app_id}")
            
            # Create DerivAPI instance with explicit configuration
            self.api = DerivAPI(
                app_id=self.app_id,
                endpoint=self.endpoint,
            )

            # Test basic connection first with shorter timeout
            logger.info("Testing basic connection...")
            try:
                ping_response = await asyncio.wait_for(
                    self.api.ping(),
                    timeout=5.0
                )
                if not ping_response:
                    logger.error("Initial ping failed")
                    return False
                    
                logger.info("Ping successful, proceeding with authorization")
                
            except asyncio.TimeoutError:
                logger.error("Initial ping timed out")
                return False

            logger.info("Basic connection established, authorizing...")
            
            # Authenticate with token
            if self.api_token:
                try:
                    auth_response = await asyncio.wait_for(
                        self.api.authorize(self.api_token),
                        timeout=10.0
                    )
                    
                    if auth_response is None:
                        logger.error("No response from authorization request")
                        return False
                        
                    if "error" in auth_response:
                        logger.error(f"Authorization failed: {auth_response['error']['message']}")
                        return False
                        
                    self._is_connected = True
                    
                    # Start heartbeat
                    if self._heartbeat_task and not self._heartbeat_task.done():
                        self._heartbeat_task.cancel()
                    self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
                    
                    # Log account info
                    balance = auth_response.get('authorize', {}).get('balance', 0)
                    currency = auth_response.get('authorize', {}).get('currency', 'USD')
                    
                    logger.info(f"API authorization successful - {'DEMO' if self.demo else 'REAL'} account")
                    logger.info(f"Account balance: {balance:.2f} {currency}")
                    
                    return True
                    
                except asyncio.TimeoutError:
                    logger.error("Authorization request timed out")
                    return False
                except Exception as auth_error:
                    logger.error(f"Authorization error: {str(auth_error)}")
                    return False
            else:
                logger.error("No API token provided")
                return False
                
        except Exception as e:
            logger.error(f"Connection error: {str(e)}")
            self._is_connected = False
            return False

    async def disconnect(self):
        """Close the API connection"""
        try:
            if self._heartbeat_task:
                self._heartbeat_task.cancel()
                
            if self.api:
                await self.api.disconnect()
                
            self._is_connected = False
            logger.info("Connection closed")
            
        except Exception as e:
            logger.error(f"Error during disconnect: {str(e)}")

    async def _heartbeat_loop(self):
        """Maintain connection with periodic pings"""
        while self._is_connected:
            try:
                await self.api.ping()
                await asyncio.sleep(30)
            except Exception as e:
                logger.error(f"Heartbeat error: {str(e)}")
                break
                
    async def subscribe_to_ticks(self, symbol: str, callback):
        """Subscribe to price ticks for a symbol"""
        try:
            ticks = await self.api.subscribe_to_ticks(symbol)
            self._subscription_refs[symbol] = ticks
            async for tick in ticks:
                await callback(tick)
        except Exception as e:
            logger.error(f"Tick subscription error: {str(e)}")

    async def buy_contract(self, params: Dict[str, Any]):
        """Place a contract purchase"""
        try:
            response = await self.api.buy(params)
            return response
        except Exception as e:
            logger.error(f"Contract purchase error: {str(e)}")
            return None

    async def get_candles(self, symbol: str, count: int = 1000):
        """Get historical candle data"""
        try:
            request = {
                "ticks_history": symbol,
                "end": "latest",
                "style": "candles",
                "granularity": 60,  # 1-minute candles
                "adjust_start_time": 1,
                "count": count
            }
            
            response = await self.api.ticks_history(request)
            if response.get('error'):
                logger.error(f"API error: {response['error']['message']}")
                return []
                
            return response.get('candles', [])
            
        except Exception as e:
            logger.error(f"Error fetching candles: {str(e)}")
            return []

    async def get_server_time(self):
        """Get server time"""
        try:
            response = await self.api.time()
            return response
        except Exception as e:
            logger.error(f"Error fetching server time: {str(e)}")
            return None

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    async def check_connection(self):
        """Check if connection is active and working"""
        try:
            if not self._is_connected or not self.api:
                return False
                
            # Test connection with ping
            response = await self.api.ping()
            return response and not response.get('error')
            
        except Exception as e:
            logger.error(f"Connection check failed: {str(e)}")
            return False
            
    async def close(self):
        """Properly close the connection"""
        try:
            if self.api:
                await self.api.disconnect()
                self._is_connected = False
                logger.info("Connection closed cleanly")
        except Exception as e:
            logger.error(f"Error closing connection: {str(e)}")

    async def reconnect(self):
        """Attempt to reconnect to the API"""
        try:
            await self.close()
            await asyncio.sleep(1)  # Brief pause before reconnecting
            return await self.connect()
        except Exception as e:
            logger.error(f"Reconnection failed: {str(e)}")
            return False
            
    def _get_request_id(self):
        """Generate unique request ID"""
        self._req_id += 1
        return self._req_id

    async def get_active_symbols(self):
        """Get list of active trading symbols"""
        try:
            if not self._is_connected:
                await self.connect()
            return await self.api.active_symbols()
        except Exception as e:
            logger.error(f"Error getting active symbols: {str(e)}")
            return None
            
    async def send_request(self, request):
        """Send a request to the API"""
        try:
            if not self._is_connected:
                await self.connect()
            return await self.api.send(request)
        except Exception as e:
            logger.error(f"Error sending request: {str(e)}")
            return {"error": {"message": str(e)}}