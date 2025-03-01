"""
Market Data Fetcher Module

Location: deriv_bot/data/data_fetcher.py

Purpose:
Retrieves and formats market data from Deriv API, handles data streaming,
and provides historical data access.

Dependencies:
- pandas: Data manipulation and analysis
- numpy: Numerical operations
- deriv_bot.data.deriv_connector: API connection handling

Interactions:
- Input: API queries, market symbols
- Output: Formatted market data (DataFrame)
- Relations: Used by main trading loop and model training

Author: Trading Bot Team
Last modified: 2025-02-27
"""
import pandas as pd
import numpy as np
import asyncio
import time
import sys  # Added for memory size calculations
from deriv_bot.monitor.logger import setup_logger

logger = setup_logger(__name__)

class DataFetcher:
    def __init__(self, connector):
        self.connector = connector
        self.last_fetch_time = {}  # Track last request timestamp per symbol to limit request frequency
        self.fetch_cooldown = 10   # Minimum time between requests for the same symbol
        self.cache = {}            # Simple cache of data by symbol and interval
        self.cache_expiry = 3600   # Cache expiry in seconds (1 hour default)

    async def check_trading_enabled(self, symbol):
        """
        Check if trading is enabled for the symbol with improved error handling
        """
        try:
            # First check connection
            if not await self.connector.check_connection():
                logger.warning("Connection not available, attempting to reconnect...")
                if not await self.connector.reconnect():
                    return False

            # Add delay to avoid rate limiting
            await asyncio.sleep(1)
            
            # Try to get active symbols with retry
            for attempt in range(3):
                try:
                    active_symbols = await self.connector.get_active_symbols()
                    if active_symbols and not isinstance(active_symbols, dict):
                        # Handle case where response is not in expected format
                        logger.error(f"Unexpected response format: {type(active_symbols)}")
                        continue
                        
                    if active_symbols and "error" not in active_symbols:
                        for sym in active_symbols.get("active_symbols", []):
                            if sym["symbol"] == symbol:
                                return sym["exchange_is_open"] == 1
                        break
                    else:
                        error_msg = active_symbols.get("error", {}).get("message", "Unknown error")
                        logger.warning(f"Error getting active symbols (attempt {attempt+1}): {error_msg}")
                        
                    if attempt < 2:  # Don't sleep on last attempt
                        await asyncio.sleep(2 * (attempt + 1))
                        
                except Exception as e:
                    logger.error(f"Error checking symbol availability (attempt {attempt+1}): {str(e)}")
                    if attempt < 2:
                        await asyncio.sleep(2 * (attempt + 1))

            return False
            
        except Exception as e:
            logger.error(f"Error in check_trading_enabled: {str(e)}")
            return False

    def is_symbol_available(self, symbol):
        """
        Synchronous method to check if a symbol is available for trading.
        This method is used by AssetSelector.

        Args:
            symbol: Symbol to check

        Returns:
            bool: True if the symbol is available, False otherwise
        """
        # Create a loop event to call the asynchronous method
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If the loop is already running, create a task
                future = asyncio.run_coroutine_threadsafe(
                    self.check_trading_enabled(symbol), loop)
                return future.result(timeout=10)  # 10 second timeout
            else:
                # If no loop is running, execute directly
                return loop.run_until_complete(self.check_trading_enabled(symbol))
        except Exception as e:
            logger.error(f"Error in is_symbol_available for {symbol}: {str(e)}")
            return False

    async def fetch_historical_data(self, symbol, interval, count=1000, retry_attempts=5, use_cache=True):
        """
        Fetch historical data with improved error handling and caching

        Args:
            symbol: Trading symbol
            interval: Candle interval in seconds
            count: Number of candles to request
            retry_attempts: Number of retry attempts
            use_cache: Whether to use cached data if available

        Returns:
            DataFrame with historical data or None if failed
        """
        # First check connection availability
        if not await self.connector.check_connection():
            logger.warning("Connection not available, attempting to reconnect...")
            if not await self.connector.reconnect():
                logger.error("Failed to establish connection")
                return None

        # Check if trading is enabled for this symbol
        if not await self.check_trading_enabled(symbol):
            logger.error(f"Trading not available for {symbol}")
            return None

        cache_key = f"{symbol}_{interval}"

        # Check rate limiting
        current_time = time.time()
        if symbol in self.last_fetch_time:
            time_since_last = current_time - self.last_fetch_time[symbol]
            if time_since_last < self.fetch_cooldown:
                # If we have cache and we're within cooldown period, use cache
                if use_cache and cache_key in self.cache:
                    logger.debug(f"Using cached data for {symbol} (cooldown: {self.fetch_cooldown - time_since_last:.1f}s)")
                    return self.cache[cache_key]

                # If we need to wait, calculate remaining time
                wait_time = self.fetch_cooldown - time_since_last
                logger.debug(f"Rate limiting for {symbol}, waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time)

        # Attempt up to the specified number of retries
        for attempt in range(retry_attempts):
            try:
                # Validate symbol format
                if not symbol.startswith(('frx', 'R_')):
                    logger.error(f"Invalid symbol format: {symbol}")
                    return None

                # Check connection status
                if not await self.connector.check_connection():
                    logger.warning(f"Connection not available for {symbol} data fetch, attempt {attempt+1}/{retry_attempts}")
                    if attempt < retry_attempts - 1:
                        await asyncio.sleep(2 * (attempt + 1))  # Increasing wait between attempts
                    continue

                # Check if trading is enabled
                if not await self.check_trading_enabled(symbol):
                    logger.warning(f"Trading not available for {symbol} at this time")
                    return None

                # Request historical data
                # Request more candles to compensate for potential missing data
                adjusted_count = min(int(count * 1.2), 5000)  # 20% more, maximum 5000

                request = {
                    "ticks_history": symbol,
                    "adjust_start_time": 1,
                    "count": adjusted_count,
                    "end": "latest",
                    "granularity": interval,
                    "style": "candles",
                    "req_id": self.connector._get_request_id()
                }

                response = await self.connector.send_request(request)

                # Update last request timestamp
                self.last_fetch_time[symbol] = time.time()

                if "error" in response:
                    error_msg = response["error"]["message"]
                    logger.error(f"Error fetching historical data: {error_msg}")
                    if attempt < retry_attempts - 1:
                        await asyncio.sleep(2 * (attempt + 1))
                    continue

                # Validate response structure
                if "candles" not in response:
                    logger.error("Invalid response: missing candles data")
                    if attempt < retry_attempts - 1:
                        await asyncio.sleep(2 * (attempt + 1))
                    continue

                candles = response["candles"]

                # Check if we have enough candles
                if len(candles) < count:
                    logger.warning(f"Received fewer candles than requested: {len(candles)} vs {count}")
                    # If it's too few, try with a higher count
                    if len(candles) < count * 0.8 and attempt < retry_attempts - 1:  # Less than 80% of requested
                        increased_count = min(int(count * 1.5), 5000)  # Increase by 50% but not more than 5000
                        logger.warning(f"Requesting increased count of {increased_count} candles...")
                        request["count"] = increased_count
                        await asyncio.sleep(2)
                        continue

                # Create DataFrame
                df = pd.DataFrame([{
                    'time': candle['epoch'],
                    'open': float(candle['open']),
                    'high': float(candle['high']),
                    'low': float(candle['low']),
                    'close': float(candle['close'])
                } for candle in candles])

                # Convert timestamp and set index
                df['time'] = pd.to_datetime(df['time'], unit='s')
                df.set_index('time', inplace=True)

                # Sort index to ensure chronological order
                df.sort_index(inplace=True)

                # Save to cache
                self.cache[cache_key] = df

                logger.info(f"Successfully fetched {len(df)} candles for {symbol}")
                return df

            except Exception as e:
                logger.error(f"Error in fetch_historical_data: {str(e)}")
                if attempt < retry_attempts - 1:
                    await asyncio.sleep(2 * (attempt + 1))

        # If we get here, all attempts failed
        # Try to return cache as a last resort
        if use_cache and cache_key in self.cache:
            logger.warning(f"All fetch attempts failed for {symbol}, using cached data")
            return self.cache[cache_key]

        return None

    async def fetch_sufficient_data(self, symbol, interval, min_required_samples, max_attempts=3):
        """
        Ensures that sufficient samples are obtained for analysis

        Args:
            symbol: Trading symbol
            interval: Candle interval in seconds
            min_required_samples: Minimum number of samples required
            max_attempts: Maximum number of attempts

        Returns:
            DataFrame with sufficient historical data or None if failed
        """
        for attempt in range(max_attempts):
            # Calculate how many candles we need with a margin
            count_with_margin = min_required_samples * 1.5
            # Limited to 3000 to not exceed API limits
            count_to_fetch = min(3000, int(count_with_margin))

            logger.info(f"Fetching {count_to_fetch} candles to ensure {min_required_samples} samples (attempt {attempt+1})")
            df = await self.fetch_historical_data(symbol, interval, count=count_to_fetch)

            if df is None:
                logger.warning(f"Failed to fetch data, attempt {attempt+1}/{max_attempts}")
                if attempt < max_attempts - 1:
                    await asyncio.sleep(3)
                continue

            # Check if we have enough data
            if len(df) >= min_required_samples:
                logger.info(f"Successfully obtained {len(df)} samples (needed {min_required_samples})")
                return df
            else:
                logger.warning(f"Insufficient data: {len(df)} samples, need at least {min_required_samples}")
                # If it's the last attempt, return what we have
                if attempt == max_attempts - 1:
                    logger.warning("Returning incomplete data after maximum attempts")
                    return df
                await asyncio.sleep(2)

        return None

    async def subscribe_to_ticks(self, symbol, retry_attempts=3):
        """
        Subscribe to real-time price ticks

        Args:
            symbol: Trading symbol (e.g., "frxEURUSD")
            retry_attempts: Number of retry attempts

        Returns:
            Subscription response or None if failed
        """
        for attempt in range(retry_attempts):
            try:
                # Check connection status
                if not await self.connector.check_connection():
                    logger.warning(f"Connection not available for tick subscription, attempt {attempt+1}/{retry_attempts}")
                    if attempt < retry_attempts - 1:
                        await asyncio.sleep(2 * (attempt + 1))
                    continue

                response = await self.connector.subscribe_to_ticks(symbol)

                if "error" in response:
                    logger.error(f"Error subscribing to ticks: {response['error']['message']}")
                    if attempt < retry_attempts - 1:
                        await asyncio.sleep(2 * (attempt + 1))
                    continue

                logger.info(f"Successfully subscribed to ticks for {symbol}")
                return response

            except Exception as e:
                logger.error(f"Error subscribing to ticks: {str(e)}")
                if attempt < retry_attempts - 1:
                    await asyncio.sleep(2 * (attempt + 1))

        return None

    async def get_available_symbols(self, retry_attempts=3):
        """
        Get list of available trading symbols

        Args:
            retry_attempts: Number of retry attempts

        Returns:
            List of available symbols or None if failed
        """
        for attempt in range(retry_attempts):
            try:
                # Check connection status
                if not await self.connector.check_connection():
                    logger.warning(f"Connection not available for symbol list, attempt {attempt+1}/{retry_attempts}")
                    if attempt < retry_attempts - 1:
                        await asyncio.sleep(2 * (attempt + 1))
                    continue

                response = await self.connector.get_active_symbols()

                if "error" in response:
                    logger.error(f"Error fetching symbols: {response['error']['message']}")
                    if attempt < retry_attempts - 1:
                        await asyncio.sleep(2 * (attempt + 1))
                    continue

                symbols = []
                if "active_symbols" in response:
                    symbols = [symbol['symbol'] for symbol in response['active_symbols']]
                    logger.info(f"Found {len(symbols)} active trading symbols")

                return symbols

            except Exception as e:
                logger.error(f"Error fetching available symbols: {str(e)}")
                if attempt < retry_attempts - 1:
                    await asyncio.sleep(2 * (attempt + 1))

        return None

    def clear_cache(self, symbol=None, older_than=3600):
        """
        Clear data cache

        Args:
            symbol: Specific symbol to clear (None for all)
            older_than: Remove entries older than these seconds
        """
        current_time = time.time()

        if symbol:
            # Clear only for the specified symbol
            keys_to_remove = [k for k in self.cache if k.startswith(f"{symbol}_")]
            for key in keys_to_remove:
                del self.cache[key]
                if symbol in self.last_fetch_time:
                    del self.last_fetch_time[symbol]
            logger.debug(f"Cleared cache for {symbol}")
        else:
            # Clear old entries for all symbols
            for symbol in list(self.last_fetch_time.keys()):
                if current_time - self.last_fetch_time[symbol] > older_than:
                    # Remove all cache entries for this symbol
                    keys_to_remove = [k for k in self.cache if k.startswith(f"{symbol}_")]
                    for key in keys_to_remove:
                        del self.cache[key]
                    del self.last_fetch_time[symbol]

            logger.debug("Cleared expired cache entries")

    def get_cache_info(self):
        """
        Get information about the current cache state

        Returns:
            dict: Cache statistics
        """
        stats = {
            'total_cached_items': len(self.cache),
            'symbols_in_cache': len(set([k.split('_')[0] for k in self.cache.keys()])) if self.cache else 0,
            'cache_size_kb': sum(sys.getsizeof(df) for df in self.cache.values()) / 1024 if self.cache else 0,
        }
        return stats

    async def optimize_cache(self, max_size_mb=100):
        """
        Optimize cache by removing least recently used items
        when the cache exceeds the maximum size

        Args:
            max_size_mb: Maximum cache size in MB
        """
        try:
            # Calculate current cache size
            current_size_bytes = sum(sys.getsizeof(df) for df in self.cache.values())
            current_size_mb = current_size_bytes / (1024 * 1024)

            if current_size_mb > max_size_mb:
                logger.info(f"Cache size ({current_size_mb:.2f} MB) exceeds limit ({max_size_mb} MB), optimizing...")

                # Sort items by last fetch time
                sorted_items = sorted(
                    [(k, self.last_fetch_time.get(k.split('_')[0], 0)) for k in self.cache.keys()],
                    key=lambda x: x[1]
                )

                # Remove oldest items until we're under the limit
                items_removed = 0
                while current_size_mb > max_size_mb * 0.8 and sorted_items:  # Target 80% of max
                    oldest_key = sorted_items.pop(0)[0]
                    if oldest_key in self.cache:
                        current_size_mb -= sys.getsizeof(self.cache[oldest_key]) / (1024 * 1024)
                        del self.cache[oldest_key]
                        items_removed += 1

                logger.info(f"Cache optimization complete: removed {items_removed} items, " 
                           f"new size: {current_size_mb:.2f} MB")
        except Exception as e:
            logger.error(f"Error optimizing cache: {str(e)}")