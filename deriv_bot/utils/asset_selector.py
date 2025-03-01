"""
Asset Selection Module

Location: deriv_bot/utils/asset_selector.py

Purpose:
Manages asset selection logic including market hours,
asset availability, and selection strategies.

Dependencies:
- pandas: Data analysis
- deriv_bot.monitor.logger: Logging functionality
- deriv_bot.data.data_fetcher: Market data access

Interactions:
- Input: Market conditions and time data
- Output: Asset selection decisions
- Relations: Used by main loop for symbol selection

Author: Trading Bot Team
Last modified: 2024-02-26
"""

import logging
import datetime
from typing import List, Dict, Optional, Tuple
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo
import pandas as pd
import numpy as np

# Configurar logger
logger = logging.getLogger(__name__)

# Definir zonas horarias para los diferentes mercados
UTC = ZoneInfo("UTC")
NEW_YORK = ZoneInfo("America/New_York")
LONDON = ZoneInfo("Europe/London")
TOKYO = ZoneInfo("Asia/Tokyo")
SYDNEY = ZoneInfo("Australia/Sydney")

# Assets disponibles 24/7 (para fallback)
ALWAYS_AVAILABLE_ASSETS = [
    "frxEURUSD",  # EUR/USD
    "frxBTCUSD",  # Bitcoin/USD
    "R_10",       # Volatility Index 10
    "R_25",       # Volatility Index 25
    "R_50",       # Volatility Index 50
    "R_75",       # Volatility Index 75
    "R_100",      # Volatility Index 100
]

# Definir horarios de mercado para diferentes activos
# format: "symbol": [(día_inicio, hora_inicio, día_fin, hora_fin, zona_horaria), ...]
# día_inicio y día_fin: 0=domingo, 1=lunes, ..., 6=sábado
MARKET_HOURS: Dict[str, List[Tuple[int, time, int, time, ZoneInfo]]] = {
    # Forex majors - horarios aproximados
    "frxEURUSD": [(0, time(0, 0), 4, time(23, 59), UTC), (6, time(22, 0), 6, time(23, 59), UTC)],  # Domingo-Jueves completo, Viernes parcial
    "frxUSDJPY": [(0, time(0, 0), 4, time(23, 59), UTC), (6, time(22, 0), 6, time(23, 59), UTC)],
    "frxGBPUSD": [(0, time(0, 0), 4, time(23, 59), UTC), (6, time(22, 0), 6, time(23, 59), UTC)],

    # Indices bursátiles con horarios específicos
    "OTC_SPX": [(1, time(9, 30), 5, time(16, 0), NEW_YORK)],  # S&P 500 - Lunes a Viernes, horario NYSE
    "OTC_DJI": [(1, time(9, 30), 5, time(16, 0), NEW_YORK)],  # Dow Jones - Lunes a Viernes, horario NYSE
    "OTC_N225": [(1, time(9, 0), 5, time(15, 0), TOKYO)],     # Nikkei 225 - Lunes a Viernes, horario TSE
    "OTC_FTSE": [(1, time(8, 0), 5, time(16, 30), LONDON)],   # FTSE 100 - Lunes a Viernes, horario LSE

    # Materias primas
    "frxXAUUSD": [(0, time(23, 0), 5, time(22, 0), UTC)],     # Oro (Gold) - Domingo noche a Viernes

    # Criptomonedas (disponibles 24/7)
    "frxBTCUSD": [(0, time(0, 0), 6, time(23, 59), UTC)],     # Bitcoin - Todos los días
    "frxETHUSD": [(0, time(0, 0), 6, time(23, 59), UTC)],     # Ethereum - Todos los días

    # Índices de volatilidad (disponibles 24/7)
    "R_10": [(0, time(0, 0), 6, time(23, 59), UTC)],          # Volatility 10 Index - Todos los días
    "R_25": [(0, time(0, 0), 6, time(23, 59), UTC)],          # Volatility 25 Index - Todos los días
    "R_50": [(0, time(0, 0), 6, time(23, 59), UTC)],          # Volatility 50 Index - Todos los días
    "R_75": [(0, time(0, 0), 6, time(23, 59), UTC)],          # Volatility 75 Index - Todos los días
    "R_100": [(0, time(0, 0), 6, time(23, 59), UTC)],         # Volatility 100 Index - Todos los días
}

# Market hours definitions (in UTC)
MARKET_HOURS = {
    'forex': {
        'standard': {
            'monday': [(0, 24)],  # 24h
            'tuesday': [(0, 24)],
            'wednesday': [(0, 24)],
            'thursday': [(0, 24)],
            'friday': [(0, 24)],
            'saturday': [],  # Closed
            'sunday': [(21, 24)]  # Opens at 21:00 UTC
        }
    }
}

# Definir preferencias de activos por hora del día
# [(hora_inicio, hora_fin, [lista de activos en orden de preferencia])]
TIME_BASED_PREFERENCES = [
    # Madrugada/Mañana temprano (0:00-8:00 UTC): Mercados asiáticos más activos
    (time(0, 0), time(8, 0), ["frxUSDJPY", "OTC_N225", "frxEURUSD", "frxGBPUSD"]),

    # Mañana/Tarde (8:00-16:00 UTC): Mercados europeos más activos, apertura de US
    (time(8, 0), time(16, 0), ["frxEURUSD", "frxGBPUSD", "OTC_FTSE", "OTC_SPX", "OTC_DJI"]),

    # Tarde/Noche (16:00-24:00 UTC): Mercados americanos más activos
    (time(16, 0), time(23, 59), ["frxEURUSD", "OTC_SPX", "OTC_DJI", "frxXAUUSD", "frxGBPUSD"]),
]


class AssetSelector:
    """
    Clase para seleccionar activos basados en el horario actual y disponibilidad de mercado.
    """

    def __init__(self, data_fetcher=None, preferred_assets=None):
        """
        Inicializa el selector de activos.

        Args:
            data_fetcher: Instancia del fetcher de datos para verificar disponibilidad real
            preferred_assets: Lista opcional de activos preferidos por el usuario
        """
        self.data_fetcher = data_fetcher
        self.preferred_assets = preferred_assets or []
        self.available_assets_cache = {}
        self.cache_timestamp = None
        self.cache_validity = timedelta(minutes=15)  # Validez del caché
        self.unavailable_assets_cache = {}  # New cache for temporarily unavailable assets
        self.unavailable_cache_validity = timedelta(minutes=5)  # How long to remember unavailable assets
        self.volatility_cache = {}
        self.volatility_timestamp = None
        self.volatility_cache_validity = timedelta(minutes=5)
        self.min_volatility_threshold = 0.0001  # Minimum volatility threshold

    def is_market_open(self, symbol: str, current_datetime: Optional[datetime] = None) -> bool:
        """
        Verifica si el mercado para un símbolo específico está abierto según su horario programado.
        Implementa lógica mejorada para mercados de acciones y considera zonas horarias.

        Args:
            symbol: Símbolo del activo a verificar
            current_datetime: Fecha y hora actual (UTC). Si es None, se usa la hora actual.

        Returns:
            bool: True si el mercado está abierto, False en caso contrario
        """
        if current_datetime is None:
            current_datetime = datetime.now(UTC)

        # Si el símbolo no está en el diccionario de horarios, asumimos que no está disponible
        if symbol not in MARKET_HOURS:
            logger.warning(f"No hay información de horario para el símbolo: {symbol}")
            return False

        # Obtener el día de la semana (0=lunes, 6=domingo) según el formato de Python
        weekday = current_datetime.weekday()

        # Convertir de formato Python (0=lunes, 6=domingo) a formato de calendario común (0=domingo, 6=sábado)
        calendar_weekday = (weekday + 1) % 7

        # Verificar cada rango de horario definido para el símbolo
        for start_day, start_time, end_day, end_time, timezone in MARKET_HOURS[symbol]:
            # Convertir la fecha y hora actual a la zona horaria del mercado
            market_datetime = current_datetime.astimezone(timezone)

            # Obtener hora actual en la zona horaria del mercado
            market_time = market_datetime.time()

            # Obtener día de la semana en la zona horaria del mercado (formato Python: 0=lunes)
            market_weekday = market_datetime.weekday()

            # Convertir a formato de calendario común (0=domingo, 6=sábado)
            market_calendar_weekday = (market_weekday + 1) % 7

            # Si es el mismo día, simplemente verificamos el rango de horas
            if start_day == end_day and market_calendar_weekday == start_day:
                if start_time <= market_time <= end_time:
                    return True

            # Si es un rango que cruza días
            elif start_day <= end_day:
                # Caso normal: ej. Lunes(1) a Viernes(5)
                if start_day <= market_calendar_weekday <= end_day:
                    # Primer día del rango: verificar que sea después de la hora de inicio
                    if market_calendar_weekday == start_day and market_time >= start_time:
                        return True
                    # Último día del rango: verificar que sea antes de la hora de fin
                    elif market_calendar_weekday == end_day and market_time <= end_time:
                        return True
                    # Días intermedios: mercado abierto todo el día
                    elif start_day < market_calendar_weekday < end_day:
                        return True
            else:
                # Caso que cruza fin de semana: ej. Viernes(5) a Lunes(1)
                if market_calendar_weekday >= start_day or market_calendar_weekday <= end_day:
                    # Primer día del rango
                    if market_calendar_weekday == start_day and market_time >= start_time:
                        return True
                    # Último día del rango
                    elif market_calendar_weekday == end_day and market_time <= end_time:
                        return True
                    # Días intermedios
                    elif (market_calendar_weekday > start_day or market_calendar_weekday < end_day):
                        return True

        return False

    def verify_asset_availability(self, symbol: str) -> bool:
        """
        Verifica si un activo está realmente disponible para trading.

        Args:
            symbol: Símbolo del activo a verificar

        Returns:
            bool: True si el activo está disponible, False en caso contrario
        """
        # Primero verificamos si el mercado debería estar abierto según el horario
        if not self.is_market_open(symbol):
            logger.debug(f"Mercado cerrado para {symbol} según horario programado")
            return False

        # Si tenemos un data_fetcher, verificamos la disponibilidad real mediante la API
        if self.data_fetcher:
            try:
                # Intentamos obtener el último tick para verificar disponibilidad
                # Esto podría ser reemplazado por una llamada más específica si está disponible
                is_available = self.data_fetcher.is_symbol_available(symbol)
                return is_available
            except Exception as e:
                logger.warning(f"Error al verificar disponibilidad de {symbol}: {e}")
                # Si hay un error, asumimos que no está disponible
                return False

        # Si no hay data_fetcher, asumimos que está disponible si el mercado está abierto
        return True

    async def is_trading_available(self, symbol: str) -> bool:
        """
        Checks if actual trading is available for the symbol by attempting to get a price quote.
        
        Args:
            symbol: Symbol to check
            
        Returns:
            bool: True if trading is available, False otherwise
        """
        # Check if we recently found this asset to be unavailable
        if symbol in self.unavailable_assets_cache:
            last_check, _ = self.unavailable_assets_cache[symbol]
            if datetime.now(UTC) - last_check < self.unavailable_cache_validity:
                return False

        if not self.data_fetcher:
            return True

        try:
            # Try to get a price quote
            quote = await self.data_fetcher.get_price_quote(symbol)
            if quote and "error" not in quote:
                # Clear from unavailable cache if it was there
                self.unavailable_assets_cache.pop(symbol, None)
                return True
            
            # If we get here, trading is not available
            self.unavailable_assets_cache[symbol] = (datetime.now(UTC), quote.get("error", {}).get("message", "Unknown error"))
            return False
        except Exception as e:
            logger.warning(f"Error checking trading availability for {symbol}: {e}")
            self.unavailable_assets_cache[symbol] = (datetime.now(UTC), str(e))
            return False

    async def get_available_assets(self, force_refresh=False) -> List[str]:
        """
        Gets a list of all assets available for trading right now.
        
        Args:
            force_refresh: If True, ignores cache and checks current availability
            
        Returns:
            List[str]: List of available asset symbols
        """
        current_time = datetime.now(UTC)

        if (not force_refresh and 
            self.cache_timestamp and 
            current_time - self.cache_timestamp < self.cache_validity and
            self.available_assets_cache):
            return list(self.available_assets_cache)

        available_assets = []

        # Check each asset in our market hours dictionary
        for symbol in MARKET_HOURS.keys():
            if self.is_market_open(symbol):
                # Add extra check for actual trading availability
                if await self.is_trading_available(symbol):
                    available_assets.append(symbol)

        # Update cache
        self.available_assets_cache = available_assets
        self.cache_timestamp = current_time

        return available_assets

    def get_preferred_assets_by_time(self, current_time=None) -> List[str]:
        """
        Obtiene una lista de activos preferidos según la hora del día.

        Args:
            current_time: Hora actual (objeto time). Si es None, se usa la hora actual.

        Returns:
            List[str]: Lista de símbolos de activos en orden de preferencia
        """
        if current_time is None:
            current_time = datetime.now(UTC).time()

        # Buscar en las preferencias por hora
        for start_time, end_time, assets in TIME_BASED_PREFERENCES:
            # Manejar el caso especial cuando el rango cruza la medianoche
            if start_time <= end_time:
                if start_time <= current_time <= end_time:
                    return assets
            else:
                if current_time >= start_time or current_time <= end_time:
                    return assets

        # Si no encontramos ninguna coincidencia (no debería ocurrir), devolvemos la primera lista
        return TIME_BASED_PREFERENCES[0][2]

    async def select_asset(self, preferred_asset=None) -> Optional[str]:
        """
        Selects the best available asset for trading.
        
        Args:
            preferred_asset: Specific preferred asset for this selection
            
        Returns:
            str or None: Selected asset symbol, or None if no assets are available
        """
        logger.info("Starting asset selection")

        # Get available assets
        available_assets = await self.get_available_assets()

        # Function to check if an asset is truly available for trading
        async def is_asset_valid(asset):
            return asset in available_assets and await self.is_trading_available(asset)

        # Try preferred asset first
        if preferred_asset and await is_asset_valid(preferred_asset):
            logger.info(f"Using specific preferred asset: {preferred_asset}")
            return preferred_asset

        # Try user's preferred assets
        if self.preferred_assets:
            for asset in self.preferred_assets:
                if await is_asset_valid(asset):
                    logger.info(f"Using user's preferred asset: {asset}")
                    return asset

        # Try time-based preferences
        time_based_preferences = self.get_preferred_assets_by_time()
        for asset in time_based_preferences:
            if await is_asset_valid(asset):
                logger.info(f"Using time-based preferred asset: {asset}")
                return asset

        # Try 24/7 assets
        for asset in ALWAYS_AVAILABLE_ASSETS:
            if await is_asset_valid(asset):
                logger.info(f"Using 24/7 fallback asset: {asset}")
                return asset

        # Try any available asset
        for asset in available_assets:
            if await is_asset_valid(asset):
                logger.info(f"Using first available asset: {asset}")
                return asset

        logger.warning("No tradable assets found")
        return None

    def is_asset_tradeable(self, symbol: str) -> bool:
        """Check if an asset is available for trading right now"""
        # First check if market is open
        if not self.is_market_open(symbol):
            logger.warning(f"{symbol} market is closed at this time")
            return False

        # Then check API availability
        if self.data_fetcher and not self.data_fetcher.is_symbol_available(symbol):
            logger.warning(f"{symbol} is not available via API")
            return False

        return True

    async def calculate_asset_volatility(self, symbol: str) -> float:
        """Calculate recent price volatility for an asset"""
        try:
            # Get recent price history (last hour)
            candles = await self.data_fetcher.fetch_historical_data(
                symbol, interval=60, count=60
            )
            if candles is None or len(candles) < 10:
                return 0.0

            # Calculate price changes
            prices = candles['close'].values
            returns = np.diff(prices) / prices[:-1]
            
            # Calculate volatility (standard deviation of returns)
            volatility = np.std(returns)
            
            # Cache the result
            self.volatility_cache[symbol] = {
                'value': volatility,
                'timestamp': datetime.now(UTC)
            }
            
            return volatility
            
        except Exception as e:
            logger.error(f"Error calculating volatility for {symbol}: {e}")
            return 0.0

    async def get_best_trading_assets(self, max_assets: int = 5) -> List[Dict]:
        """Get the best assets for trading based on multiple criteria"""
        available_assets = await self.get_available_assets(force_refresh=True)
        asset_scores = []

        for symbol in available_assets:
            try:
                # Skip if we've recently found the asset unavailable
                if symbol in self.unavailable_assets_cache:
                    last_check, _ = self.unavailable_assets_cache[symbol]
                    if datetime.now(UTC) - last_check < self.unavailable_cache_validity:
                        continue

                # Get or calculate volatility
                volatility = 0.0
                if symbol in self.volatility_cache:
                    cache_data = self.volatility_cache[symbol]
                    if datetime.now(UTC) - cache_data['timestamp'] < self.volatility_cache_validity:
                        volatility = cache_data['value']
                
                if volatility == 0.0:
                    volatility = await self.calculate_asset_volatility(symbol)

                # Skip assets with too low volatility
                if volatility < self.min_volatility_threshold:
                    continue

                # Get recent quote data
                quote = await self.data_fetcher.get_price_quote(symbol)
                if not quote or "error" in quote:
                    continue

                # Calculate trading score based on multiple factors
                score = self._calculate_trading_score(
                    symbol=symbol,
                    volatility=volatility,
                    spread=quote.get('spread', float('inf')),
                    is_preferred=symbol in self.preferred_assets
                )

                asset_scores.append({
                    'symbol': symbol,
                    'score': score,
                    'volatility': volatility,
                    'spread': quote.get('spread', 0),
                    'quote': quote
                })

            except Exception as e:
                logger.error(f"Error evaluating {symbol}: {e}")
                continue

        # Sort by score and return top assets
        asset_scores.sort(key=lambda x: x['score'], reverse=True)
        return asset_scores[:max_assets]

    def _calculate_trading_score(self, symbol: str, volatility: float, 
                               spread: float, is_preferred: bool) -> float:
        """Calculate a trading score for an asset based on multiple factors"""
        # Base score from volatility
        score = volatility * 100

        # Penalize high spreads
        if spread != float('inf'):
            score = score / (1 + spread)

        # Bonus for preferred assets
        if is_preferred:
            score *= 1.2

        # Time-based preferences
        time_preferences = self.get_preferred_assets_by_time()
        if symbol in time_preferences[:3]:  # Bonus for top 3 time-preferred assets
            score *= 1.1

        return score

    def is_market_open(self, symbol: str) -> bool:
        """Check if the market is open for this symbol"""
        now = datetime.now(UTC)
        day = now.strftime('%A').lower()
        hour = now.hour

        # Always available synthetic indices
        if symbol.startswith('R_'):
            return True

        # Standard forex pairs
        if symbol.startswith('frx'):
            if day in MARKET_HOURS['forex']['standard']:
                for start, end in MARKET_HOURS['forex']['standard'][day]:
                    if start <= hour < end:
                        return True
            return False

        # Default to closed if unknown symbol
        return False