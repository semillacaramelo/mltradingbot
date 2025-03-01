"""
Asset Selector Module Unit Tests

Location: tests/test_asset_selector.py

Purpose:
Unit tests for the asset selection logic including market hours,
asset availability, and selection strategies.

Dependencies:
- unittest: Testing framework
- datetime: Time handling
- deriv_bot.utils.asset_selector: Module being tested

Interactions:
- Input: Test market conditions and time data
- Output: Asset selection validations
- Relations: Validates asset selection functionality

Author: Trading Bot Team
Last modified: 2024-02-26
"""

import unittest
import datetime
from unittest.mock import Mock, patch
from datetime import datetime, time
from zoneinfo import ZoneInfo

from deriv_bot.utils.asset_selector import AssetSelector, ALWAYS_AVAILABLE_ASSETS, MARKET_HOURS

class TestAssetSelector(unittest.TestCase):
    """Pruebas unitarias para la clase AssetSelector."""
    
    def setUp(self):
        """Configuración inicial para las pruebas."""
        self.mock_data_fetcher = Mock()
        self.mock_data_fetcher.is_symbol_available = Mock(return_value=True)
        self.selector = AssetSelector(data_fetcher=self.mock_data_fetcher)
        
    def test_is_market_open_forex(self):
        """Prueba para verificar si el mercado de Forex está abierto."""
        # Miércoles a las 12:00 UTC (día de semana normal)
        wednesday_noon = datetime(2025, 2, 26, 12, 0, 0, tzinfo=ZoneInfo("UTC"))
        self.assertTrue(self.selector.is_market_open("frxEURUSD", wednesday_noon))
        
        # Sábado a las 12:00 UTC (fin de semana, mercado cerrado)
        saturday_noon = datetime(2025, 3, 1, 12, 0, 0, tzinfo=ZoneInfo("UTC"))
        self.assertFalse(self.selector.is_market_open("frxEURUSD", saturday_noon))
        
    def test_is_market_open_stocks(self):
        """Prueba para verificar si el mercado de acciones está abierto."""
        # Lunes a las 14:30 UTC (9:30 AM en NY, mercado abierto)
        monday_ny_market_open = datetime(2025, 2, 24, 14, 30, 0, tzinfo=ZoneInfo("UTC"))
        self.assertTrue(self.selector.is_market_open("OTC_SPX", monday_ny_market_open))
        
        # Lunes a las 21:00 UTC (16:00 en NY, mercado cerrado)
        monday_ny_market_closed = datetime(2025, 2, 24, 21, 0, 0, tzinfo=ZoneInfo("UTC"))
        self.assertFalse(self.selector.is_market_open("OTC_SPX", monday_ny_market_closed))
        
    def test_is_market_open_crypto(self):
        """Prueba para verificar si el mercado de criptomonedas está abierto (24/7)."""
        # Miércoles a las 12:00 UTC
        wednesday_noon = datetime(2025, 2, 26, 12, 0, 0, tzinfo=ZoneInfo("UTC"))
        self.assertTrue(self.selector.is_market_open("frxBTCUSD", wednesday_noon))
        
        # Sábado a las 12:00 UTC (fin de semana, pero cripto está abierto 24/7)
        saturday_noon = datetime(2025, 3, 1, 12, 0, 0, tzinfo=ZoneInfo("UTC"))
        self.assertTrue(self.selector.is_market_open("frxBTCUSD", saturday_noon))
        
    def test_verify_asset_availability_market_closed(self):
        """Prueba para verificar la disponibilidad cuando el mercado está cerrado."""
        # Parchear is_market_open para que devuelva False
        with patch.object(self.selector, 'is_market_open', return_value=False):
            # Incluso si la API dice que está disponible, si el mercado está cerrado, debería retornar False
            self.assertFalse(self.selector.verify_asset_availability("frxEURUSD"))
            # No debería llamar a is_symbol_available
            self.mock_data_fetcher.is_symbol_available.assert_not_called()
    
    def test_verify_asset_availability_market_open(self):
        """Prueba para verificar la disponibilidad cuando el mercado está abierto."""
        # Parchear is_market_open para que devuelva True
        with patch.object(self.selector, 'is_market_open', return_value=True):
            # Configurar el mock para que is_symbol_available devuelva True
            self.mock_data_fetcher.is_symbol_available.return_value = True
            self.assertTrue(self.selector.verify_asset_availability("frxEURUSD"))
            self.mock_data_fetcher.is_symbol_available.assert_called_once_with("frxEURUSD")
            
            # Configurar el mock para que is_symbol_available devuelva False
            self.mock_data_fetcher.is_symbol_available.reset_mock()
            self.mock_data_fetcher.is_symbol_available.return_value = False
            self.assertFalse(self.selector.verify_asset_availability("frxEURUSD"))
            self.mock_data_fetcher.is_symbol_available.assert_called_once_with("frxEURUSD")
    
    def test_get_available_assets(self):
        """Prueba para obtener la lista de activos disponibles."""
        # Parchear verify_asset_availability para controlar qué activos están disponibles
        with patch.object(self.selector, 'verify_asset_availability', side_effect=lambda asset: asset in ["frxEURUSD", "R_10"]):
            available = self.selector.get_available_assets(force_refresh=True)
            self.assertEqual(len(available), 2)
            self.assertIn("frxEURUSD", available)
            self.assertIn("R_10", available)
    
    def test_get_preferred_assets_by_time(self):
        """Prueba para obtener los activos preferidos según la hora."""
        # 05:00 UTC (madrugada, preferencia por mercados asiáticos)
        morning_time = time(5, 0)
        morning_assets = self.selector.get_preferred_assets_by_time(morning_time)
        self.assertEqual(morning_assets[0], "frxUSDJPY")  # El primero debería ser USD/JPY
        
        # 14:00 UTC (tarde en Europa, mercados europeos y apertura de US)
        afternoon_time = time(14, 0)
        afternoon_assets = self.selector.get_preferred_assets_by_time(afternoon_time)
        self.assertEqual(afternoon_assets[0], "frxEURUSD")  # El primero debería ser EUR/USD
        
        # 20:00 UTC (noche en Europa, mercados de US activos)
        evening_time = time(20, 0)
        evening_assets = self.selector.get_preferred_assets_by_time(evening_time)
        self.assertEqual(evening_assets[0], "frxEURUSD")  # EUR/USD sigue siendo preferido
    
    def test_select_asset_preferred_available(self):
        """Prueba para seleccionar un activo cuando el preferido está disponible."""
        # Parchear get_available_assets para controlar qué activos están disponibles
        with patch.object(self.selector, 'get_available_assets', return_value=["frxEURUSD", "frxGBPUSD", "R_10"]):
            # Seleccionar con un activo preferido específico que está disponible
            selected = self.selector.select_asset(preferred_asset="frxEURUSD")
            self.assertEqual(selected, "frxEURUSD")
    
    def test_select_asset_preferred_not_available(self):
        """Prueba para seleccionar un activo cuando el preferido no está disponible."""
        # Parchear get_available_assets y get_preferred_assets_by_time
        with patch.object(self.selector, 'get_available_assets', return_value=["frxGBPUSD", "R_10"]):
            with patch.object(self.selector, 'get_preferred_assets_by_time', return_value=["frxEURUSD", "frxGBPUSD", "OTC_SPX"]):
                # Seleccionar con un activo preferido específico que NO está disponible
                selected = self.selector.select_asset(preferred_asset="frxEURUSD")
                # Debería seleccionar frxGBPUSD que está en la lista de preferidos y disponible
                self.assertEqual(selected, "frxGBPUSD")
    
    def test_select_asset_fallback(self):
        """Prueba para seleccionar un activo de fallback cuando los preferidos no están disponibles."""
        # Parchear get_available_assets y get_preferred_assets_by_time
        with patch.object(self.selector, 'get_available_assets', return_value=["R_10", "R_25"]):
            with patch.object(self.selector, 'get_preferred_assets_by_time', return_value=["frxEURUSD", "frxGBPUSD", "OTC_SPX"]):
                # Ningún activo preferido está disponible
                selected = self.selector.select_asset()
                # Debería seleccionar R_10 que está en la lista de fallback y disponible
                self.assertEqual(selected, "R_10")
    
    def test_select_asset_no_available(self):
        """Prueba para seleccionar un activo cuando ninguno está disponible."""
        # Parchear get_available_assets para que devuelva una lista vacía
        with patch.object(self.selector, 'get_available_assets', return_value=[]):
            selected = self.selector.select_asset()
            # Debería seleccionar el primer activo de la lista de fallback
            self.assertEqual(selected, ALWAYS_AVAILABLE_ASSETS[0])
    
    def test_select_asset_user_preferences(self):
        """Prueba para seleccionar un activo considerando las preferencias del usuario."""
        # Configurar preferencias del usuario
        self.selector.preferred_assets = ["frxGBPUSD", "frxXAUUSD"]
        
        # Parchear get_available_assets para controlar qué activos están disponibles
        with patch.object(self.selector, 'get_available_assets', return_value=["frxEURUSD", "frxGBPUSD", "R_10"]):
            selected = self.selector.select_asset()
            # Debería seleccionar frxGBPUSD que está en las preferencias del usuario y disponible
            self.assertEqual(selected, "frxGBPUSD")

if __name__ == '__main__':
    unittest.main()
