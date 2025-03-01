"""
Configuration management module
"""
"""
Configuration Management Module

Location: deriv_bot/utils/config.py

Purpose:
Manages bot configuration including environment variables,
trading parameters, and runtime settings.

Dependencies:
- os: Environment variable access
- dotenv: Environment file loading
- deriv_bot.monitor.logger: Logging functionality

Interactions:
- Input: Environment variables and config files
- Output: Configuration parameters
- Relations: Used by all modules for configuration

Author: Trading Bot Team
Last modified: 2024-02-26
"""
import os
import logging
import json
import time
from pathlib import Path
from dotenv import load_dotenv
from deriv_bot.monitor.logger import setup_logger

logger = setup_logger(__name__)

class Config:
    def __init__(self):
        self.load_environment()
        self.trading_config = {
            'symbol': os.getenv('DEFAULT_SYMBOL', 'frxEURUSD'),
            'stake_amount': float(os.getenv('DEFAULT_STAKE_AMOUNT', '10.0')),
            'duration': int(os.getenv('DEFAULT_DURATION', '30')),  # seconds - más rápido para demo
            'max_position_size': float(os.getenv('MAX_POSITION_SIZE', '100.0')),
            'max_daily_loss': float(os.getenv('MAX_DAILY_LOSS', '50.0'))
        }
        self.environment = os.getenv('DERIV_BOT_ENV', 'demo').lower()

        # Verificar que el ambiente sea válido
        if self.environment not in ['demo', 'real']:
            logger.warning(f"Invalid environment '{self.environment}', defaulting to 'demo'")
            self.environment = 'demo'
            os.environ['DERIV_BOT_ENV'] = 'demo'

        # Verificar que los tokens necesarios estén presentes
        self._verify_tokens()

        logger.info(f"Trading bot initialized in {self.environment.upper()} mode")

    def _verify_tokens(self):
        """Verify that necessary API tokens are available"""
        env = self.environment
        token_key = f"DERIV_API_TOKEN_{env.upper()}"
        token_value = os.getenv(token_key)

        if not token_value:
            logger.warning(f"Missing {token_key} for {env} environment")

            # Si estamos en modo real pero falta el token real, cambiar a demo
            if env == 'real':
                demo_token = os.getenv('DERIV_API_TOKEN_DEMO')
                if demo_token:
                    logger.warning("Missing real token, switching to DEMO mode")
                    self.environment = 'demo'
                    os.environ['DERIV_BOT_ENV'] = 'demo'

        # Verificar confirmación para modo real
        if env == 'real':
            if os.getenv('DERIV_REAL_MODE_CONFIRMED', '').lower() != 'yes':
                logger.warning("Real mode requires confirmation (DERIV_REAL_MODE_CONFIRMED=yes), switching to DEMO mode")
                self.environment = 'demo'
                os.environ['DERIV_BOT_ENV'] = 'demo'

    def load_environment(self):
        """Load environment variables"""
        try:
            # Buscar archivo .env en varias ubicaciones posibles
            env_paths = [
                Path(".env"),  # Directorio actual
                Path("../.env"),  # Directorio padre
                Path(os.path.join(os.getcwd(), '.env')),  # Ruta absoluta
            ]

            env_loaded = False
            for env_path in env_paths:
                if env_path.exists():
                    load_dotenv(dotenv_path=env_path)
                    logger.info(f"Loaded environment from {env_path}")
                    env_loaded = True
                    break

            if not env_loaded:
                logger.warning("No .env file found, using existing environment variables")

            required_vars = [
                'DERIV_API_TOKEN_DEMO',
                'DERIV_API_TOKEN_REAL',
                'DERIV_BOT_ENV'
            ]

            missing_vars = []
            for var in required_vars:
                if not os.getenv(var):
                    missing_vars.append(var)

            if missing_vars:
                logger.warning(f"Missing environment variables: {', '.join(missing_vars)}")
                # Si falta el token de demo, esto es crítico
                if 'DERIV_API_TOKEN_DEMO' in missing_vars:
                    logger.error("DERIV_API_TOKEN_DEMO is required for operation")

        except Exception as e:
            logger.error(f"Error loading environment variables: {str(e)}")

    def get_api_token(self):
        """Get appropriate API token based on environment"""
        env = self.environment
        token_key = f"DERIV_API_TOKEN_{env.upper()}"
        token = os.getenv(token_key)

        if not token:
            logger.error(f"No API token found for {env} environment")
            if env == 'real' and os.getenv('DERIV_API_TOKEN_DEMO'):
                logger.warning("Switching to demo mode due to missing real token")
                self.environment = 'demo'
                os.environ['DERIV_BOT_ENV'] = 'demo'
                return os.getenv('DERIV_API_TOKEN_DEMO')
            else:
                raise ValueError(f"Missing API token for {env} environment. Set {token_key} in .env file")

        return token

    def set_environment(self, env_mode):
        """Set trading environment (demo/real) with validation"""
        env_mode = env_mode.lower()
        if env_mode not in ['demo', 'real']:
            logger.error(f"Invalid environment mode: {env_mode}")
            return False

        # Validate required tokens
        token = os.getenv(f'DERIV_API_TOKEN_{env_mode.upper()}')
        if not token:
            logger.error(f"Missing API token for {env_mode} environment")
            return False

        # Additional validation for real mode
        if env_mode == 'real':
            confirm = os.getenv('DERIV_REAL_MODE_CONFIRMED', 'no').lower() == 'yes'
            if not confirm:
                logger.error("Real mode not confirmed in environment settings")
                return False

        self.environment = env_mode
        logger.info(f"Environment set to {env_mode.upper()} mode")
        return True

    def get_environment(self):
        """Get current trading environment"""
        return self.environment

    def is_demo(self):
        """Check if running in demo mode"""
        return self.environment == 'demo'

    def update_trading_config(self, **kwargs):
        """
        Update trading configuration parameters

        Args:
            **kwargs: Configuration parameters to update
        """
        for key, value in kwargs.items():
            if key in self.trading_config:
                # Validar tipos de datos
                if isinstance(self.trading_config[key], float):
                    try:
                        value = float(value)
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid value for {key}: {value}. Must be a number.")
                        continue
                elif isinstance(self.trading_config[key], int):
                    try:
                        value = int(value)
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid value for {key}: {value}. Must be an integer.")
                        continue

                # Validar límites
                if key == 'stake_amount' and value <= 0:
                    logger.warning(f"Invalid stake amount: {value}. Must be positive.")
                    continue
                elif key == 'duration' and value < 5:
                    logger.warning(f"Invalid duration: {value}. Must be at least 5 seconds.")
                    continue

                # Actualizar valor
                old_value = self.trading_config[key]
                self.trading_config[key] = value
                logger.info(f"Updated {key}: {old_value} -> {value}")
            else:
                logger.warning(f"Unknown configuration parameter: {key}")

        # Guardar configuración actualizada
        self._save_state()

    def _save_state(self):
        """Save current state to file for persistence"""
        try:
            state_dir = Path("data")
            state_dir.mkdir(exist_ok=True)

            state_file = state_dir / "config_state.json"
            state = {
                "environment": self.environment,
                "trading_config": self.trading_config,
                "timestamp": int(time.time())
            }

            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)

            logger.debug(f"Saved configuration state to {state_file}")
        except Exception as e:
            logger.error(f"Error saving configuration state: {str(e)}")

    def load_saved_state(self):
        """Load previously saved state"""
        try:
            state_file = Path("data/config_state.json")
            if not state_file.exists():
                logger.debug("No saved state found")
                return False

            with open(state_file, 'r') as f:
                state = json.load(f)

            # Cargar entorno guardado
            saved_env = state.get("environment")
            if saved_env in ['demo', 'real']:
                # No cambiar a real si faltan condiciones necesarias
                if saved_env == 'real':
                    if not os.getenv('DERIV_API_TOKEN_REAL'):
                        logger.warning("Cannot restore REAL mode: Missing DERIV_API_TOKEN_REAL")
                    elif os.getenv('DERIV_REAL_MODE_CONFIRMED', '').lower() != 'yes':
                        logger.warning("Cannot restore REAL mode: DERIV_REAL_MODE_CONFIRMED must be 'yes'")
                    else:
                        self.environment = saved_env
                        os.environ['DERIV_BOT_ENV'] = saved_env
                else:
                    self.environment = saved_env
                    os.environ['DERIV_BOT_ENV'] = saved_env

            # Cargar configuración de trading
            saved_config = state.get("trading_config", {})
            for key, value in saved_config.items():
                if key in self.trading_config:
                    self.trading_config[key] = value

            logger.info(f"Loaded saved configuration (environment: {self.environment.upper()})")
            return True

        except Exception as e:
            logger.error(f"Error loading saved state: {str(e)}")
            return False