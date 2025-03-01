"""
Test API connectivity to Deriv.com
Verifies API tokens, connection, and basic functionality
"""

import os
import sys
import asyncio
import logging
from datetime import datetime
from time import perf_counter
import colorama
from colorama import Fore, Style
from deriv_bot.utils.config import Config
from deriv_bot.data.deriv_connector import DerivConnector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('api_test')

# Initialize colorama for Windows
colorama.init()

def print_status(message, success=True):
    """Print colored status messages"""
    if success:
        print(f"{Fore.GREEN}[+] {message}{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}[-] {message}{Style.RESET_ALL}")

def check_env_vars():
    """Verify required environment variables"""
    print("\nVerifying environment variables:")
    
    required_vars = {
        'DERIV_API_TOKEN_DEMO': os.getenv('DERIV_API_TOKEN_DEMO'),
        'DERIV_API_TOKEN_REAL': os.getenv('DERIV_API_TOKEN_REAL'),
        'DERIV_BOT_ENV': os.getenv('DERIV_BOT_ENV'),
        'APP_ID': os.getenv('APP_ID')
    }
    
    all_configured = True
    for var, value in required_vars.items():
        if value:
            print(f"- {var}: [OK] Configured")
        else:
            print(f"- {var}: [MISSING] Not configured")
            all_configured = False
            
    return all_configured

async def test_api_connectivity():
    """Main API connectivity test"""
    connector = None
    try:
        print("\nInitializing configuration...")
        config = Config()
        
        # Add debug info
        print(f"Environment: {config.environment}")
        print(f"APP_ID: {os.getenv('APP_ID')}")
        token = config.get_api_token()
        print("API Token: " + "*" * (len(token) if token else 0))
        print(f"Token length: {len(token) if token else 0} characters")
        print(f"Using endpoint: ws.binaryws.com/websockets/v3")

        connector = DerivConnector(config)
        
        # Test connection with timeout and debug info
        print(f"\nTesting API connection in {config.environment.upper()} mode...")
        start_time = perf_counter()
        
        # Add shorter timeouts for initial attempt
        try:
            print("Initiating connection...")
            print("Attempting ping...")
            connected = await asyncio.wait_for(connector.connect(), timeout=10.0)
            print("Connection attempt completed")
            
            if not connected:
                print("\nDiagnostics:")
                print(f"- APP_ID: {connector.app_id}")
                print(f"- Token valid: {'Yes' if len(token) > 10 else 'No'}")
                print(f"- Environment: {config.environment}")
                print("Initial connection failed, retrying once...")
                await asyncio.sleep(2)
                connected = await asyncio.wait_for(connector.connect(), timeout=10.0)
                
        except asyncio.TimeoutError:
            print_status("Connection attempts timed out - possible network issue or invalid credentials", False)
            return False
        except Exception as e:
            print_status(f"Connection error: {str(e)}", False)
            return False
            
        connection_time = perf_counter() - start_time
        
        if not connected:
            print_status("Failed to connect to Deriv API", False)
            return False
            
        print_status(f"Successfully connected to Deriv API ({connection_time:.2f}s)")
        
        # Test connection status explicitly with debug info
        print("\nVerifying connection status...")
        try:
            is_connected = await asyncio.wait_for(connector.check_connection(), timeout=5.0)
            if not is_connected:
                print_status("Connection check failed after initial connection", False)
                return False
        except asyncio.TimeoutError:
            print_status("Connection check timed out", False)
            return False
            
        print_status("Connection check passed")

        # Test server time with timeout
        print("\nVerifying server time...")
        try:
            time_response = await asyncio.wait_for(connector.get_server_time(), timeout=10.0)
        except asyncio.TimeoutError:
            print_status("Server time request timed out", False)
            return False

        if not time_response or 'time' not in time_response:
            print_status("Failed to get server time", False)
            return False
            
        server_time = datetime.fromtimestamp(time_response['time'])
        print_status(f"Server time: {server_time}")
        
        # Test basic market data with timeout
        print("\nTesting market data access...")
        symbol = "R_100"  # Volatility 100 Index
        try:
            candles = await asyncio.wait_for(connector.get_candles(symbol, count=10), timeout=10.0)
        except asyncio.TimeoutError:
            print_status(f"Market data request timed out for {symbol}", False)
            return False

        if not candles:
            print_status(f"Failed to get market data for {symbol}", False)
            return False
            
        print_status(f"Successfully retrieved {len(candles)} candles for {symbol}")
        
        return True
        
    except Exception as e:
        logger.error("Error during API test", exc_info=True)
        print_status(f"Error during test: {str(e)}", False)
        return False
        
    finally:
        if connector:
            try:
                await asyncio.wait_for(connector.disconnect(), timeout=5.0)
            except asyncio.TimeoutError:
                print_status("Warning: Disconnect operation timed out", False)
            except Exception as e:
                print_status(f"Warning: Error during disconnect: {str(e)}", False)

async def main():
    """Main entry point"""
    print("Starting Deriv API connectivity test...")
    print("===============================================")
    
    # Add timeout to entire test
    try:
        # Overall timeout of 60 seconds for the entire test
        async with asyncio.timeout(60):
            config = Config()
            print(f"Mode: {config.environment.upper()}\n")
            
            print("===== API CONNECTIVITY TEST =====\n")
            
            # Check environment variables
            env_ok = check_env_vars()
            if not env_ok:
                print("\nWarning: Some environment variables are not configured.")
                response = input("Continue anyway? (y/n): ")
                if response.lower() != 'y':
                    return
            
            # Run connectivity test
            success = await test_api_connectivity()
            
            print("\n===============================================")
            if success:
                print_status("All API connectivity tests passed!")
            else:
                print_status("Some API connectivity tests failed!", False)
                sys.exit(1)
    except asyncio.TimeoutError:
        print_status("Test timed out after 60 seconds", False)
        sys.exit(1)
    except Exception as e:
        print_status(f"Unexpected error: {str(e)}", False)
        logger.error("Unexpected error during test", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        sys.exit(1)