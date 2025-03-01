"""
Module for setting up logging configuration
"""
"""
Logging Configuration Module

Location: deriv_bot/monitor/logger.py

Purpose:
Provides centralized logging configuration and setup for the entire trading bot.
Implements custom formatters and handlers for different logging requirements.

Dependencies:
- logging: Python's built-in logging module

Interactions:
- Input: Logging level and format configurations
- Output: Configured logger instances
- Relations: Used by all other modules for logging

Author: Trading Bot Team
Last modified: 2024-02-26
"""
import logging
import sys
from logging.handlers import RotatingFileHandler

def setup_logger(name):
    """
    Set up logger with both file and console handlers
    
    Args:
        name: Logger name (usually __name__)
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler
    file_handler = RotatingFileHandler(
        'trading_bot.log',
        maxBytes=1024 * 1024,  # 1MB
        backupCount=5
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger
