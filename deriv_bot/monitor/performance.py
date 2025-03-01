"""
Module for tracking trading performance metrics
"""
"""
Performance Tracking Module

Location: deriv_bot/monitor/performance.py

Purpose:
Tracks and analyzes trading performance metrics, including win rates,
profit/loss statistics, and prediction accuracy.

Dependencies:
- pandas: Data analysis and statistics
- deriv_bot.monitor.logger: Logging functionality

Interactions:
- Input: Trade results and performance data
- Output: Performance metrics and statistics
- Relations: Used by main trading loop for performance monitoring

Author: Trading Bot Team
Last modified: 2024-02-26
"""
import pandas as pd
import numpy as np
from deriv_bot.monitor.logger import setup_logger

logger = setup_logger(__name__)

class PerformanceTracker:
    def __init__(self):
        self.trades = []
        self.wins = 0
        self.losses = 0
        self.total_profit = 0
        self.max_drawdown = 0
        self.running_balance = 0
        self.peak_balance = 0

    def add_trade(self, trade_data):
        """
        Record a completed trade with enhanced metrics

        Args:
            trade_data: Dictionary containing trade details
        """
        try:
            # Calculate profit based on entry and exit prices
            entry_price = float(trade_data['entry_price'])
            exit_price = float(trade_data['exit_price'])
            amount = float(trade_data['amount'])

            # Calculate profit based on trade type
            if trade_data['type'] == 'CALL':
                profit = (exit_price - entry_price) * amount
            else:  # PUT
                profit = (entry_price - exit_price) * amount

            # Update running balance and track drawdown
            self.running_balance += profit
            self.peak_balance = max(self.peak_balance, self.running_balance)
            current_drawdown = (self.peak_balance - self.running_balance) / self.peak_balance if self.peak_balance > 0 else 0
            self.max_drawdown = max(self.max_drawdown, current_drawdown)

            # Add enhanced metrics to trade data
            trade_data.update({
                'profit': profit,
                'running_balance': self.running_balance,
                'drawdown': current_drawdown,
                'prediction_error': abs(trade_data['predicted_change'] - trade_data['actual_change']),
                'prediction_direction_correct': (
                    (trade_data['type'] == 'CALL' and trade_data['actual_change'] > 0) or
                    (trade_data['type'] == 'PUT' and trade_data['actual_change'] < 0)
                )
            })

            self.trades.append(trade_data)

            if profit > 0:
                self.wins += 1
            else:
                self.losses += 1

            self.total_profit += profit
            logger.info(f"Trade recorded: {trade_data}")

        except Exception as e:
            logger.error(f"Error recording trade: {str(e)}")

    def get_statistics(self):
        """Calculate and return comprehensive performance statistics"""
        try:
            total_trades = self.wins + self.losses
            win_rate = (self.wins / total_trades * 100) if total_trades > 0 else 0

            if self.trades:
                df = pd.DataFrame(self.trades)
                avg_prediction_error = df['prediction_error'].mean()
                direction_accuracy = df['prediction_direction_correct'].mean() * 100
                avg_profit_per_trade = df['profit'].mean()
                profit_factor = abs(df[df['profit'] > 0]['profit'].sum() / df[df['profit'] < 0]['profit'].sum()) if len(df[df['profit'] < 0]) > 0 else float('inf')

                # Calculate Sharpe Ratio (assuming daily returns)
                returns = df['profit'].pct_change()
                sharpe_ratio = np.sqrt(252) * (returns.mean() / returns.std()) if len(returns) > 1 else 0
            else:
                avg_prediction_error = 0
                direction_accuracy = 0
                avg_profit_per_trade = 0
                profit_factor = 0
                sharpe_ratio = 0

            stats = {
                'total_trades': total_trades,
                'wins': self.wins,
                'losses': self.losses,
                'win_rate': win_rate,
                'total_profit': self.total_profit,
                'max_drawdown': self.max_drawdown * 100,  # Convert to percentage
                'avg_prediction_error': avg_prediction_error,
                'direction_accuracy': direction_accuracy,
                'avg_profit_per_trade': avg_profit_per_trade,
                'profit_factor': profit_factor,
                'sharpe_ratio': sharpe_ratio
            }

            return stats

        except Exception as e:
            logger.error(f"Error calculating statistics: {str(e)}")
            return None

    def export_history(self, filename='trade_history.csv'):
        """Export detailed trade history to CSV file"""
        try:
            if not self.trades:
                logger.warning("No trades to export")
                return

            df = pd.DataFrame(self.trades)

            # Add derived metrics
            if len(df) > 0:
                df['cumulative_profit'] = df['profit'].cumsum()
                df['drawdown_pct'] = df['drawdown'] * 100

                # Calculate rolling metrics
                df['rolling_win_rate'] = df['profit'].rolling(window=10).apply(lambda x: (x > 0).mean() * 100)
                df['rolling_avg_profit'] = df['profit'].rolling(window=10).mean()

            df.to_csv(filename, index=False)
            logger.info(f"Trade history exported to {filename}")

        except Exception as e:
            logger.error(f"Error exporting trade history: {str(e)}")